"""Cross-Encoder Reranker for second-stage retrieval.

Cross-encoders process (query, document) pairs jointly through BERT,
producing much more accurate relevance scores than bi-encoders.
The tradeoff: they're too slow for first-stage retrieval over millions
of docs, but perfect for reranking top-k candidates.

Pipeline: Bi-encoder (fast, top-100) → Cross-encoder (accurate, top-10)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieval results using a cross-encoder model.

    Cross-encoders see query and document together, enabling deep
    cross-attention between query and document tokens. This is
    strictly more powerful than bi-encoder cosine similarity.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    (6-layer cross-encoder, fast and accurate)
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            self._model = "failed"

    def rerank(self, query, documents, doc_ids=None):
        """Rerank documents for a query.

        Args:
            query: query string
            documents: list of document texts
            doc_ids: optional list of document IDs

        Returns: list of (doc_id, score) sorted by relevance
        """
        if not documents:
            return []

        self._load_model()
        if self._model == "failed":
            # Fallback: return as-is
            if doc_ids:
                return [(did, 0.0) for did in doc_ids]
            return [(str(i), 0.0) for i in range(len(documents))]

        # Create query-document pairs
        pairs = [(query, doc[:self.max_length * 4]) for doc in documents]

        # Score with cross-encoder
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair with IDs and sort
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        scored = list(zip(doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def rerank_from_retrieval(self, query, retrieval_results, corpus, top_k=10):
        """Rerank results from a first-stage retriever.

        Args:
            query: query string
            retrieval_results: list of (doc_id, score) from first stage
            corpus: dict mapping doc_id -> {"title": ..., "text": ...}
            top_k: number of results to return after reranking

        Returns: list of (doc_id, reranked_score) sorted by relevance
        """
        doc_ids = [did for did, _ in retrieval_results]
        documents = []

        for did in doc_ids:
            if isinstance(corpus, dict) and did in corpus:
                doc = corpus[did]
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            else:
                text = str(did)  # Fallback
            documents.append(text)

        reranked = self.rerank(query, documents, doc_ids)
        return reranked[:top_k]


class TwoStageRetriever:
    """Complete two-stage retrieval pipeline.

    Stage 1: Fast bi-encoder retrieval (dense, BM25, hybrid, etc.)
    Stage 2: Cross-encoder reranking

    This is the standard architecture for production retrieval systems.
    """

    def __init__(self, first_stage_retriever, reranker=None,
                 first_stage_k=100, final_k=10):
        self.first_stage = first_stage_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.first_stage_k = first_stage_k
        self.final_k = final_k
        self._corpus = None

    def set_corpus(self, corpus):
        """Set the corpus for reranking (need document texts)."""
        self._corpus = corpus

    def retrieve(self, query, top_k=None):
        """Two-stage retrieval: fast retrieve → rerank."""
        if top_k is None:
            top_k = self.final_k

        # Stage 1: fast retrieval
        first_results = self.first_stage.retrieve(query, top_k=self.first_stage_k)

        if self._corpus is None:
            # No corpus available for reranking, return first stage results
            return first_results[:top_k]

        # Stage 2: cross-encoder reranking
        reranked = self.reranker.rerank_from_retrieval(
            query, first_results, self._corpus, top_k=top_k
        )

        return reranked
