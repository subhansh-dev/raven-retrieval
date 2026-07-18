"""Contextual Retrieval (Anthropic, 2024).

Prepends document-level context to each chunk before embedding/scoring.
Reduces the "lost in the middle" problem where chunks lose document context.

Two flavors:
1. Contextual BM25: prepend context, then BM25 index
2. Contextual Embeddings: prepend context, then dense embed

Reference: Anthropic, "Contextual Retrieval" (2024)
           https://www.anthropic.com/news/contextual-retrieval
"""

import numpy as np
import hashlib
import logging
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class ContextualChunker:
    """Chunks documents with contextual prefixes.

    Instead of embedding "The cat sat on the mat" alone,
    embeds "Document: Alice in Wonderland. The cat sat on the mat."

    This preserves document-level context that would otherwise be lost.
    """

    def __init__(self, chunk_size=200, chunk_overlap=50, context_prefix_maxlen=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_prefix_maxlen = context_prefix_maxlen

    def _extract_context(self, doc):
        """Extract contextual prefix from document metadata."""
        title = doc.get("title", "")
        text = doc.get("text", "")

        # Use title if available
        if title:
            context = f"Document title: {title}."
        else:
            # Extract first sentence as context
            first_sent = text.split(". ")[0] if ". " in text else text[:self.context_prefix_maxlen]
            context = f"Document begins: {first_sent}."

        # Truncate if too long
        words = context.split()
        if len(words) > self.context_prefix_maxlen:
            context = " ".join(words[:self.context_prefix_maxlen]) + "..."

        return context

    def chunk_with_context(self, doc, doc_id=None):
        """Chunk a document with contextual prefixes on each chunk."""
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        context = self._extract_context(doc)

        tokens = full_text.split()
        chunks = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = " ".join(tokens[start:end])

            # Prepend context
            contextual_text = f"{context} {chunk_text}"

            chunk_id = f"{doc_id}::l0::{idx}" if doc_id else f"l0::{idx}"
            chunks.append({
                "id": chunk_id,
                "text": contextual_text,  # Context-enriched text
                "raw_text": chunk_text,    # Original chunk without context
                "context": context,
                "token_count": end - start,
                "context_token_count": len(context.split()),
                "level": 0,
                "doc_id": doc_id,
            })

            start += self.chunk_size - self.chunk_overlap
            idx += 1

        return chunks

    def chunk_corpus_with_context(self, corpus):
        """Chunk entire corpus with contextual prefixes."""
        all_chunks = []

        if isinstance(corpus, dict):
            corpus_iter = [(did, d) for did, d in corpus.items()]
        else:
            corpus_iter = [(doc["_id"], doc) for doc in corpus]

        for doc_id, doc in corpus_iter:
            chunks = self.chunk_with_context(doc, doc_id=doc_id)
            all_chunks.extend(chunks)

        return all_chunks


class ContextualDenseRetriever:
    """Dense retrieval with contextual chunk embeddings.

    Each chunk is prefixed with document context before being embedded.
    This helps the embedding model understand WHERE the chunk comes from.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50):
        self.model = SentenceTransformer(model_name)
        self.chunker = ContextualChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_raw_texts = []
        self.corpus_embeddings = None

    def index(self, corpus):
        """Index with contextual chunk embeddings."""
        chunks = self.chunker.chunk_corpus_with_context(corpus)

        self.corpus_ids = [c["id"] for c in chunks]
        self.corpus_texts = [c["text"] for c in chunks]      # Context-enriched
        self.corpus_raw_texts = [c["raw_text"] for c in chunks]  # Original

        # Embed context-enriched texts
        embeddings = self.model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = np.array(embeddings)

        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / norms

        return self

    def retrieve(self, query, top_k=10):
        """Retrieve using contextual embeddings."""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / query_norm

        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()

        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class ContextualBM25Retriever:
    """BM25 retrieval with contextual chunk text.

    BM25 over context-enriched chunks — the extra context words
    help bridge vocabulary gaps between queries and documents.
    """

    def __init__(self, chunk_size=200, chunk_overlap=50):
        self.chunker = ContextualChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.corpus_ids = []
        self.corpus_texts = []
        self.bm25 = None

    def index(self, corpus):
        chunks = self.chunker.chunk_corpus_with_context(corpus)

        self.corpus_ids = [c["id"] for c in chunks]
        self.corpus_texts = [c["text"] for c in chunks]

        tokenized = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        return self

    def retrieve(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class ContextualHybridRetriever:
    """Hybrid retrieval combining contextual BM25 + contextual dense + RRF.

    The best of both worlds: BM25 with expanded context AND dense embeddings
    with document-level understanding.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60):
        self.dense = ContextualDenseRetriever(model_name, chunk_size, chunk_overlap)
        self.bm25 = ContextualBM25Retriever(chunk_size, chunk_overlap)
        self.rrf_k = rrf_k
        self._indexed = False

    def index(self, corpus):
        self.dense.index(corpus)
        self.bm25.index(corpus)
        self._indexed = True
        return self

    def reciprocal_rank_fusion(self, rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_k=10):
        dense_results = self.dense.retrieve(query, top_k=top_k * 3)
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 3)
        fused = self.reciprocal_rank_fusion([dense_results, bm25_results], k=self.rrf_k)
        return fused[:top_k]
