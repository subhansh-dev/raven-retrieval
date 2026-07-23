"""Contextual Retrieval (Anthropic, 2024).

Prepends document-level context to each chunk before embedding/scoring.
Reduces the "lost in the middle" problem where chunks lose document context.

Three flavors:
1. ContextualDenseRetriever: prepend context, then dense embed
2. ContextualBM25Retriever: prepend context, then BM25 index
3. ContextualHybridRetriever: both + RRF

Reference: Anthropic, "Contextual Retrieval" (2024)
           https://www.anthropic.com/news/contextual-retrieval
"""

import numpy as np
import logging

from ..utils import aggregate_doc_scores, reciprocal_rank_fusion, l2_normalize, iter_corpus, full_doc_text, tokenize_for_bm25

logger = logging.getLogger(__name__)


class ContextualChunker:
    """Chunks documents with contextual prefixes.

    Instead of embedding "The cat sat on the mat" alone,
    embeds "Document title: Alice in Wonderland. The cat sat on the mat."
    """

    def __init__(self, chunk_size=200, chunk_overlap=50, context_prefix_maxlen=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_prefix_maxlen = context_prefix_maxlen

    def _extract_context(self, doc):
        """Extract contextual prefix from document metadata."""
        title = doc.get("title", "")
        text = doc.get("text", "")

        if title:
            context = f"Document title: {title}."
        else:
            first_sent = text.split(". ")[0] if ". " in text else text[:self.context_prefix_maxlen]
            context = f"Document begins: {first_sent}."

        words = context.split()
        if len(words) > self.context_prefix_maxlen:
            context = " ".join(words[:self.context_prefix_maxlen]) + "..."

        return context

    def chunk_with_context(self, doc, doc_id=None):
        """Chunk a document with contextual prefixes on each chunk."""
        full_text = full_doc_text(doc)
        context = self._extract_context(doc)

        tokens = full_text.split()
        chunks = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            contextual_text = f"{context} {chunk_text}"

            chunk_id = f"{doc_id}::l0::{idx}" if doc_id else f"l0::{idx}"
            chunks.append({
                "id": chunk_id,
                "text": contextual_text,
                "raw_text": chunk_text,
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
        for doc_id, doc in iter_corpus(corpus):
            all_chunks.extend(self.chunk_with_context(doc, doc_id=doc_id))
        return all_chunks


class ContextualDenseRetriever:
    """Dense retrieval with contextual chunk embeddings."""

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, model=None):
        if model is not None:
            self.model = model
        else:
            from sentence_transformers import SentenceTransformer
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
        self.corpus_texts = [c["text"] for c in chunks]      # context-enriched
        self.corpus_raw_texts = [c["raw_text"] for c in chunks]  # original

        embeddings = self.model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = l2_normalize(np.array(embeddings))

        return self

    def retrieve(self, query, top_k=10):
        """Retrieve using contextual embeddings."""
        query_embedding = l2_normalize(np.array(self.model.encode([query])))
        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        return aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k)


class ContextualBM25Retriever:
    """BM25 retrieval with contextual chunk text."""

    def __init__(self, chunk_size=200, chunk_overlap=50):
        self.chunker = ContextualChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.corpus_ids = []
        self.corpus_texts = []
        self.bm25 = None

    def index(self, corpus):
        from rank_bm25 import BM25Okapi
        chunks = self.chunker.chunk_corpus_with_context(corpus)

        self.corpus_ids = [c["id"] for c in chunks]
        self.corpus_texts = [c["text"] for c in chunks]

        tokenized = [tokenize_for_bm25(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        return self

    def retrieve(self, query, top_k=10):
        tokenized_query = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        return aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k)


class ContextualHybridRetriever:
    """Hybrid retrieval combining contextual BM25 + contextual dense + RRF."""

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60, model=None):
        self.dense = ContextualDenseRetriever(model_name, chunk_size, chunk_overlap, model=model)
        self.bm25 = ContextualBM25Retriever(chunk_size, chunk_overlap)
        self.rrf_k = rrf_k
        self._indexed = False

    def index(self, corpus):
        self.dense.index(corpus)
        self.bm25.index(corpus)
        self._indexed = True
        return self

    def retrieve(self, query, top_k=10):
        dense_results = self.dense.retrieve(query, top_k=top_k * 3)
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 3)
        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=self.rrf_k)
        return fused[:top_k]
