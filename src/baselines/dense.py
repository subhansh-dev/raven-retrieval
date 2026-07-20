"""Naive dense retrieval: SBERT embeddings + cosine similarity.

Chunks are scored individually, then aggregated to document level by
max-pooling chunk scores (standard practice for chunked RAG).
"""

import numpy as np
import logging

from ..utils import chunk_corpus, aggregate_doc_scores, l2_normalize

logger = logging.getLogger(__name__)


class DenseRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, model=None):
        if model is not None:
            self.model = model  # dependency injection for tests
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None

    def index(self, corpus):
        self.corpus_ids, self.corpus_texts = chunk_corpus(
            corpus, self.chunk_size, self.chunk_overlap
        )
        embeddings = self.model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = l2_normalize(np.array(embeddings))
        return self

    def retrieve(self, query, top_k=10):
        query_embedding = l2_normalize(np.array(self.model.encode([query])))
        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        return aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k)

    def retrieve_batch(self, queries, top_k=10):
        """Batch retrieval — returns DOC-aggregated results (fixed: the old
        version returned raw chunk IDs, which breaks BEIR evaluation)."""
        query_embeddings = l2_normalize(np.array(
            self.model.encode(queries, show_progress_bar=True, batch_size=64)
        ))
        all_results = []
        for i in range(len(queries)):
            scores = np.dot(self.corpus_embeddings, query_embeddings[i]).flatten()
            all_results.append(aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k))
        return all_results
