"""Hybrid retrieval: BM25 + dense SBERT fused with Reciprocal Rank Fusion."""

import numpy as np
import logging

from ..utils import chunk_corpus, aggregate_doc_scores, reciprocal_rank_fusion, l2_normalize

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60, model=None):
        if model is not None:
            self.dense_model = model  # dependency injection for tests
        else:
            from sentence_transformers import SentenceTransformer
            self.dense_model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None
        self.bm25 = None

    def index(self, corpus):
        from rank_bm25 import BM25Okapi
        self.corpus_ids, self.corpus_texts = chunk_corpus(
            corpus, self.chunk_size, self.chunk_overlap
        )
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        embeddings = self.dense_model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = l2_normalize(np.array(embeddings))
        return self

    def retrieve(self, query, top_k=10):
        # BM25 channel
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = aggregate_doc_scores(self.corpus_ids, bm25_scores, top_k=top_k * 3)

        # Dense channel
        query_embedding = l2_normalize(np.array(self.dense_model.encode([query])))
        dense_scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        dense_ranking = aggregate_doc_scores(self.corpus_ids, dense_scores, top_k=top_k * 3)

        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=self.rrf_k)
        return fused[:top_k]

    def retrieve_batch(self, queries, top_k=10):
        return [self.retrieve(query, top_k=top_k) for query in queries]
