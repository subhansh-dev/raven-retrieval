"""BM25 with Rocchio Pseudo-Relevance Feedback (PRF).

Process:
1. Initial BM25 retrieval
2. Extract TF*IDF-weighted terms from top-k documents
3. Expand query with the best terms
4. Weighted re-scoring — but ONLY over the candidate pool from stage 1

The old implementation re-scored the ENTIRE corpus per expansion term
(~20 full-corpus BM25 passes per query). The two-stage design here
(first-stage candidates → weighted rerank of candidates only) is the
classic efficient PRF architecture and cuts cost by ~100x at corpus scale.

Reference: Rocchio (1971), "Relevance Feedback in Information Retrieval"
           Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework"
"""

import numpy as np
import logging

from ..utils import chunk_corpus, aggregate_doc_scores

logger = logging.getLogger(__name__)


class BM25PRFRetriever:
    """BM25 with Pseudo-Relevance Feedback (two-stage, efficient)."""

    def __init__(self, chunk_size=200, chunk_overlap=50,
                 prf_k=5, expansion_terms=10, alpha=1.0, beta=0.75,
                 candidate_pool=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prf_k = prf_k               # top docs used for term extraction
        self.expansion_terms = expansion_terms
        self.alpha = alpha               # weight for original query terms
        self.beta = beta                 # weight for expansion terms
        self.candidate_pool = candidate_pool  # chunks rescored in stage 2

        self.corpus_ids = []
        self.corpus_texts = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._doc_freq = {}
        self._n_docs = 0

    def index(self, corpus):
        from rank_bm25 import BM25Okapi
        self.corpus_ids, self.corpus_texts = chunk_corpus(
            corpus, self.chunk_size, self.chunk_overlap
        )
        self.tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._n_docs = len(self.tokenized_corpus)

        self._doc_freq = {}
        for tokens in self.tokenized_corpus:
            for t in set(tokens):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        return self

    def _compute_idf(self, term):
        """BM25 IDF (Robertson-Walker) for a term."""
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return np.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _expand_query(self, query_tokens, top_doc_indices):
        """Extract expansion terms from top documents (Rocchio-style).

        Terms are scored by TF (in top docs) * IDF (corpus) — frequent in
        feedback docs AND rare in the corpus.
        """
        term_scores = {}
        query_set = set(query_tokens)

        for doc_idx in top_doc_indices:
            doc_tokens = self.tokenized_corpus[doc_idx]
            for term in set(doc_tokens):
                if len(term) < 3 or term in query_set:
                    continue
                tf = doc_tokens.count(term)
                idf = self._compute_idf(term)
                term_scores[term] = term_scores.get(term, 0.0) + tf * idf

        if term_scores:
            max_score = max(term_scores.values())
            if max_score > 0:
                for t in term_scores:
                    term_scores[t] /= max_score

        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:self.expansion_terms]

    def _weighted_rescore(self, query_weights, candidate_indices):
        """Weighted BM25 re-scoring over the candidate pool only.

        For each candidate chunk, score = Σ_terms w_t * idf_t * tf_component.
        Uses the BM25Okapi internal parameters for exact scoring consistency.
        """
        k1 = self.bm25.k1
        b = self.bm25.b
        avgdl = self.bm25.avgdl if self.bm25.avgdl > 0 else 1.0

        scores = {}
        for idx in candidate_indices:
            doc_tokens = self.tokenized_corpus[idx]
            dl = len(doc_tokens)
            K = k1 * (1 - b + b * dl / avgdl)
            # Term frequencies for this doc (computed once per candidate)
            tf_map = {}
            for t in doc_tokens:
                if t in query_weights:
                    tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for term, f in tf_map.items():
                idf = self._compute_idf(term)
                tf_component = f * (k1 + 1) / (f + K)
                score += query_weights[term] * idf * tf_component
            scores[idx] = score
        return scores

    def retrieve(self, query, top_k=10):
        """Retrieve with BM25 + PRF expansion (two-stage)."""
        tokenized_query = query.lower().split()

        # Stage 1: initial BM25 retrieval over full corpus (one pass)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]

        # Candidate pool for stage 2
        pool_size = max(self.candidate_pool, top_k * 3)
        candidate_indices = ranked_indices[:pool_size].tolist()
        feedback_indices = ranked_indices[:self.prf_k].tolist()

        # Stage 2: expand query and weighted rescore of candidates only
        expansion = self._expand_query(tokenized_query, feedback_indices)

        query_weights = {}
        for t in set(tokenized_query):
            query_weights[t] = self.alpha
        for t, s in expansion:
            query_weights[t] = query_weights.get(t, 0.0) + self.beta * s

        candidate_scores = self._weighted_rescore(query_weights, candidate_indices)

        # Aggregate by document ID (max chunk score)
        chunk_ids = [self.corpus_ids[idx] for idx in candidate_indices]
        chunk_scores = [candidate_scores[idx] for idx in candidate_indices]
        return aggregate_doc_scores(chunk_ids, chunk_scores, top_k=top_k)

    def retrieve_no_prf(self, query, top_k=10):
        """Standard BM25 retrieval without PRF (for comparison)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k)
