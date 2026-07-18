"""BM25 with Rocchio Pseudo-Relevance Feedback (PRF).

Standard BM25 retrieves based on exact term overlap. PRF expands the
query by analyzing top-k initial results — terms that appear frequently
in top results but rarely in the corpus are added to the query.

This is a classic IR technique (Rocchio, 1971) that still works remarkably
well and requires zero ML models.

Reference: Rocchio (1971), "Relevance Feedback in Information Retrieval"
           Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"
"""

import numpy as np
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25PRFRetriever:
    """BM25 with Pseudo-Relevance Feedback.

    Process:
    1. Initial BM25 retrieval (top-k)
    2. Extract terms from top-k documents
    3. Weight terms by TF-IDF difference (high in results, low in corpus)
    4. Expand query with top-weighted terms
    5. Re-retrieve with expanded query

    This bridges the vocabulary gap between query and documents
    without any neural network.
    """

    def __init__(self, chunk_size=200, chunk_overlap=50,
                 prf_k=5, expansion_terms=10, alpha=1.0, beta=0.75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prf_k = prf_k           # Number of top docs for PRF
        self.expansion_terms = expansion_terms  # Terms to add
        self.alpha = alpha           # Weight for original query terms
        self.beta = beta             # Weight for expansion terms

        self.corpus_ids = []
        self.corpus_texts = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._doc_freq = {}          # Document frequency per term
        self._n_docs = 0

    def chunk_text(self, text):
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def index(self, corpus):
        all_texts = []
        all_ids = []

        if isinstance(corpus, dict):
            corpus_iter = [(did, d) for did, d in corpus.items()]
        else:
            corpus_iter = [(doc["_id"], doc) for doc in corpus]

        for doc_id, doc in corpus_iter:
            full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_ids.append(f"{doc_id}::chunk::{i}")

        self.corpus_ids = all_ids
        self.corpus_texts = all_texts
        self.tokenized_corpus = [text.lower().split() for text in all_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._n_docs = len(self.tokenized_corpus)

        # Compute document frequencies
        self._doc_freq = {}
        for tokens in self.tokenized_corpus:
            unique = set(tokens)
            for t in unique:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        return self

    def _compute_idf(self, term):
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return np.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _expand_query(self, query_tokens, top_doc_indices):
        """Expand query using Rocchio-style PRF.

        For each term in top documents, compute a weight that balances:
        - Term frequency in top documents (higher = more relevant)
        - Inverse document frequency in corpus (higher = more discriminative)
        - Presence in original query (already covered terms get less boost)
        """
        # Term scores from top documents
        term_scores = {}
        query_set = set(query_tokens)

        for doc_idx in top_doc_indices:
            doc_tokens = self.tokenized_corpus[doc_idx]
            for term in doc_tokens:
                if len(term) < 3:  # Skip very short terms
                    continue
                if term in query_set:
                    continue  # Don't expand with existing query terms

                tf = doc_tokens.count(term)
                idf = self._compute_idf(term)
                score = tf * idf

                if term not in term_scores:
                    term_scores[term] = 0.0
                term_scores[term] += score

        # Normalize scores
        if term_scores:
            max_score = max(term_scores.values())
            if max_score > 0:
                for t in term_scores:
                    term_scores[t] /= max_score

        # Select top expansion terms
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion = [(t, s) for t, s in sorted_terms[:self.expansion_terms]]

        return expansion

    def retrieve(self, query, top_k=10):
        """Retrieve with BM25 + PRF expansion."""
        tokenized_query = query.lower().split()

        # Step 1: Initial BM25 retrieval
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k document indices for PRF
        top_indices = np.argsort(scores)[::-1][:self.prf_k]

        # Step 2: Expand query
        expansion = self._expand_query(tokenized_query, top_indices)

        # Step 3: Build expanded query weights
        query_weights = {}
        for t in tokenized_query:
            query_weights[t] = self.alpha
        for t, s in expansion:
            if t in query_weights:
                query_weights[t] += self.beta * s
            else:
                query_weights[t] = self.beta * s

        # Step 4: Re-score with expanded query
        expanded_scores = np.zeros(self._n_docs)
        for term, weight in query_weights.items():
            term_bm25 = self.bm25.get_scores([term])
            expanded_scores += weight * term_bm25

        # Aggregate by document ID
        doc_scores = {}
        for idx in range(len(expanded_scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or expanded_scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(expanded_scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def retrieve_no_prf(self, query, top_k=10):
        """Standard BM25 retrieval without PRF (for comparison)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
