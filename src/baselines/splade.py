"""SPLADE: SParse Lexical AnD Expansion (Formal et al., 2021).

Learns sparse representations where each dimension corresponds to a vocabulary term.
Uses MLM (Masked Language Model) logits to weight terms — including terms NOT
present in the original text (expansion). Produces sparse vectors that can be
efficiently indexed with inverted indices.

Key advantage: combines lexical matching (like BM25) with semantic expansion
(like dense retrieval) in a single sparse representation.

Reference: Formal et al., "SPLADE: Sparse Lexical and Expansion Model for
           First Stage Ranking" (SIGIR 2021)
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class SPLADERetriever:
    """SPLADE-style sparse retrieval using MLM logit weighting.

    Uses BERT's MLM head to produce sparse term weights for each token position,
    then takes the max across positions. This naturally expands terms — e.g.,
    "cat" might activate "feline", "kitten", "pet" in the vocabulary.
    """

    def __init__(self, model_name="bert-base-uncased", chunk_size=200, chunk_overlap=50,
                 regularization_lambda=3e-4):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reg_lambda = regularization_lambda

        self._tokenizer = None
        self._model = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_sparse_vectors = []  # List of sparse dicts {vocab_id: weight}
        self._idf_cache = {}

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                self._model = self._model.to("cpu")

            self._model.eval()
            logger.info(f"Loaded SPLADE model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise

    def _text_to_sparse(self, text, max_length=256):
        """Convert text to sparse SPLADE vector.

        Returns dict mapping vocab_id → log-weight (only non-zero entries).
        """
        import torch

        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # MLM logits: (batch, seq_len, vocab_size)
            logits = outputs.logits

        # ReLU + log(1 + x) activation (SPLADE's sparse activation)
        # Max over sequence positions → single sparse vector
        sparse = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze(0)

        # Convert to sparse dict (only keep non-zero entries)
        sparse_np = sparse.cpu().numpy()
        sparse_dict = {}
        for idx in np.nonzero(sparse_np)[0]:
            sparse_dict[int(idx)] = float(sparse_np[idx])

        return sparse_dict

    def _sparse_dot(self, a, b):
        """Compute dot product between two sparse vectors (dicts)."""
        if len(a) > len(b):
            a, b = b, a  # iterate over the smaller one
        score = 0.0
        for idx, val in a.items():
            if idx in b:
                score += val * b[idx]
        return score

    def _sparse_norm(self, a):
        """L2 norm of sparse vector."""
        return np.sqrt(sum(v * v for v in a.values()))

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
        """Index corpus with SPLADE sparse vectors."""
        self._load_model()

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
                chunk_id = f"{doc_id}::chunk::{i}"
                all_texts.append(chunk)
                all_ids.append(chunk_id)

        self.corpus_ids = all_ids
        self.corpus_texts = all_texts

        logger.info(f"Computing SPLADE vectors for {len(all_texts)} chunks...")
        self.corpus_sparse_vectors = []
        for i, text in enumerate(all_texts):
            sparse_vec = self._text_to_sparse(text)
            self.corpus_sparse_vectors.append(sparse_vec)
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1}/{len(all_texts)}")

        # Compute IDF-like weights for re-weighting
        doc_freq = {}
        for sv in self.corpus_sparse_vectors:
            for idx in sv:
                doc_freq[idx] = doc_freq.get(idx, 0) + 1

        n_docs = len(self.corpus_sparse_vectors)
        self._idf_cache = {}
        for idx, df in doc_freq.items():
            self._idf_cache[idx] = np.log((n_docs + 1) / (df + 1)) + 1.0

        return self

    def retrieve(self, query, top_k=10, use_idf_reweighting=True):
        """Retrieve using SPLADE sparse scoring."""
        query_sparse = self._text_to_sparse(query)

        if use_idf_reweighting:
            # Re-weight query terms by IDF
            reweighted = {}
            for idx, val in query_sparse.items():
                idf = self._idf_cache.get(idx, 1.0)
                reweighted[idx] = val * idf
            query_sparse = reweighted

        # Score against all documents
        scores = []
        for i, doc_sparse in enumerate(self.corpus_sparse_vectors):
            score = self._sparse_dot(query_sparse, doc_sparse)
            scores.append((i, score))

        # Aggregate by document ID (max chunk score)
        doc_scores = {}
        for idx, score in scores:
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def retrieve_batch(self, queries, top_k=10):
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            all_results.append(results)
        return all_results

    def get_expansion_terms(self, text, top_k=10):
        """Show which terms SPLADE would expand for a given text.

        Useful for interpretability — shows what the model thinks is relevant
        beyond the literal text.
        """
        sparse_vec = self._text_to_sparse(text)
        vocab = self._tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        # Get top-weighted terms
        sorted_terms = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)
        terms = []
        for idx, weight in sorted_terms[:top_k]:
            token = id_to_token.get(idx, f"[{idx}]")
            terms.append((token, weight))

        return terms


class HybridSPLADERetriever:
    """Hybrid SPLADE + Dense retrieval with Reciprocal Rank Fusion.

    Combines SPLADE's sparse semantic expansion with dense embedding similarity.
    This captures both lexical matching AND deep semantic understanding.
    """

    def __init__(self, splade_model_name="bert-base-uncased",
                 dense_model_name="all-MiniLM-L6-v2",
                 chunk_size=200, chunk_overlap=50, rrf_k=60):
        self.splade = SPLADERetriever(
            model_name=splade_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        from sentence_transformers import SentenceTransformer
        self.dense_model = SentenceTransformer(dense_model_name)
        self.rrf_k = rrf_k

        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_dense_embeddings = None

    def index(self, corpus):
        """Index with both SPLADE sparse and dense embeddings."""
        # SPLADE indexing
        self.splade.index(corpus)
        self.corpus_ids = self.splade.corpus_ids
        self.corpus_texts = self.splade.corpus_texts

        # Dense indexing
        embeddings = self.dense_model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_dense_embeddings = np.array(embeddings)
        norms = np.linalg.norm(self.corpus_dense_embeddings, axis=1, keepdims=True)
        self.corpus_dense_embeddings = self.corpus_dense_embeddings / norms

        return self

    def reciprocal_rank_fusion(self, rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def retrieve(self, query, top_k=10):
        """Retrieve using hybrid SPLADE + Dense + RRF."""
        # SPLADE retrieval
        splade_results = self.splade.retrieve(query, top_k=top_k * 3)

        # Dense retrieval
        query_emb = self.dense_model.encode([query])
        query_emb = np.array(query_emb)
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / query_norm
        dense_scores = np.dot(self.corpus_dense_embeddings, query_emb.T).flatten()

        dense_doc_scores = {}
        for idx in range(len(dense_scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in dense_doc_scores or dense_scores[idx] > dense_doc_scores[doc_id]:
                dense_doc_scores[doc_id] = float(dense_scores[idx])

        dense_sorted = sorted(dense_doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        dense_ranking = [(did, s) for did, s in dense_sorted]

        # RRF fusion
        fused = self.reciprocal_rank_fusion([splade_results, dense_ranking], k=self.rrf_k)
        return fused[:top_k]
