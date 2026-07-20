"""SPLADE: SParse Lexical AnD Expansion (Formal et al., 2021).

Learns sparse representations where each dimension corresponds to a vocabulary term.
Uses MLM (Masked Language Model) logits to weight terms — including terms NOT
present in the original text (expansion). Produces sparse vectors indexed with
an inverted index (term → postings), which is how SPLADE is meant to be served:
query scoring only touches chunks that share at least one query term.

Reference: Formal et al., "SPLADE: Sparse Lexical and Expansion Model for
           First Stage Ranking" (SIGIR 2021)
"""

import numpy as np
import logging
from collections import defaultdict

from ..utils import chunk_corpus, aggregate_doc_scores, l2_normalize, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class SPLADERetriever:
    """SPLADE-style sparse retrieval using MLM logit weighting.

    Uses BERT's MLM head to produce sparse term weights for each token position,
    then takes the max across positions. This naturally expands terms — e.g.,
    "cat" might activate "feline", "kitten", "pet" in the vocabulary.

    Index structure: inverted index {vocab_id: [(chunk_idx, weight), ...]}
    plus per-chunk L2 norms for cosine normalization.
    """

    def __init__(self, model_name="bert-base-uncased", chunk_size=200, chunk_overlap=50,
                 regularization_lambda=3e-4, batch_size=16):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reg_lambda = regularization_lambda
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_sparse_vectors = []   # kept for interpretability / expansion terms
        self._inverted_index = {}         # vocab_id -> [(chunk_idx, weight)]
        self._chunk_norms = None          # L2 norm per chunk
        self._idf_cache = {}

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        else:
            self._model = self._model.to("cpu")

        self._model.eval()
        logger.info(f"Loaded SPLADE model: {self.model_name}")

    def _texts_to_sparse_batch(self, texts, max_length=256):
        """Convert a batch of texts to sparse SPLADE vectors in ONE forward pass."""
        import torch

        self._load_model()

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab)

        # SPLADE activation: ReLU + log(1+x), max over sequence positions
        sparse = torch.log1p(torch.relu(logits)).max(dim=1).values  # (batch, vocab)
        sparse_np = sparse.cpu().numpy()

        results = []
        for row in sparse_np:
            nz = np.nonzero(row)[0]
            results.append({int(idx): float(row[idx]) for idx in nz})
        return results

    def _text_to_sparse(self, text, max_length=256):
        """Convert a single text to a sparse SPLADE vector."""
        return self._texts_to_sparse_batch([text], max_length=max_length)[0]

    def index(self, corpus):
        """Index corpus with SPLADE sparse vectors (batched encoding + inverted index)."""
        self._load_model()

        self.corpus_ids, self.corpus_texts = chunk_corpus(
            corpus, self.chunk_size, self.chunk_overlap
        )

        logger.info(f"Computing SPLADE vectors for {len(self.corpus_texts)} chunks "
                    f"(batch_size={self.batch_size})...")
        self.corpus_sparse_vectors = []
        for i in range(0, len(self.corpus_texts), self.batch_size):
            batch = self.corpus_texts[i:i + self.batch_size]
            self.corpus_sparse_vectors.extend(self._texts_to_sparse_batch(batch))
            if (i // self.batch_size + 1) % 50 == 0:
                logger.info(f"  Processed {min(i + self.batch_size, len(self.corpus_texts))}"
                            f"/{len(self.corpus_texts)}")

        # Build inverted index + norms
        self._inverted_index = defaultdict(list)
        self._chunk_norms = np.zeros(len(self.corpus_sparse_vectors), dtype=np.float32)
        for chunk_idx, sv in enumerate(self.corpus_sparse_vectors):
            norm_sq = 0.0
            for term_id, weight in sv.items():
                self._inverted_index[term_id].append((chunk_idx, weight))
                norm_sq += weight * weight
            self._chunk_norms[chunk_idx] = np.sqrt(norm_sq)

        # IDF weights for query re-weighting
        n_docs = len(self.corpus_sparse_vectors)
        self._idf_cache = {}
        for term_id, postings in self._inverted_index.items():
            self._idf_cache[term_id] = np.log((n_docs + 1) / (len(postings) + 1)) + 1.0

        return self

    def retrieve(self, query, top_k=10, use_idf_reweighting=True):
        """Retrieve using SPLADE sparse scoring over the inverted index.

        Only chunks sharing a term with the query are scored — this is the
        entire point of sparse retrieval (sub-linear in corpus size).
        """
        query_sparse = self._text_to_sparse(query)

        if use_idf_reweighting:
            query_sparse = {idx: val * self._idf_cache.get(idx, 1.0)
                            for idx, val in query_sparse.items()}

        # Accumulate scores via inverted index (cosine: normalize at the end)
        chunk_scores = defaultdict(float)
        for term_id, q_weight in query_sparse.items():
            postings = self._inverted_index.get(term_id)
            if postings is None:
                continue
            for chunk_idx, d_weight in postings:
                chunk_scores[chunk_idx] += q_weight * d_weight

        if not chunk_scores:
            return []

        # Cosine normalization
        query_norm = np.sqrt(sum(w * w for w in query_sparse.values()))
        if query_norm < 1e-8:
            query_norm = 1.0

        indices = list(chunk_scores.keys())
        scores = np.array([chunk_scores[i] for i in indices], dtype=np.float32)
        norms = self._chunk_norms[indices] * query_norm
        scores = scores / np.clip(norms, 1e-8, None)

        chunk_ids = [self.corpus_ids[i] for i in indices]
        return aggregate_doc_scores(chunk_ids, scores, top_k=top_k)

    def retrieve_batch(self, queries, top_k=10):
        return [self.retrieve(query, top_k=top_k) for query in queries]

    def get_expansion_terms(self, text, top_k=10):
        """Show which terms SPLADE would expand for a given text (interpretability)."""
        sparse_vec = self._text_to_sparse(text)
        vocab = self._tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        sorted_terms = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)
        return [(id_to_token.get(idx, f"[{idx}]"), weight) for idx, weight in sorted_terms[:top_k]]


class HybridSPLADERetriever:
    """Hybrid SPLADE + Dense retrieval with Reciprocal Rank Fusion.

    Combines SPLADE's sparse semantic expansion with dense embedding similarity.
    """

    def __init__(self, splade_model_name="bert-base-uncased",
                 dense_model_name="all-MiniLM-L6-v2",
                 chunk_size=200, chunk_overlap=50, rrf_k=60, dense_model=None):
        self.splade = SPLADERetriever(
            model_name=splade_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if dense_model is not None:
            self.dense_model = dense_model
        else:
            from sentence_transformers import SentenceTransformer
            self.dense_model = SentenceTransformer(dense_model_name)
        self.rrf_k = rrf_k

        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_dense_embeddings = None

    def index(self, corpus):
        """Index with both SPLADE sparse and dense embeddings."""
        self.splade.index(corpus)
        self.corpus_ids = self.splade.corpus_ids
        self.corpus_texts = self.splade.corpus_texts

        embeddings = self.dense_model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_dense_embeddings = l2_normalize(np.array(embeddings))

        return self

    def retrieve(self, query, top_k=10):
        """Retrieve using hybrid SPLADE + Dense + RRF."""
        splade_results = self.splade.retrieve(query, top_k=top_k * 3)

        query_emb = l2_normalize(np.array(self.dense_model.encode([query])))
        dense_scores = np.dot(self.corpus_dense_embeddings, query_emb.T).flatten()
        dense_ranking = aggregate_doc_scores(self.corpus_ids, dense_scores, top_k=top_k * 3)

        fused = reciprocal_rank_fusion([splade_results, dense_ranking], k=self.rrf_k)
        return fused[:top_k]
