"""PLAID-style approximate MaxSim retrieval.

Pipeline: K-means centroids over all token embeddings → per-document
centroid sets → query-time centroid overlap for candidate generation →
full MaxSim reranking on candidates only.

Vectorized: centroid assignment is done with matrix ops, not per-token
Python loops (the old implementation was ~256x slower than necessary).
"""

import numpy as np

from ..utils import to_numpy as _to_numpy

try:
    import faiss
except ImportError:
    faiss = None


def _assign_centroids(centroids, tokens):
    """Assign each token to its nearest centroid. Vectorized.

    Uses ||a-b||² = |a|² + |b|² - 2a·b to avoid materializing
    (n_tokens, n_centroids, dim) arrays.
    """
    tokens = _to_numpy(tokens)
    centroid_sq = (centroids ** 2).sum(axis=1)
    token_sq = (tokens ** 2).sum(axis=1, keepdims=True)
    dists_sq = token_sq + centroid_sq[None, :] - 2.0 * (tokens @ centroids.T)
    return np.argmin(dists_sq, axis=1)


class CentroidIndex:

    def __init__(self, num_centroids=256, nbits=8):
        self.num_centroids = num_centroids
        self.nbits = nbits
        self.index = None
        self.centroids = None
        self.assignments = None

    def build(self, token_embeddings):
        if faiss is None:
            raise ImportError("faiss is required for CentroidIndex. Install with: pip install faiss-cpu")
        token_embeddings = _to_numpy(token_embeddings)
        if token_embeddings.ndim == 3:
            token_embeddings = token_embeddings.reshape(-1, token_embeddings.shape[-1])
        dimension = token_embeddings.shape[1]
        n_centroids = min(self.num_centroids, max(1, token_embeddings.shape[0]))
        kmeans = faiss.Kmeans(dimension, n_centroids, niter=20, gpu=False)
        kmeans.train(token_embeddings)
        self.centroids = kmeans.centroids
        self.assignments = _assign_centroids(self.centroids, token_embeddings)
        return self

    def encode_document(self, token_embeddings):
        """Encode a document as (centroid_ids, residuals). Vectorized."""
        tokens = _to_numpy(token_embeddings)
        centroid_ids = _assign_centroids(self.centroids, tokens)
        residuals = tokens - self.centroids[centroid_ids]
        return centroid_ids, residuals

    def search_centroids(self, query_embeddings, top_centroids=10):
        """Find the top_centroids nearest centroids for each query token."""
        query_embeddings = _to_numpy(query_embeddings)
        centroid_sq = (self.centroids ** 2).sum(axis=1)
        query_sq = (query_embeddings ** 2).sum(axis=1, keepdims=True)
        dists_sq = query_sq + centroid_sq[None, :] - 2.0 * (query_embeddings @ self.centroids.T)
        top_centroids = min(top_centroids, self.centroids.shape[0])
        top_indices = np.argsort(dists_sq, axis=1)[:, :top_centroids]
        return top_indices


class ApproximateMaxSim:

    def __init__(self, centroid_index, prune_ratio=0.1):
        self.centroid_index = centroid_index
        self.prune_ratio = prune_ratio
        self.doc_centroid_sets = []
        self.doc_token_embeddings = []
        self._flat_embeddings = None
        self._doc_lengths = None

    def index_documents(self, token_embeddings_list):
        self.doc_centroid_sets = []
        self.doc_token_embeddings = []
        for doc_emb in token_embeddings_list:
            doc_emb_np = _to_numpy(doc_emb)
            centroid_ids, _ = self.centroid_index.encode_document(doc_emb_np)
            self.doc_centroid_sets.append(set(centroid_ids.tolist()))
            self.doc_token_embeddings.append(doc_emb_np)
        # Pre-pack for fast reranking
        from .brute_force import pack_doc_embeddings
        self._flat_embeddings, self._doc_lengths = pack_doc_embeddings(self.doc_token_embeddings)

    def retrieve(self, query_embeddings, top_k=10):
        query_emb_np = _to_numpy(query_embeddings)
        top_centroid_ids = self.centroid_index.search_centroids(query_emb_np)
        query_centroids = set(top_centroid_ids.flatten().tolist())

        candidate_indices = []
        for idx, doc_centroids in enumerate(self.doc_centroid_sets):
            overlap = len(query_centroids & doc_centroids)
            if overlap > 0:
                candidate_indices.append((idx, overlap))
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        max_candidates = max(top_k * 5, int(len(self.doc_token_embeddings) * self.prune_ratio))
        candidate_indices = candidate_indices[:max_candidates]

        if not candidate_indices:
            return []

        # Full MaxSim reranking on candidates only
        from .brute_force import maxsim_score_batch
        candidate_ids = [idx for idx, _ in candidate_indices]
        sub_lengths = [self._doc_lengths[idx] for idx in candidate_ids]
        offsets = np.concatenate([[0], np.cumsum(self._doc_lengths)])
        sub_flat = np.concatenate([
            self._flat_embeddings[offsets[idx]:offsets[idx] + self._doc_lengths[idx]]
            for idx in candidate_ids
        ], axis=0)
        scores = maxsim_score_batch(query_emb_np, sub_flat, sub_lengths)
        order = np.argsort(np.asarray(scores))[::-1][:top_k]
        return [(candidate_ids[i], float(scores[i])) for i in order]

    def compute_fidelity(self, query_embeddings, brute_force_ranking, k=10):
        approximate_ranking = self.retrieve(query_embeddings, top_k=k)
        approx_ids = set(idx for idx, _ in approximate_ranking)
        brute_ids = set(idx for idx, _ in brute_force_ranking[:k])
        overlap = len(approx_ids & brute_ids)
        return overlap / k
