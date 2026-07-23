"""ColBERTv2-style Residual Compression for Late Interaction.

Standard ColBERT stores full 128-dim embeddings per token → huge storage.
ColBERTv2 (Santhanam et al., 2022) compresses these by:
1. Learning centroids over all token embeddings
2. Storing only: centroid ID + quantized residual (difference from centroid)
3. At query time: centroid interaction for candidate generation, then
   decompress residuals for full MaxSim reranking

This module implements the compression/decompression pipeline.
Works with numpy only (no torch required for inference).

Reference: Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval
           via Lightweight Late Interaction" (NAACL 2022)
"""

import numpy as np
import logging
import json
import os

from ..utils import assign_centroids as _assign_centroids_shared

logger = logging.getLogger(__name__)


class ResidualCompressor:
    """Compress per-token embeddings using centroid residuals.

    Instead of storing 128 floats per token, stores:
    - 1 byte for centroid ID (up to 256 centroids)
    - Quantized residual (e.g., 1 bit per dimension = 16 bytes for 128-dim)

    Compression ratio: 128*4=512 bytes → ~17 bytes per token (30x compression)
    """

    def __init__(self, num_centroids=256, nbits=2):
        """Initialize compressor.

        Args:
            num_centroids: number of centroids (k in k-means)
            nbits: bits per residual dimension (1, 2, or 4)
        """
        self.num_centroids = num_centroids
        self.nbits = nbits
        self.centroids = None
        self.residual_min = None
        self.residual_max = None
        self._fitted = False

    def fit(self, token_embeddings):
        """Learn centroids and residual quantization bounds.

        Args:
            token_embeddings: numpy array of shape (N, dim) or (N, max_tokens, dim)
        """
        if token_embeddings.ndim == 3:
            token_embeddings = token_embeddings.reshape(-1, token_embeddings.shape[-1])

        token_embeddings = token_embeddings.astype(np.float32)

        # K-means for centroids
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_centroids, random_state=42, n_init=3)
        kmeans.fit(token_embeddings)
        self.centroids = kmeans.cluster_centers_

        # Compute residuals for all tokens
        assignments = kmeans.labels_
        residuals = token_embeddings - self.centroids[assignments]

        # Learn quantization bounds
        self.residual_min = residuals.min(axis=0)
        self.residual_max = residuals.max(axis=0)

        self._fitted = True
        logger.info(f"Compressor fitted: {self.num_centroids} centroids, "
                    f"{self.nbits}-bit residuals, dim={token_embeddings.shape[1]}")

        return self

    def _quantize_residual(self, residual):
        """Quantize a residual vector to nbits per dimension.

        Maps each dimension from [min, max] to [0, 2^nbits - 1].
        """
        levels = 2 ** self.nbits
        # Normalize to [0, 1]
        range_vals = self.residual_max - self.residual_min
        range_vals = np.clip(range_vals, 1e-8, None)
        normalized = (residual - self.residual_min) / range_vals
        # Quantize to integer levels
        quantized = np.clip(normalized * levels, 0, levels - 1).astype(np.uint8)
        return quantized

    def _dequantize_residual(self, quantized):
        """Dequantize back to float residual."""
        levels = 2 ** self.nbits
        normalized = quantized.astype(np.float32) / levels
        range_vals = self.residual_max - self.residual_min
        residual = normalized * range_vals + self.residual_min
        return residual

    def _assign_centroids(self, tokens):
        """Assign each token to its nearest centroid.

        Uses the shared assign_centroids() from utils (canonical version).
        Kept as a method for API compatibility with existing callers.
        """
        return _assign_centroids_shared(tokens, self.centroids)

    def compress_document(self, token_embeddings):
        """Compress a document's token embeddings. Vectorized.

        Returns:
            centroid_ids: array of centroid indices (1 byte each)
            quantized_residuals: quantized residual vectors
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        if token_embeddings.ndim == 3:
            token_embeddings = token_embeddings.squeeze(0)

        token_embeddings = token_embeddings.astype(np.float32)

        centroid_ids = _assign_centroids_shared(token_embeddings, self.centroids)
        residuals = token_embeddings - self.centroids[centroid_ids]
        quantized_residuals = self._quantize_residual(residuals)

        return centroid_ids.astype(np.uint8), quantized_residuals.astype(np.uint8)

    def decompress_document(self, centroid_ids, quantized_residuals):
        """Decompress back to approximate token embeddings. Vectorized.

        Returns: approximate token embeddings (centroid + dequantized residual)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        centroid_ids = np.asarray(centroid_ids)
        quantized_residuals = np.asarray(quantized_residuals)
        residuals = self._dequantize_residual(quantized_residuals)
        return (self.centroids[centroid_ids] + residuals).astype(np.float32)

    def compute_compression_ratio(self, original_dim, max_tokens=256):
        """Compute storage savings.

        Original: 4 bytes * dim * max_tokens per document
        Compressed: (1 byte + nbits/8 * dim) * max_tokens per document
        """
        original_bytes = 4 * original_dim * max_tokens
        compressed_bytes_per_token = 1 + (self.nbits * original_dim) / 8
        compressed_bytes = compressed_bytes_per_token * max_tokens
        ratio = original_bytes / compressed_bytes
        return ratio

    def save(self, path):
        """Save compressor state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 centroids=self.centroids,
                 residual_min=self.residual_min,
                 residual_max=self.residual_max,
                 num_centroids=self.num_centroids,
                 nbits=self.nbits)

    def load(self, path):
        """Load compressor state."""
        data = np.load(path)
        self.centroids = data["centroids"]
        self.residual_min = data["residual_min"]
        self.residual_max = data["residual_max"]
        self.num_centroids = int(data["num_centroids"])
        self.nbits = int(data["nbits"])
        self._fitted = True
        return self


class CompressedCorpusIndex:
    """Index for compressed document embeddings.

    Stores compressed token embeddings with centroid-based
    candidate generation (PLAID-style).
    """

    def __init__(self, compressor):
        self.compressor = compressor
        self.doc_ids = []
        self.compressed_docs = []  # List of (centroid_ids, quantized_residuals)
        self.centroid_to_docs = {}  # centroid_id -> [doc_indices]

    def add_documents(self, doc_ids, token_embeddings_list):
        """Compress and index documents.

        Args:
            doc_ids: list of document IDs
            token_embeddings_list: list of token embedding arrays
        """
        for i, (doc_id, embeddings) in enumerate(zip(doc_ids, token_embeddings_list)):
            centroid_ids, quantized_residuals = self.compressor.compress_document(embeddings)
            self.doc_ids.append(doc_id)
            self.compressed_docs.append((centroid_ids, quantized_residuals))

            # Build centroid -> document mapping
            for cid in set(centroid_ids.tolist()):
                if cid not in self.centroid_to_docs:
                    self.centroid_to_docs[cid] = []
                self.centroid_to_docs[cid].append(len(self.doc_ids) - 1)

        logger.info(f"Indexed {len(doc_ids)} documents, "
                    f"{len(self.centroid_to_docs)} unique centroids")

    def search_centroids(self, query_embeddings, top_centroids=10):
        """Find top-k centroids closest to query tokens. Vectorized.

        This is the cheap first stage of PLAID-style search.
        """
        if query_embeddings.ndim == 3:
            query_embeddings = query_embeddings.squeeze(0)

        centroids = self.compressor.centroids
        centroid_sq = (centroids ** 2).sum(axis=1)
        query_sq = (query_embeddings ** 2).sum(axis=1, keepdims=True)
        dists_sq = query_sq + centroid_sq[None, :] - 2.0 * (query_embeddings @ centroids.T)
        top_centroids = min(top_centroids, centroids.shape[0])
        top_indices = np.argsort(dists_sq, axis=1)[:, :top_centroids]
        return set(top_indices.flatten().tolist())

    def retrieve(self, query_embeddings, top_k=10, top_centroids=10):
        """Retrieve using PLAID-style two-stage search.

        Stage 1: Find candidate documents via centroid overlap (fast)
        Stage 2: Decompress and compute full MaxSim for candidates (accurate)
        """
        from .brute_force import maxsim_score

        if query_embeddings.ndim == 3:
            query_embeddings = query_embeddings.squeeze(0)

        # Stage 1: Centroid-based candidate generation
        query_centroids = self.search_centroids(query_embeddings, top_centroids)

        candidate_set = set()
        for cid in query_centroids:
            if cid in self.centroid_to_docs:
                candidate_set.update(self.centroid_to_docs[cid])

        if not candidate_set:
            return []

        # Stage 2: Full MaxSim on candidates
        scores = []
        for doc_idx in candidate_set:
            centroid_ids, quantized_residuals = self.compressed_docs[doc_idx]
            doc_embeddings = self.compressor.decompress_document(centroid_ids, quantized_residuals)
            score = maxsim_score(query_embeddings, doc_embeddings)
            scores.append((self.doc_ids[doc_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path):
        """Save index to disk."""
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "compressed_docs": self.compressed_docs,
                "centroid_to_docs": self.centroid_to_docs,
            }, f)

    def load(self, path):
        """Load index from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.compressed_docs = data["compressed_docs"]
        self.centroid_to_docs = data["centroid_to_docs"]
        return self
