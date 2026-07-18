import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import faiss
except ImportError:
    faiss = None


class CentroidIndex:

    def __init__(self, num_centroids=256, nbits=8):
        self.num_centroids = num_centroids
        self.nbits = nbits
        self.index = None
        self.centroids = None
        self.residual_codebooks = None

    def build(self, token_embeddings):
        if faiss is None:
            raise ImportError("faiss is required for CentroidIndex. Install with: pip install faiss-cpu")
        if _HAS_TORCH and hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.detach().cpu().numpy()
        token_embeddings = np.asarray(token_embeddings, dtype=np.float32)
        if token_embeddings.ndim == 3:
            token_embeddings = token_embeddings.reshape(-1, token_embeddings.shape[-1])
        dimension = token_embeddings.shape[1]
        kmeans = faiss.Kmeans(dimension, self.num_centroids, niter=20, gpu=False)
        kmeans.train(token_embeddings)
        self.centroids = kmeans.centroids
        _, assignments = kmeans.index.search(token_embeddings, 1)
        self.assignments = assignments.flatten()
        residuals = token_embeddings - self.centroids[self.assignments]
        self.residual_quantizer = faiss.IndexFlatL2(dimension)
        self.residual_quantizer.add(residuals.astype(np.float32))
        return self

    def encode_document(self, token_embeddings):
        if _HAS_TORCH and hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.detach().cpu().numpy()
        if token_embeddings.ndim == 3:
            token_embeddings = token_embeddings.squeeze(0)
        centroid_ids = []
        residuals = []
        for token in token_embeddings:
            dists = np.linalg.norm(self.centroids - token, axis=1)
            cid = np.argmin(dists)
            centroid_ids.append(cid)
            residuals.append(token - self.centroids[cid])
        return np.array(centroid_ids), np.array(residuals)

    def search_centroids(self, query_embeddings, top_centroids=10):
        if _HAS_TORCH and hasattr(query_embeddings, 'numpy'):
            query_embeddings = query_embeddings.detach().cpu().numpy()
        if query_embeddings.ndim == 3:
            query_embeddings = query_embeddings.squeeze(0)
        query_dists = np.linalg.norm(
            self.centroids[None, :, :] - query_embeddings[:, None, :],
            axis=2
        )
        top_indices = np.argsort(query_dists, axis=1)[:, :top_centroids]
        return top_indices


class ApproximateMaxSim:

    def __init__(self, centroid_index, prune_ratio=0.1):
        self.centroid_index = centroid_index
        self.prune_ratio = prune_ratio
        self.doc_centroid_sets = []
        self.doc_token_embeddings = []

    def index_documents(self, token_embeddings_list):
        self.doc_centroid_sets = []
        self.doc_token_embeddings = []
        for doc_emb in token_embeddings_list:
            if _HAS_TORCH and hasattr(doc_emb, 'numpy'):
                doc_emb_np = doc_emb.detach().cpu().numpy()
            else:
                doc_emb_np = np.asarray(doc_emb, dtype=np.float32)
            if doc_emb_np.ndim == 3:
                doc_emb_np = doc_emb_np.squeeze(0)
            centroid_ids, _ = self.centroid_index.encode_document(doc_emb_np)
            self.doc_centroid_sets.append(set(centroid_ids.tolist()))
            self.doc_token_embeddings.append(doc_emb_np)

    def retrieve(self, query_embeddings, top_k=10):
        if _HAS_TORCH and hasattr(query_embeddings, 'numpy'):
            query_emb_np = query_embeddings.detach().cpu().numpy()
        else:
            query_emb_np = np.asarray(query_embeddings, dtype=np.float32)
        if query_emb_np.ndim == 3:
            query_emb_np = query_emb_np.squeeze(0)
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
        def _normalize_np(x):
            norms = np.linalg.norm(x, axis=-1, keepdims=True)
            return x / np.clip(norms, 1e-8, None)

        query_norm = _normalize_np(query_emb_np)
        scored = []
        for idx, _ in candidate_indices:
            doc_emb = self.doc_token_embeddings[idx]
            doc_norm = _normalize_np(doc_emb)
            sim = query_norm @ doc_norm.T
            per_token_max = sim.max(axis=1)
            score = float(per_token_max.sum())
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def compute_fidelity(self, query_embeddings, brute_force_ranking, k=10):
        approximate_ranking = self.retrieve(query_embeddings, top_k=k)
        approx_ids = set(idx for idx, _ in approximate_ranking)
        brute_ids = set(idx for idx, _ in brute_force_ranking[:k])
        overlap = len(approx_ids & brute_ids)
        return overlap / k
