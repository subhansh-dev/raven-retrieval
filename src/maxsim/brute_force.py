import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _normalize_np(x, axis=-1):
    """L2 normalize numpy array."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def maxsim_score(query_embeddings, document_embeddings):
    """Compute MaxSim score between query and document embeddings.

    Works with both torch tensors and numpy arrays.
    For each query token, finds max similarity with any document token, then sums.
    """
    if _HAS_TORCH and isinstance(query_embeddings, (type(None), )):
        pass

    # Convert to numpy
    if _HAS_TORCH and hasattr(query_embeddings, 'numpy'):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if _HAS_TORCH and hasattr(document_embeddings, 'numpy'):
        document_embeddings = document_embeddings.detach().cpu().numpy()

    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

    if query_embeddings.ndim == 3:
        query_embeddings = query_embeddings.squeeze(0)
    if document_embeddings.ndim == 3:
        document_embeddings = document_embeddings.squeeze(0)

    query_norm = _normalize_np(query_embeddings)
    doc_norm = _normalize_np(document_embeddings)
    similarity_matrix = query_norm @ doc_norm.T
    per_token_max = similarity_matrix.max(axis=1)
    total_score = float(per_token_max.sum())
    return total_score


def brute_force_rank(query_embeddings, document_embeddings_list, top_k=10):
    """Rank documents by MaxSim score against query."""
    scores = []
    for idx, doc_emb in enumerate(document_embeddings_list):
        score = maxsim_score(query_embeddings, doc_emb)
        scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def maxsim_score_batch(query_embeddings, doc_embeddings_tensor, doc_lengths):
    """Batch MaxSim scoring for concatenated document embeddings."""
    # Convert to numpy
    if _HAS_TORCH and hasattr(query_embeddings, 'numpy'):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if _HAS_TORCH and hasattr(doc_embeddings_tensor, 'numpy'):
        doc_embeddings_tensor = doc_embeddings_tensor.detach().cpu().numpy()

    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    doc_embeddings_tensor = np.asarray(doc_embeddings_tensor, dtype=np.float32)

    if query_embeddings.ndim == 3:
        query_embeddings = query_embeddings.squeeze(0)

    query_norm = _normalize_np(query_embeddings)
    doc_norm = _normalize_np(doc_embeddings_tensor)
    similarity = query_norm @ doc_norm.T

    scores = []
    offset = 0
    for length in doc_lengths:
        doc_sim = similarity[:, offset:offset + length]
        per_token_max = doc_sim.max(axis=1)
        scores.append(float(per_token_max.sum()))
        offset += length
    return scores
