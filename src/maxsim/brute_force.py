"""MaxSim scoring (ColBERT late interaction) — pure numpy.

MaxSim(Q, D) = Σᵢ maxⱼ cos(qᵢ, dⱼ)

All functions accept numpy arrays or torch tensors (converted automatically).
Query/doc inputs may be (n_tokens, dim) or (1, n_tokens, dim).
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_numpy(x):
    if _HAS_TORCH and hasattr(x, 'numpy'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x.squeeze(0)
    return x


def _normalize_np(x, axis=-1):
    """L2 normalize numpy array."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def maxsim_score(query_embeddings, document_embeddings):
    """Compute MaxSim score between query and document token embeddings."""
    query_embeddings = _to_numpy(query_embeddings)
    document_embeddings = _to_numpy(document_embeddings)

    if query_embeddings.shape[0] == 0 or document_embeddings.shape[0] == 0:
        return 0.0

    query_norm = _normalize_np(query_embeddings)
    doc_norm = _normalize_np(document_embeddings)
    similarity_matrix = query_norm @ doc_norm.T
    per_token_max = similarity_matrix.max(axis=1)
    total_score = float(per_token_max.sum())
    return total_score


def maxsim_score_masked(query_embeddings, query_mask, document_embeddings, doc_mask):
    """MaxSim with explicit masks (1 = real token, 0 = padding).

    Padded query rows contribute nothing; padded doc columns are never
    selected as a max. Use this when working with padded batch tensors.
    """
    query_embeddings = _to_numpy(query_embeddings)
    document_embeddings = _to_numpy(document_embeddings)
    query_mask = np.asarray(query_mask).reshape(-1).astype(bool)
    doc_mask = np.asarray(doc_mask).reshape(-1).astype(bool)

    q = query_embeddings[query_mask]
    d = document_embeddings[doc_mask]
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    return maxsim_score(q, d)


def brute_force_rank(query_embeddings, document_embeddings_list, top_k=10):
    """Rank documents by MaxSim score against query.

    Prefer brute_force_rank_fast for large corpora — it scores all
    documents in one matrix multiply instead of a Python loop.
    """
    scores = []
    for idx, doc_emb in enumerate(document_embeddings_list):
        score = maxsim_score(query_embeddings, doc_emb)
        scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def maxsim_score_batch(query_embeddings, doc_embeddings_tensor, doc_lengths):
    """Batch MaxSim scoring for concatenated document embeddings."""
    query_embeddings = _to_numpy(query_embeddings)
    doc_embeddings_tensor = np.asarray(doc_embeddings_tensor, dtype=np.float32)
    if doc_embeddings_tensor.ndim == 3:
        doc_embeddings_tensor = doc_embeddings_tensor.reshape(-1, doc_embeddings_tensor.shape[-1])

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


def pack_doc_embeddings(document_embeddings_list):
    """Pack a list of (n_i, dim) arrays into (flat_array, doc_lengths).

    This is the input format for fast scoring — concatenate once, reuse
    for every query.
    """
    arrays = [_to_numpy(d) for d in document_embeddings_list]
    doc_lengths = [a.shape[0] for a in arrays]
    if arrays:
        flat = np.concatenate(arrays, axis=0)
    else:
        flat = np.zeros((0, 1), dtype=np.float32)
    return flat, doc_lengths


def brute_force_rank_fast(query_embeddings, flat_doc_embeddings, doc_lengths, top_k=10):
    """Rank all documents against a query in ONE matrix multiply.

    Equivalent to brute_force_rank but ~Nx faster for N documents:
    computes the full (n_query_tokens × total_doc_tokens) similarity
    matrix once, then reduces per document via offset slicing.

    Args:
        query_embeddings: (n_q, dim) query token embeddings
        flat_doc_embeddings: (total_tokens, dim) — from pack_doc_embeddings
        doc_lengths: list of per-doc token counts
        top_k: results to return

    Returns: [(doc_index, score), ...] sorted by score desc.
    """
    scores = maxsim_score_batch(query_embeddings, flat_doc_embeddings, doc_lengths)
    order = np.argsort(np.asarray(scores))[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order]
