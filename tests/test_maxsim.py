import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.maxsim.brute_force import maxsim_score, brute_force_rank, maxsim_score_batch
from src.maxsim.approximate import CentroidIndex, ApproximateMaxSim


def test_maxsim_basic():
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    score1 = maxsim_score(q, d1)
    score2 = maxsim_score(q, d2)
    assert score1 > score2, f"Expected d1 > d2, got {score1} vs {score2}"
    assert abs(score1 - 1.0) < 1e-6, f"Expected ~1.0, got {score1}"


def test_maxsim_ranking():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ]
    ranked = brute_force_rank(q, docs, top_k=3)
    assert ranked[-1][0] == 1, f"Expected doc 1 last, got {ranked[-1][0]}"
    assert ranked[0][1] > ranked[-1][1], "Top score should exceed bottom score"


def test_maxsim_batch():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs_flat = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    lengths = [2, 2]
    scores = maxsim_score_batch(q, docs_flat, lengths)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_centroid_index():
    rng = np.random.RandomState(42)
    tokens = rng.randn(100, 8).astype(np.float32)
    index = CentroidIndex(num_centroids=4)
    index.build(tokens)
    assert index.centroids.shape == (4, 8)
    doc_emb = rng.randn(10, 8).astype(np.float32)
    cids, residuals = index.encode_document(doc_emb)
    assert len(cids) == 10
    assert residuals.shape == (10, 8)


def test_approximate_maxsim():
    rng = np.random.RandomState(42)
    dim = 8
    all_tokens = rng.randn(200, dim).astype(np.float32)
    index = CentroidIndex(num_centroids=4)
    index.build(all_tokens)
    approx = ApproximateMaxSim(index, prune_ratio=0.5)
    doc_embs = [rng.randn(10, dim).astype(np.float32) for _ in range(5)]
    approx.index_documents(doc_embs)
    q = rng.randn(5, dim).astype(np.float32)
    results = approx.retrieve(q, top_k=3)
    assert len(results) <= 3
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)


def test_approximate_fidelity():
    rng = np.random.RandomState(42)
    dim = 8
    all_tokens = rng.randn(200, dim).astype(np.float32)
    index = CentroidIndex(num_centroids=4)
    index.build(all_tokens)
    approx = ApproximateMaxSim(index, prune_ratio=1.0)
    doc_embs = [rng.randn(10, dim).astype(np.float32) for _ in range(10)]
    approx.index_documents(doc_embs)
    q = rng.randn(5, dim).astype(np.float32)
    approx_ranking = approx.retrieve(q, top_k=5)
    brute_ranking = brute_force_rank(q, doc_embs, top_k=5)
    fidelity = approx.compute_fidelity(q, brute_ranking, k=5)
    assert 0.0 <= fidelity <= 1.0


if __name__ == "__main__":
    test_maxsim_basic()
    test_maxsim_ranking()
    test_maxsim_batch()
    test_centroid_index()
    test_approximate_maxsim()
    test_approximate_fidelity()
    print("All MaxSim tests passed.")
