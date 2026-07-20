"""Tests for the internal (pure numpy) evaluation metrics.

These pin the trec_eval-compatible nDCG / recall / precision / MAP
implementations against hand-computed values — guaranteeing that the
per-query scores feeding significance tests are REAL values (the old
implementation silently returned zeros by treating corpus-averaged
floats as a per-query dict).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import (
    ndcg_per_query, recall_per_query, precision_per_query,
    map_per_query, evaluate_internal, collect_per_query_ndcg,
)


def _good_ranking(qrels, results):
    """Helper: build a results dict that ranks docs in ideal-relevance order."""
    out = {}
    for qid, rels in qrels.items():
        ranked = sorted(rels.items(), key=lambda x: -x[1])
        for rank, (did, _) in enumerate(ranked):
            out.setdefault(qid, {})[did] = 10.0 - rank  # high score = high rank
    return out


def test_ndcg_per_query_perfect_ranking_is_one():
    qrels = {"q1": {"d1": 3, "d2": 2, "d3": 1, "d4": 0}}
    results = {"q1": {"d1": 0.9, "d2": 0.5, "d3": 0.1}}  # ideal order
    scores = ndcg_per_query(qrels, results, k=10)
    assert abs(scores["q1"] - 1.0) < 1e-6, f"perfect ranking should be 1.0, got {scores['q1']}"


def test_ndcg_per_query_worse_ranking_lower():
    qrels = {"q1": {"d1": 3, "d2": 0}}
    perfect = {"q1": {"d1": 1.0}}
    bad = {"q1": {"d2": 1.0}}
    assert ndcg_per_query(qrels, perfect, k=10)["q1"] == 1.0
    assert ndcg_per_query(qrels, bad, k=10)["q1"] == 0.0


def test_ndcg_discount_matches_trec_eval():
    """DCG@2 for rels [3 (rank1), 0 (rank2)] = 3/log2(2) + 0 = 3.
    IDCG@2 for sorted rels [3] = 3/log2(2) = 3. nDCG = 1.0.
    With rels flipped to [0 (rank1), 3 (rank2)]: DCG = 0 + 3/log2(3) = 3/1.585 ≈ 1.894.
    nDCG = 1.894 / 3 ≈ 0.631."""
    qrels = {"q1": {"d1": 3, "d2": 0}}
    results = {"q1": {"d2": 1.0, "d1": 0.5}}  # rank1=d2(rel0), rank2=d1(rel3)
    scores = ndcg_per_query(qrels, results, k=2)
    expected = (3 / np.log2(3)) / 3.0
    assert abs(scores["q1"] - expected) < 1e-6


def test_recall_per_query():
    qrels = {"q1": {"d1": 1, "d2": 1, "d3": 1}}
    # retrieve d1, d2, junk — 2 of 3 relevant in top-3
    results = {"q1": {"d1": 0.9, "dXX": 0.8, "d2": 0.7, "dYY": 0.6}}
    assert abs(recall_per_query(qrels, results, k=3)["q1"] - 2/3) < 1e-6
    # retrieve junk, junk, d3 — d3 is rank 3 → 1 of 3 relevant in top-3
    results = {"q1": {"dXX": 0.9, "dYY": 0.8, "d3": 0.7, "dZZ": 0.6}}
    assert abs(recall_per_query(qrels, results, k=3)["q1"] - 1/3) < 1e-6
    # rank-4 (d3 not in top-3) → 0 of 3
    results = {"q1": {"dXX": 0.9, "dYY": 0.8, "dZZ": 0.7, "d3": 0.6}}
    assert abs(recall_per_query(qrels, results, k=3)["q1"] - 0.0) < 1e-6


def test_precision_per_query():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    results = {"q1": {"d1": 0.9, "dXX": 0.8, "d2": 0.7}}
    assert abs(precision_per_query(qrels, results, k=3)["q1"] - 2/3) < 1e-6


def test_map_per_query():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    # ranks: 1=dXX(rel0), 2=d1(rel1, P@2=1/2), 3=d2(rel1, P@3=2/3)
    # AP = (0.5 + 2/3) / 2 (divide by total relevant)
    results = {"q1": {"dXX": 0.9, "d1": 0.8, "d2": 0.7}}
    expected = (0.5 + 2/3) / 2
    assert abs(map_per_query(qrels, results, k=3)["q1"] - expected) < 1e-6


def test_evaluate_internal_structure():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"dXX": 1.0}}
    m = evaluate_internal(qrels, results, k_values=[1, 5, 10])
    assert set(m.keys()) == {"ndcg", "map", "recall", "precision"}
    assert set(m["ndcg"].keys()) == {"NDCG@1", "NDCG@5", "NDCG@10"}
    # q1 perfect (nDCG 1.0), q2 total miss (0.0) → average 0.5
    assert abs(m["ndcg"]["NDCG@10"] - 0.5) < 1e-4


def test_collect_per_query_ndcg_aligns_to_qrels_keys():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"dXX": 1.0}, "q3": {"d3": 1.0}}
    scores = collect_per_query_ndcg(qrels, results, k=10)
    assert len(scores) == 3
    assert scores == [1.0, 0.0, 1.0]  # aligned to qrels key order


def test_no_relevant_docs_excluded():
    """Queries with no relevant docs should be excluded from averages
    (matching trec_eval), and absent from per-query dicts."""
    qrels = {"q1": {"d1": 0}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    pq = ndcg_per_query(qrels, results, k=10)
    assert "q1" not in pq  # excluded
    assert "q2" in pq
    m = evaluate_internal(qrels, results, k_values=[10])
    # average over q2 only → 1.0
    assert abs(m["ndcg"]["NDCG@10"] - 1.0) < 1e-4


def test_empty_results_zero():
    qrels = {"q1": {"d1": 1}}
    scores = ndcg_per_query(qrels, {}, k=10)
    assert scores == {"q1": 0.0}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("All metrics tests passed.")