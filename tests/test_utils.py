"""Tests for shared utilities (src.utils).

These run with numpy only — no torch/transformers — so they execute in
both the lightweight CI job and the full test suite.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    chunk_text, chunk_corpus, iter_corpus, full_doc_text,
    aggregate_doc_scores, reciprocal_rank_fusion,
    l2_normalize, masked_mean_pool, PipelineTimer,
)


def test_chunk_text_basic():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(25)), chunk_size=10, chunk_overlap=0)
    assert len(chunks) == 3
    assert chunks[0].split() == [f"w{i}" for i in range(10)]
    assert len(chunks[2].split()) == 5


def test_chunk_text_overlap():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(25)), chunk_size=10, chunk_overlap=5)
    # stride = 5, so chunks start at 0,5,10,15,20 → 5 chunks
    assert len(chunks) == 5


def test_chunk_text_empty_and_bad():
    assert chunk_text("", 10, 0) == []
    try:
        chunk_text("a b c", chunk_size=0, chunk_overlap=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        chunk_text("a b c", chunk_size=10, chunk_overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_iter_corpus_dict_and_list():
    d = {"d1": {"title": "A", "text": "alpha"}, "d2": {"title": "B", "text": "beta"}}
    items = list(iter_corpus(d))
    assert {i[0] for i in items} == {"d1", "d2"}

    lst = [{"_id": "d1", "title": "A", "text": "alpha"}]
    items = list(iter_corpus(lst))
    assert items[0][0] == "d1"


def test_full_doc_text():
    assert full_doc_text({"title": "T", "text": "X"}) == "T X"
    assert full_doc_text({"title": "", "text": "X"}) == "X"


def test_chunk_corpus_id_format():
    corpus = {"d1": {"title": "", "text": " ".join("w" * 25)}}
    ids, texts = chunk_corpus(corpus, chunk_size=10, chunk_overlap=0)
    assert ids[0] == "d1::chunk::0"
    assert len(ids) == 3


def test_aggregate_doc_scores_max_pool():
    chunk_ids = ["d1::chunk::0", "d1::chunk::1", "d2::chunk::0"]
    scores = [0.5, 0.9, 0.3]
    agg = aggregate_doc_scores(chunk_ids, scores, top_k=10)
    assert agg[0] == ("d1", 0.9)  # max-pooled, sorted desc
    assert agg[1] == ("d2", 0.3)
    assert len(agg) == 2


def test_aggregate_doc_scores_top_k_truncation():
    ids = [f"d{i}::chunk::0" for i in range(5)]
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    agg = aggregate_doc_scores(ids, scores, top_k=3)
    assert len(agg) == 3
    assert agg[0] == ("d2", 0.9)


def test_reciprocal_rank_fusion():
    ranking_a = [("d1", 0.9), ("d2", 0.5), ("d3", 0.1)]
    ranking_b = [("d2", 0.8), ("d1", 0.4), ("d4", 0.2)]
    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b], k=60))
    # d1: 1/(60+1) + 1/(60+2) ; d2: 1/(60+2) + 1/(60+1) ; d4: 1/(60+3)
    assert abs(fused["d1"] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(fused["d2"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused["d4"] - (1 / 63)) < 1e-9
    # d1 and d2 should tie (appear in both rankings), d4 should be lowest
    assert fused["d4"] < fused["d1"]


def test_l2_normalize_zero_vector_safe():
    v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    n = l2_normalize(v)
    assert not np.any(np.isnan(n))


def test_l2_normalize_unit_length():
    v = np.array([[3.0, 4.0]], dtype=np.float32)
    n = l2_normalize(v)
    assert abs(np.linalg.norm(n[0]) - 1.0) < 1e-6


def test_masked_mean_pool():
    embs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    mask = np.array([1, 1, 0])
    pooled = masked_mean_pool(embs, mask)
    # mean over first two tokens only
    assert abs(pooled[0] - 2.0) < 1e-6
    assert abs(pooled[1] - 3.0) < 1e-6
    assert pooled.shape == (2,)


def test_pipeline_timer_separates_phases():
    timer = PipelineTimer("test")
    with timer.index_phase():
        time.sleep(0.05)
    with timer.query_phase():
        time.sleep(0.05)
    record = timer.record(n_queries=2)
    # Both phases should record positive time (robust across OS timer resolutions)
    assert record["index_s"] > 0.0
    assert record["query_s"] > 0.0
    assert record["per_query_ms"] is not None and record["per_query_ms"] > 0.0
    # total_s is round(index_s + query_s, 3) directly; comparing against the
    # sum of already-rounded components can differ by up to 1e-3, so use tolerance.
    assert abs(record["total_s"] - (record["index_s"] + record["query_s"])) < 0.002


def test_pipeline_timer_accumulates():
    timer = PipelineTimer("test")
    with timer.index_phase():
        time.sleep(0.01)
    with timer.index_phase():
        time.sleep(0.01)
    assert timer.index_s >= 0.02


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("All utils tests passed.")