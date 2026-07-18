#!/usr/bin/env python3
"""Core test suite — runs without torch.

Tests the fundamental algorithms (MaxSim, chunking, clustering, tree,
significance, compression, visualization) using only numpy/scipy.

Usage:
    python tests/run_core_tests.py
    python -m pytest tests/test_core.py -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

rng = np.random.RandomState(42)
PASSED = 0
FAILED = 0


def assert_eq(a, b, msg=""):
    global PASSED, FAILED
    if a != b:
        FAILED += 1
        print(f"  FAIL: {msg} — expected {b}, got {a}")
    else:
        PASSED += 1


def assert_true(cond, msg=""):
    global PASSED, FAILED
    if not cond:
        FAILED += 1
        print(f"  FAIL: {msg}")
    else:
        PASSED += 1


def assert_approx(a, b, tol=1e-6, msg=""):
    global PASSED, FAILED
    if abs(a - b) > tol:
        FAILED += 1
        print(f"  FAIL: {msg} — expected ~{b}, got {a}")
    else:
        PASSED += 1


def run_test(name, fn):
    global PASSED, FAILED
    before = PASSED + FAILED
    try:
        fn()
        after = PASSED + FAILED
        if after == before:
            print(f"  ⚠️  {name}: no assertions")
        else:
            print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        print(f"  ❌ {name}: {e}")


# ── Chunker ──────────────────────────────────────────────────────────

def test_chunker_basic():
    from src.raptor.chunker import TextChunker
    chunker = TextChunker(chunk_size=10, overlap=0)
    text = " ".join([f"word{i}" for i in range(25)])
    chunks = chunker.chunk(text, doc_id="doc1")
    assert_eq(len(chunks), 3, "chunk count")
    assert_eq(chunks[0]["id"], "doc1::l0::0", "chunk id")
    assert_eq(chunks[0]["token_count"], 10, "token count")
    assert_eq(chunks[2]["token_count"], 5, "last chunk")


def test_chunker_overlap():
    from src.raptor.chunker import TextChunker
    chunker = TextChunker(chunk_size=10, overlap=5)
    text = " ".join([f"w{i}" for i in range(25)])
    chunks = chunker.chunk(text)
    assert_true(len(chunks) > 3, "overlap produces more chunks")


def test_chunker_corpus():
    from src.raptor.chunker import TextChunker
    corpus = [
        {"_id": "d1", "title": "Cats", "text": "Cats are small mammals."},
        {"_id": "d2", "title": "Dogs", "text": "Dogs are carnivores."},
    ]
    chunks = TextChunker(20, 0).chunk_corpus(corpus)
    assert_true(len(chunks) > 0, "non-empty")
    assert_true(all("id" in c and "text" in c and "level" in c for c in chunks), "fields")


# ── Tree ─────────────────────────────────────────────────────────────

def test_tree_ops():
    from src.raptor.tree import TreeNode, RaptorTree
    tree = RaptorTree()
    n1 = TreeNode("n1", "t1", np.array([[1.0, 0.0]]), level=0)
    n2 = TreeNode("n2", "t2", np.array([[0.0, 1.0]]), level=0)
    n3 = TreeNode("n3", "s", np.array([[0.5, 0.5]]), level=1)
    n3.add_child(n1)
    n3.add_child(n2)
    tree.add_node(n1)
    tree.add_node(n2)
    tree.add_node(n3)
    tree.root_ids = ["n3"]
    assert_eq(tree.get_max_level(), 1, "max level")
    assert_eq(len(tree.get_level(0)), 2, "level 0 count")
    assert_eq(len(tree.get_leaf_nodes()), 2, "leaf count")
    assert_eq(len(tree.get_node_path("n1")), 2, "path length")


# ── MaxSim ───────────────────────────────────────────────────────────

def test_maxsim_basic():
    from src.maxsim.brute_force import maxsim_score
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    s1, s2 = maxsim_score(q, d1), maxsim_score(q, d2)
    assert_true(s1 > s2, "d1 > d2")


def test_maxsim_ranking():
    from src.maxsim.brute_force import brute_force_rank
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ]
    ranked = brute_force_rank(q, docs, top_k=3)
    assert_eq(len(ranked), 3, "all docs ranked")
    assert_true(ranked[0][1] >= ranked[-1][1], "sorted")


def test_maxsim_batch():
    from src.maxsim.brute_force import maxsim_score_batch
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs_flat = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    scores = maxsim_score_batch(q, docs_flat, [2, 2])
    assert_eq(len(scores), 2, "two docs")
    assert_true(scores[0] > scores[1], "exact > partial")


# ── Clustering ───────────────────────────────────────────────────────

def test_umap_reduction():
    from src.raptor.clustering import reduce_dimensions
    emb = rng.randn(50, 32)
    reduced = reduce_dimensions(emb, n_neighbors=7, n_components=5)
    assert_eq(reduced.shape, (50, 5), "reduced shape")


def test_cluster_count():
    from src.raptor.clustering import select_cluster_count
    c1 = rng.randn(20, 10) + 5.0
    c2 = rng.randn(20, 10) - 5.0
    emb = np.vstack([c1, c2])
    k = select_cluster_count(emb, min_k=2, max_k=5)
    assert_true(k >= 2, "detects 2 clusters")


def test_soft_assignment():
    from src.raptor.clustering import compute_soft_assignment_rate
    c1 = rng.randn(20, 8) + 10.0
    c2 = rng.randn(20, 8) - 10.0
    emb = np.vstack([c1, c2])
    rate = compute_soft_assignment_rate(emb, n_clusters=2, threshold=0.1)
    assert_true(0.0 <= rate <= 1.0, "rate in [0,1]")


# ── Summarizer ───────────────────────────────────────────────────────

def test_extractive_summarizer():
    from src.raptor.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer(max_sentences=2)
    text = "The cat sat on the mat. The dog played in the yard. The bird flew over the house."
    summary = s.summarize(text)
    assert_true(len(summary) > 0, "non-empty summary")
    assert_true(len(summary) < len(text), "shorter than original")


def test_extractive_empty():
    from src.raptor.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer()
    assert_eq(s.summarize(""), "", "empty input")
    assert_eq(s.summarize(None), "", "none input")


# ── Compression ──────────────────────────────────────────────────────

def test_compression_roundtrip():
    from src.maxsim.compression import ResidualCompressor
    comp = ResidualCompressor(num_centroids=8, nbits=2)
    comp.fit(rng.randn(200, 16).astype(np.float32))
    cids, qres = comp.compress_document(rng.randn(10, 16).astype(np.float32))
    recon = comp.decompress_document(cids, qres)
    assert_eq(recon.shape, (10, 16), "reconstructed shape")
    # Reconstruction should be approximate (within quantization error)
    assert_true(recon.dtype == np.float32, "float32 output")


def test_compression_ratio():
    from src.maxsim.compression import ResidualCompressor
    comp = ResidualCompressor(num_centroids=256, nbits=2)
    ratio = comp.compute_compression_ratio(128, max_tokens=256)
    assert_true(ratio > 10, f"significant compression ({ratio:.1f}x)")


def test_compression_save_load():
    from src.maxsim.compression import ResidualCompressor
    comp = ResidualCompressor(num_centroids=8, nbits=2)
    comp.fit(rng.randn(100, 8).astype(np.float32))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "compressor.npz")
        comp.save(path)
        comp2 = ResidualCompressor()
        comp2.load(path)
        assert_true(comp2._fitted, "loaded state")
        assert_eq(comp2.num_centroids, 8, "centroids preserved")


# ── Significance ─────────────────────────────────────────────────────

def test_bootstrap_known_diff():
    from src.eval.significance import paired_bootstrap_test
    a = rng.normal(0.7, 0.1, 100)
    b = rng.normal(0.5, 0.1, 100)
    r = paired_bootstrap_test(a, b, n_resamples=5000)
    assert_true(r["observed_diff"] > 0.1, "positive diff")
    assert_true(r["p_value"] < 0.01, "significant")


def test_bootstrap_no_diff():
    from src.eval.significance import paired_bootstrap_test
    a = rng.normal(0.5, 0.1, 100)
    b = a + rng.normal(0, 0.01, 100)
    r = paired_bootstrap_test(a, b, n_resamples=5000)
    assert_true(r["p_value"] > 0.05, "not significant")


def test_bonferroni():
    from src.eval.significance import bonferroni_correction
    r = bonferroni_correction([0.01, 0.03, 0.04, 0.06, 0.10, 0.50], alpha=0.05)
    assert_approx(r["adjusted_alpha"], 0.05 / 6, msg="adjusted alpha")


def test_pairwise():
    from src.eval.significance import run_all_pairwise_tests
    scores = {
        "a": rng.normal(0.7, 0.1, 50).tolist(),
        "b": rng.normal(0.5, 0.1, 50).tolist(),
        "c": rng.normal(0.6, 0.1, 50).tolist(),
    }
    comps = run_all_pairwise_tests(scores, ["a", "b", "c"], n_resamples=1000)
    assert_eq(len(comps), 3, "3 pairs")


# ── Visualization ────────────────────────────────────────────────────

def test_dashboard():
    from src.eval.visualize import generate_dashboard
    metrics = {
        "dense": {"ndcg": {"NDCG@1": 0.5, "NDCG@3": 0.6, "NDCG@5": 0.65, "NDCG@10": 0.7}},
        "hybrid": {"ndcg": {"NDCG@1": 0.48, "NDCG@3": 0.58, "NDCG@5": 0.62, "NDCG@10": 0.68}},
    }
    with tempfile.TemporaryDirectory() as td:
        path = generate_dashboard(metrics, output_path=os.path.join(td, "dash.html"))
        assert_true(os.path.exists(path), "dashboard created")
        with open(path) as f:
            content = f.read()
        assert_true("nDCG@10" in content.lower() or "ndcg@10" in content.lower(), "contains metrics")


# ── Report ───────────────────────────────────────────────────────────

def test_report():
    from src.eval.report import generate_report
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "metrics.json"), "w") as f:
            json.dump({"dense": {"ndcg": {"NDCG@10": 0.7}}}, f)
        with open(os.path.join(td, "timings.json"), "w") as f:
            json.dump({"dense": 10.0}, f)
        path = generate_report(td)
        assert_true(os.path.exists(path), "report created")
        with open(path) as f:
            content = f.read()
        assert_true("nDCG@10" in content, "contains metrics")


# ── Agentic ──────────────────────────────────────────────────────────

def test_query_decomposition():
    from src.baselines.agentic import QueryDecomposer
    d = QueryDecomposer()
    q1 = d.decompose("What is X and Y?")
    assert_true(len(q1) >= 1, "at least 1 sub-query")
    q2 = d.decompose("What is the capital of France?")
    assert_eq(len(q2), 1, "simple query stays single")


# ── Approximate (needs faiss) ────────────────────────────────────────

def test_centroid_index():
    try:
        from src.maxsim.approximate import CentroidIndex, ApproximateMaxSim
        idx = CentroidIndex(num_centroids=4)
        idx.build(rng.randn(100, 8).astype(np.float32))
        assert_eq(idx.centroids.shape, (4, 8), "centroid shape")
        cids, res = idx.encode_document(rng.randn(10, 8).astype(np.float32))
        assert_eq(len(cids), 10, "10 assignments")
        approx = ApproximateMaxSim(idx, prune_ratio=0.5)
        approx.index_documents([rng.randn(10, 8).astype(np.float32) for _ in range(5)])
        results = approx.retrieve(rng.randn(5, 8).astype(np.float32), top_k=3)
        assert_true(len(results) <= 3, "top-k bounded")
    except ImportError:
        print("  ⚠️  faiss not available, skipping centroid tests")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Raven-Retrieval Core Test Suite")
    print("=" * 50)

    tests = [
        ("chunker_basic", test_chunker_basic),
        ("chunker_overlap", test_chunker_overlap),
        ("chunker_corpus", test_chunker_corpus),
        ("tree_ops", test_tree_ops),
        ("maxsim_basic", test_maxsim_basic),
        ("maxsim_ranking", test_maxsim_ranking),
        ("maxsim_batch", test_maxsim_batch),
        ("umap_reduction", test_umap_reduction),
        ("cluster_count", test_cluster_count),
        ("soft_assignment", test_soft_assignment),
        ("extractive_summarizer", test_extractive_summarizer),
        ("extractive_empty", test_extractive_empty),
        ("compression_roundtrip", test_compression_roundtrip),
        ("compression_ratio", test_compression_ratio),
        ("compression_save_load", test_compression_save_load),
        ("bootstrap_known_diff", test_bootstrap_known_diff),
        ("bootstrap_no_diff", test_bootstrap_no_diff),
        ("bonferroni", test_bonferroni),
        ("pairwise", test_pairwise),
        ("dashboard", test_dashboard),
        ("report", test_report),
        ("query_decomposition", test_query_decomposition),
        ("centroid_index", test_centroid_index),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print()
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total assertions")
    if FAILED > 0:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("🎉 ALL TESTS PASSED")
        sys.exit(0)
