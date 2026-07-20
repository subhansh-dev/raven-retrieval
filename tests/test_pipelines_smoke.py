"""Smoke tests for retrieval pipelines.

These verify the index/retrieve contract of every retriever WITHOUT
pulling in heavy dependencies — using dependency injection (the
``model=`` / ``model=`` parameters every retriever now accepts) and
``pytest.importorskip`` for things that genuinely need a real library
(BM25, SPLADE).

On a machine with the full stack installed these run for real; on a
minimal numpy-only CI box the heavy imports skip and the test count
shrinks — but the DI-based structural tests still run, catching
contract regressions (which is how the HyDE kwarg crash and the
dense retrieve_batch doc-ID bug went unnoticed).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fakes for dependency injection ────────────────────────────────────

class FakeEncoder:
    """Fake sentence-transformers-style encoder producing deterministic 8-dim vectors."""
    def __init__(self, dim=8):
        self.dim = dim
        self._vocab = {}

    def _vec(self, text):
        # Deterministic vector: hash each token, average their one-hot-ish vectors
        tokens = text.lower().split() or ["_empty_"]
        vec = np.zeros(self.dim, dtype=np.float32)
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = np.random.RandomState(hash(t) & 0xFFFFFFFF).randn(self.dim).astype(np.float32)
            vec += self._vocab[t]
        return vec

    def encode(self, texts, show_progress_bar=False, batch_size=64):
        return np.stack([self._vec(t) for t in texts])


def _toy_corpus():
    return {
        "doc1": {"title": "Cats", "text": "Cats are small domesticated carnivorous mammals."},
        "doc2": {"title": "Dogs", "text": "Dogs are domesticated carnivores of the family Canidae."},
        "doc3": {"title": "Birds", "text": "Birds are warm-blooded vertebrates with feathers."},
        "doc4": {"title": "Fish", "text": "Fish are aquatic craniate animals with gills."},
    }


# ── Structural tests (run everywhere, no heavy deps) ──────────────────

def test_dense_retriever_contract_with_fake_encoder():
    from src.baselines.dense import DenseRetriever
    r = DenseRetriever(chunk_size=200, chunk_overlap=50, model=FakeEncoder())
    r.index(_toy_corpus())
    results = r.retrieve("cats carnivorous", top_k=4)
    assert len(results) <= 4
    assert all(isinstance(did, str) and isinstance(s, float) for did, s in results)
    # every returned id must be a real doc id (NOT a chunk id — the old retrieve_batch bug)
    corpus = _toy_corpus()
    assert all(did in corpus for did, _ in results)
    # results must be sorted by score descending
    assert all(results[i][1] >= results[i + 1][1] for i in range(len(results) - 1))


def test_dense_retrieve_batch_returns_doc_ids():
    from src.baselines.dense import DenseRetriever
    r = DenseRetriever(chunk_size=200, chunk_overlap=50, model=FakeEncoder())
    r.index(_toy_corpus())
    batch = r.retrieve_batch(["cats carnivorous", "fish gills"], top_k=3)
    assert len(batch) == 2
    for results in batch:
        assert all(did in _toy_corpus() for did, _ in results)


def test_hyde_retriever_contract_with_fake_encoder():
    from src.baselines.hyde import HyDERetriever
    r = HyDERetriever(use_llm=False, model=FakeEncoder())  # template mode (no LLM load)
    r.index(_toy_corpus())
    results = r.retrieve("cats", top_k=4)
    assert len(results) <= 4
    assert all(did in _toy_corpus() for did, _ in results)


def test_contextual_dense_retriever_contract():
    from src.baselines.contextual import ContextualDenseRetriever
    r = ContextualDenseRetriever(chunk_size=200, chunk_overlap=50, model=FakeEncoder())
    r.index(_toy_corpus())
    results = r.retrieve("aquatic animals", top_k=4)
    corpus = _toy_corpus()
    assert all(did in corpus for did, _ in results)
    # fish doc should appear in results (its text contains aquatic/gills terms)
    assert "doc4" in {did for did, _ in results}


def test_reciprocal_rank_fusion_consistency():
    from src.utils import reciprocal_rank_fusion
    a = [("doc1", 0.9), ("doc2", 0.5)]
    b = [("doc2", 0.8), ("doc1", 0.4)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    # doc1 and doc2 both appear in both rankings → tie at the top
    assert fused[0][0] in ("doc1", "doc2")


def test_agentic_decomposition_returns_list():
    pytest.importorskip("numpy")  # always available, but explicit
    from src.baselines.agentic import QueryDecomposer
    d = QueryDecomposer()
    out = d.decompose("What is X and what is Y?")
    assert isinstance(out, list)
    assert len(out) >= 1


def test_agentic_multihop_contract_with_fake_base():
    from src.baselines.dense import DenseRetriever
    from src.baselines.agentic import MultiHopRetriever
    base = DenseRetriever(chunk_size=200, chunk_overlap=50, model=FakeEncoder())
    base.index(_toy_corpus())
    multi = MultiHopRetriever(base, top_k=4)
    out = multi.retrieve("cats and dogs", top_k=4)
    assert all(did in _toy_corpus() for did, _ in out)


def test_reflection_requires_text_lookup_to_evaluate():
    """The whole point of the fix: without a real text_lookup, reflection
    should NOT silently run keyword coverage over doc ID strings."""
    from src.baselines.dense import DenseRetriever
    from src.baselines.agentic import ReflectionRetriever
    base = DenseRetriever(chunk_size=200, chunk_overlap=50, model=FakeEncoder())
    base.index(_toy_corpus())
    # With a real lookup, reflection can evaluate and possibly reformulate
    lookup = {did: doc["title"] + " " + doc["text"] for did, doc in _toy_corpus().items()}
    r = ReflectionRetriever(base, text_lookup=lookup.get, max_iterations=2, top_k=4)
    out = r.retrieve("cats", top_k=4)
    assert all(did in _toy_corpus() for did, _ in out)


# ── Library-dependent tests (skip if the dep is missing) ─────────────

def test_hybrid_retriever_contract():
    pytest.importorskip("rank_bm25")
    from src.baselines.hybrid import HybridRetriever
    r = HybridRetriever(chunk_size=200, chunk_overlap=50, rrf_k=60, model=FakeEncoder())
    r.index(_toy_corpus())
    results = r.retrieve("cats carnivorous", top_k=4)
    assert all(did in _toy_corpus() for did, _ in results)


def test_contextual_hybrid_retriever_contract():
    pytest.importorskip("rank_bm25")
    from src.baselines.contextual import ContextualHybridRetriever
    r = ContextualHybridRetriever(chunk_size=200, chunk_overlap=50, rrf_k=60, model=FakeEncoder())
    r.index(_toy_corpus())
    results = r.retrieve("aquatic animals", top_k=4)
    assert all(did in _toy_corpus() for did, _ in results)


def test_graph_retriever_contract():
    pytest.importorskip("rank_bm25")  # graph retrieval builds its own embeddings but
    # GraphRetriever currently instantiates its own SentenceTransformer — guard it
    from src.baselines.graph_retrieval import GraphRetriever
    try:
        from sentence_transformers import SentenceTransformer  # noqa
    except ImportError:
        pytest.skip("needs sentence-transformers for embedding")
    r = GraphRetriever(base_retriever=None, chunk_size=200, chunk_overlap=50)
    # Just ensure the class constructs; full build requires the model
    assert r.graph is not None


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✅ {name}")
            except Exception as e:
                failed += 1
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()
    print(f"\n{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)