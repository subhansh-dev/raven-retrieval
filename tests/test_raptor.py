import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Clustering depends on umap; skip the whole module gracefully if absent
# (collection shouldn't hard-fail in minimal environments).
pytest.importorskip("umap")

from src.raptor.chunker import TextChunker
from src.raptor.clustering import (
    reduce_dimensions, select_cluster_count, soft_cluster,
    global_local_cluster, compute_soft_assignment_rate
)
from src.raptor.tree import TreeNode, RaptorTree


def test_chunker():
    chunker = TextChunker(chunk_size=10, overlap=0)
    text = " ".join([f"word{i}" for i in range(25)])
    chunks = chunker.chunk(text, doc_id="doc1")
    assert len(chunks) == 3
    assert chunks[0]["id"] == "doc1::l0::0"
    assert chunks[0]["token_count"] == 10
    assert chunks[1]["token_count"] == 10
    assert chunks[2]["token_count"] == 5


def test_dimensionality_reduction():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(50, 32)
    reduced = reduce_dimensions(embeddings, n_neighbors=7, n_components=5)
    assert reduced.shape == (50, 5)


def test_cluster_count_selection():
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(20, 10) + np.array([5.0] * 10)
    cluster2 = rng.randn(20, 10) + np.array([-5.0] * 10)
    embeddings = np.vstack([cluster1, cluster2])
    k = select_cluster_count(embeddings, min_k=2, max_k=5)
    assert k >= 2


def test_soft_cluster():
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(15, 8) + np.array([5.0] * 8)
    cluster2 = rng.randn(15, 8) + np.array([-5.0] * 8)
    embeddings = np.vstack([cluster1, cluster2])
    reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=3)
    clusters = soft_cluster(reduced, n_clusters=2, threshold=0.1)
    assert len(clusters) >= 2
    total_assigned = sum(len(v) for v in clusters.values())
    assert total_assigned > 0


def test_tree_operations():
    tree = RaptorTree()
    n1 = TreeNode("n1", "text1", np.array([[1.0, 0.0]]), level=0)
    n2 = TreeNode("n2", "text2", np.array([[0.0, 1.0]]), level=0)
    n3 = TreeNode("n3", "summary", np.array([[0.5, 0.5]]), level=1)
    n3.add_child(n1)
    n3.add_child(n2)
    tree.add_node(n1)
    tree.add_node(n2)
    tree.add_node(n3)
    tree.root_ids = ["n3"]
    assert tree.get_max_level() == 1
    assert len(tree.get_level(0)) == 2
    assert len(tree.get_leaf_nodes()) == 2
    path = tree.get_node_path("n1")
    assert len(path) == 2
    assert path[0].node_id == "n3"
    assert path[1].node_id == "n1"


def test_soft_assignment_rate():
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(20, 8) + np.array([10.0] * 8)
    cluster2 = rng.randn(20, 8) + np.array([-10.0] * 8)
    embeddings = np.vstack([cluster1, cluster2])
    rate = compute_soft_assignment_rate(embeddings, n_clusters=2, threshold=0.1)
    assert 0.0 <= rate <= 1.0


if __name__ == "__main__":
    test_chunker()
    test_dimensionality_reduction()
    test_cluster_count_selection()
    test_soft_cluster()
    test_tree_operations()
    test_soft_assignment_rate()
    print("All RAPTOR tests passed.")
