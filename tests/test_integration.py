import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.maxsim.brute_force import maxsim_score, brute_force_rank
from src.raptor.chunker import TextChunker
from src.raptor.tree import TreeNode, RaptorTree


def make_toy_corpus():
    return [
        {"_id": "doc1", "title": "Cats", "text": "Cats are small domesticated carnivorous mammals. They are often called house cats when kept as indoor pets. Cats are valued by humans for companionship."},
        {"_id": "doc2", "title": "Dogs", "text": "Dogs are domesticated carnivores of the family Canidae. They have been selectively bred over millennia for various behaviors and sensory capabilities."},
        {"_id": "doc3", "title": "Birds", "text": "Birds are a group of warm-blooded vertebrates constituting the class Aves. They are characterized by feathers, toothless beaked jaws, and a high metabolic rate."},
        {"_id": "doc4", "title": "Fish", "text": "Fish are aquatic craniate animals that lack limbs with digits. They form a sister group to the tunicates. They are cold-blooded and have gills."},
    ]


def test_pipeline_wiring():
    corpus = make_toy_corpus()
    chunker = TextChunker(chunk_size=20, overlap=0)
    chunks = chunker.chunk_corpus(corpus)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert "level" in chunk
        assert chunk["level"] == 0


def test_tree_flat_retrieval():
    rng = np.random.RandomState(42)
    tree = RaptorTree()
    for i in range(5):
        emb = rng.randn(3, 8).astype(np.float32)
        node = TreeNode(f"node_{i}", f"text {i}", emb, level=0)
        tree.add_node(node)
    summary_emb = rng.randn(2, 8).astype(np.float32)
    summary = TreeNode("summary_0", "summary text", summary_emb, level=1)
    for nid in [f"node_{i}" for i in range(3)]:
        summary.add_child(tree.nodes[nid])
    tree.add_node(summary)
    tree.root_ids = ["summary_0"]
    assert tree.get_max_level() == 1
    all_nodes = tree.get_all_nodes_flat()
    assert len(all_nodes) == 6
    leaves = tree.get_leaf_nodes()
    assert len(leaves) == 5


def test_maxsim_end_to_end():
    rng = np.random.RandomState(42)
    dim = 16
    query_emb = rng.randn(4, dim).astype(np.float32)
    doc_embs = [rng.randn(8, dim).astype(np.float32) for _ in range(10)]
    ranked = brute_force_rank(query_emb, doc_embs, top_k=5)
    assert len(ranked) == 5
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i+1][1]
    scores = [score for _, score in ranked]
    assert all(s >= 0 for s in scores)


if __name__ == "__main__":
    test_pipeline_wiring()
    test_tree_flat_retrieval()
    test_maxsim_end_to_end()
    print("All integration tests passed.")
