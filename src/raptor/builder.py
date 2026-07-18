import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import TextChunker
from .clustering import global_local_cluster, compute_soft_assignment_rate
from .tree import TreeNode, RaptorTree
from .summarizer import LLMSummarizer


class RaptorBuilder:

    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", summarizer_model="facebook/bart-large-cnn",
                 chunk_size=100, soft_threshold=0.1, min_cluster_size=3, use_extractive_fallback=True):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.summarizer = LLMSummarizer(model_name=summarizer_model, fallback_to_extractive=use_extractive_fallback)
        self.chunker = TextChunker(chunk_size=chunk_size)
        self.soft_threshold = soft_threshold
        self.min_cluster_size = min_cluster_size

    def embed_texts(self, texts):
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, batch_size=64)
        return np.array(embeddings)

    def build(self, corpus):
        chunks = self.chunker.chunk_corpus(corpus)
        tree = RaptorTree()
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.embed_texts(chunk_texts)
        for i, chunk in enumerate(chunks):
            node = TreeNode(
                node_id=chunk["id"],
                text=chunk["text"],
                embeddings=chunk_embeddings[i:i+1],
                level=0,
                doc_id=chunk["doc_id"],
            )
            tree.add_node(node)
        current_level = 0
        current_nodes = [tree.nodes[cid] for cid in tree.get_level(0)]
        while len(current_nodes) > self.min_cluster_size:
            embeddings = np.array([n.pooled_embedding for n in current_nodes])
            clusters = global_local_cluster(embeddings, soft_threshold=self.soft_threshold)
            if len(clusters) <= 1:
                break
            next_level_nodes = []
            cluster_id = 0
            for _, member_indices in clusters.items():
                cluster_nodes = [current_nodes[i] for i in member_indices]
                cluster_texts = [n.text for n in cluster_nodes]
                summary = self.summarizer.summarize_cluster(cluster_texts)
                if not summary:
                    summary = cluster_texts[0]
                summary_embedding = self.embed_texts([summary])
                node_id = self.summarizer.generate_node_id(summary, current_level + 1, cluster_id)
                parent_node = TreeNode(
                    node_id=node_id,
                    text=summary,
                    embeddings=summary_embedding,
                    level=current_level + 1,
                )
                tree.add_node(parent_node)
                for child in cluster_nodes:
                    parent_node.add_child(child)
                next_level_nodes.append(parent_node)
                cluster_id += 1
            current_level += 1
            current_nodes = next_level_nodes
        max_level = tree.get_max_level()
        tree.root_ids = tree.get_level(max_level)
        return tree

    def measure_soft_assignment(self, corpus):
        chunks = self.chunker.chunk_corpus(corpus)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = self.embed_texts(chunk_texts)
        rate = compute_soft_assignment_rate(chunk_embeddings, threshold=self.soft_threshold)
        return rate
