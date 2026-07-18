import numpy as np
import torch

from ..raptor.chunker import TextChunker
from ..raptor.clustering import global_local_cluster, compute_soft_assignment_rate
from ..raptor.tree import TreeNode, RaptorTree
from ..raptor.summarizer import LLMSummarizer
from ..maxsim.brute_force import maxsim_score


class LateInteractionRaptor:

    def __init__(self, colbert_encoder, embedding_model=None, summarizer_model="facebook/bart-large-cnn",
                 chunk_size=100, soft_threshold=0.1, min_cluster_size=3):
        self.encoder = colbert_encoder
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model
        self.summarizer = LLMSummarizer(model_name=summarizer_model, fallback_to_extractive=True)
        self.chunker = TextChunker(chunk_size=chunk_size)
        self.soft_threshold = soft_threshold
        self.min_cluster_size = min_cluster_size
        self.tree = None

    def build(self, corpus):
        chunks = self.chunker.chunk_corpus(corpus)
        tree = RaptorTree()
        chunk_texts = [c["text"] for c in chunks]
        pooled_embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False, batch_size=64)
        pooled_embeddings = np.array(pooled_embeddings)
        device = next(self.encoder.parameters()).device
        self.encoder.eval()
        with torch.no_grad():
            token_embs = self.encoder.forward(chunk_texts, max_length=256)
            token_embs_np = token_embs.cpu().numpy()
        for i, chunk in enumerate(chunks):
            node = TreeNode(
                node_id=chunk["id"],
                text=chunk["text"],
                embeddings=token_embs_np[i],
                level=0,
                doc_id=chunk["doc_id"],
            )
            tree.add_node(node)
        current_level = 0
        current_nodes = [tree.nodes[cid] for cid in tree.get_level(0)]
        while len(current_nodes) > self.min_cluster_size:
            pooled = np.array([n.pooled_embedding for n in current_nodes])
            clusters = global_local_cluster(pooled, soft_threshold=self.soft_threshold)
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
                summary_pooled = self.embedding_model.encode([summary])
                summary_pooled = np.array(summary_pooled)
                with torch.no_grad():
                    summary_token_embs = self.encoder.forward([summary], max_length=256)
                    summary_token_embs_np = summary_token_embs.cpu().numpy()
                node_id = self.summarizer.generate_node_id(summary, current_level + 1, cluster_id)
                parent_node = TreeNode(
                    node_id=node_id,
                    text=summary,
                    embeddings=summary_token_embs_np[0],
                    level=current_level + 1,
                )
                parent_node.pooled_embedding = summary_pooled[0]
                tree.add_node(parent_node)
                for child in cluster_nodes:
                    parent_node.add_child(child)
                next_level_nodes.append(parent_node)
                cluster_id += 1
            current_level += 1
            current_nodes = next_level_nodes
        max_level = tree.get_max_level()
        tree.root_ids = tree.get_level(max_level)
        self.tree = tree
        return tree

    def retrieve_collapsed(self, query_text, top_k=10):
        if self.tree is None:
            raise ValueError("Tree not built. Call build() first.")
        with torch.no_grad():
            query_embs = self.encoder.encode_query(query_text)
            query_embs_np = query_embs.cpu().numpy()
        all_nodes = self.tree.get_all_nodes_flat()
        scored = []
        for node in all_nodes:
            score = maxsim_score(query_embs_np, node.embeddings)
            scored.append((node.node_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_traversal(self, query_text, top_k=10):
        if self.tree is None:
            raise ValueError("Tree not built. Call build() first.")
        if not self.tree.root_ids:
            return []
        with torch.no_grad():
            query_embs = self.encoder.encode_query(query_text)
            query_embs_np = query_embs.cpu().numpy()
        selected = []
        current_ids = self.tree.root_ids
        while current_ids:
            scores = []
            for nid in current_ids:
                node = self.tree.nodes[nid]
                score = maxsim_score(query_embs_np, node.embeddings)
                scores.append((nid, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_current = scores[:top_k]
            selected.extend(top_current)
            next_ids = []
            for nid, _ in top_current:
                node = self.tree.nodes[nid]
                for child in node.children:
                    next_ids.append(child.node_id)
            current_ids = next_ids
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[:top_k]

    def retrieve(self, query_text, strategy="collapsed", top_k=10):
        if strategy == "collapsed":
            return self.retrieve_collapsed(query_text, top_k=top_k)
        elif strategy == "traversal":
            return self.retrieve_traversal(query_text, top_k=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
