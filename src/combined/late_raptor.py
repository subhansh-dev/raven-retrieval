"""LateInteractionRaptor — the novel pipeline.

RAPTOR hierarchical summary tree where EVERY node (leaf chunks and summary
nodes) stores ColBERT per-token embeddings. Retrieval scores nodes with
MaxSim late interaction instead of single-vector cosine.

Fixes vs. the original implementation:
- Batched ColBERT encoding (the old code encoded ALL chunks in ONE forward
  pass — instant OOM on any real corpus)
- Padding tokens are stripped via the attention mask before storage
  (the old code stored 256-token padded tensors, so MaxSim matched against
  [PAD] embeddings and clustering pooled over pads)
- Level-0 clustering uses SBERT pooled embeddings (the old code computed
  them, threw them away, then clustered mean-of-ColBERT-tokens instead —
  a different, noisier space than the summary levels used)
"""

import numpy as np
import torch
import logging

from ..raptor.chunker import TextChunker
from ..raptor.clustering import global_local_cluster
from ..raptor.tree import TreeNode, RaptorTree
from ..raptor.summarizer import LLMSummarizer
from ..maxsim.brute_force import pack_doc_embeddings, brute_force_rank_fast

logger = logging.getLogger(__name__)


class LateInteractionRaptor:

    def __init__(self, colbert_encoder, embedding_model=None, summarizer=None,
                 summarizer_model="facebook/bart-large-cnn",
                 chunk_size=100, soft_threshold=0.1, min_cluster_size=3, encode_batch_size=32):
        self.encoder = colbert_encoder
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model
        if summarizer is not None:
            self.summarizer = summarizer
        else:
            self.summarizer = LLMSummarizer(model_name=summarizer_model, fallback_to_extractive=True)
        self.chunker = TextChunker(chunk_size=chunk_size)
        self.soft_threshold = soft_threshold
        self.min_cluster_size = min_cluster_size
        self.encode_batch_size = encode_batch_size
        self.tree = None

    def _encode_colbert(self, texts, max_length=256):
        """Batched, mask-trimmed ColBERT encoding. Returns list of (n_i, dim) arrays."""
        return self.encoder.encode_documents(
            texts, max_length=max_length, batch_size=self.encode_batch_size
        )

    def build(self, corpus):
        chunks = self.chunker.chunk_corpus(corpus)
        tree = RaptorTree()
        chunk_texts = [c["text"] for c in chunks]

        # SBERT pooled embeddings: used for CLUSTERING at every level
        pooled_embeddings = np.array(self.embedding_model.encode(
            chunk_texts, show_progress_bar=False, batch_size=64
        ))

        # ColBERT token embeddings: used for SCORING (MaxSim) — batched, no pads
        logger.info(f"Encoding {len(chunk_texts)} chunks with ColBERT (batch_size={self.encode_batch_size})...")
        self.encoder.eval()
        token_embs = self._encode_colbert(chunk_texts)

        for i, chunk in enumerate(chunks):
            node = TreeNode(
                node_id=chunk["id"],
                text=chunk["text"],
                embeddings=token_embs[i],
                level=0,
                doc_id=chunk["doc_id"],
            )
            # Cluster in SBERT space (consistent with summary levels)
            node.pooled_embedding = pooled_embeddings[i]
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

                summary_pooled = np.array(self.embedding_model.encode([summary]))[0]
                summary_token_embs = self._encode_colbert([summary])[0]

                node_id = self.summarizer.generate_node_id(summary, current_level + 1, cluster_id)
                parent_node = TreeNode(
                    node_id=node_id,
                    text=summary,
                    embeddings=summary_token_embs,
                    level=current_level + 1,
                )
                parent_node.pooled_embedding = summary_pooled
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
        logger.info(f"Tree built: {len(tree.nodes)} nodes, {max_level + 1} levels")
        return tree

    def _encode_query(self, query_text):
        with torch.no_grad():
            query_embs = self.encoder.encode_query(query_text)
        query_embs_np = query_embs.detach().cpu().numpy()
        if query_embs_np.ndim == 3:
            query_embs_np = query_embs_np.squeeze(0)
        return query_embs_np

    def retrieve_collapsed(self, query_text, top_k=10):
        """Score every node in the tree with MaxSim, take top-k.

        Uses batched scoring (brute_force_rank_fast) instead of calling
        maxsim_score per node — much faster for trees with many nodes.
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call build() first.")
        query_embs_np = self._encode_query(query_text)

        # Pack all node embeddings for fast batched scoring
        all_nodes = self.tree.get_all_nodes_flat()
        node_embs = [n.embeddings for n in all_nodes]
        flat_embs, doc_lengths = pack_doc_embeddings(node_embs)

        ranked = brute_force_rank_fast(query_embs_np, flat_embs, doc_lengths, top_k=top_k)

        # Map back from index to node_id
        scored = [(all_nodes[idx].node_id, score) for idx, score in ranked]
        return scored

    def retrieve_traversal(self, query_text, top_k=10):
        """Top-down traversal: score roots, take top-k, descend to children.

        Uses batched MaxSim scoring per level instead of per-node Python loop.
        Each level is scored in one matmul via brute_force_rank_fast, then
        we descend to children of top-k nodes. Repeat until leaves.
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call build() first.")
        if not self.tree.root_ids:
            return []
        query_embs_np = self._encode_query(query_text)
        selected = []
        current_ids = self.tree.root_ids

        while current_ids:
            # Batch-score all nodes at this level
            level_nodes = [self.tree.nodes[nid] for nid in current_ids]
            level_embs = [n.embeddings for n in level_nodes]
            flat_embs, doc_lengths = pack_doc_embeddings(level_embs)
            ranked = brute_force_rank_fast(query_embs_np, flat_embs, doc_lengths,
                                            top_k=min(top_k, len(level_nodes)))

            # Map ranked indices back to node IDs
            top_current = [(current_ids[idx], score) for idx, score in ranked]
            selected.extend(top_current)

            # Descend to children of top-k
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
