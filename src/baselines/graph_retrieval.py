"""Lightweight Document Graph Retrieval.

Simplified GraphRAG approach that doesn't require a full knowledge graph
or LLM-based entity extraction. Instead, builds a document similarity
graph and uses community detection for thematic retrieval.

Key insight: standard retrieval finds individual relevant documents.
Graph retrieval finds *clusters* of related documents, which is better
for questions about themes, trends, or "what do researchers think about X?"

Reference: Microsoft GraphRAG (2024) - https://arxiv.org/abs/2404.16130
           LightRAG (2024) - lighter weight alternative
"""

import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DocumentGraph:
    """Build and query a document similarity graph.

    Nodes = documents (or chunks)
    Edges = semantic similarity above a threshold

    Communities = clusters of related documents, detected via
    a simple connected-components or label-propagation approach.
    """

    def __init__(self, similarity_threshold=0.5, max_edges_per_node=10):
        self.threshold = similarity_threshold
        self.max_edges = max_edges_per_node
        self.nodes = {}          # node_id -> {"text": ..., "embedding": ...}
        self.edges = {}          # (id_a, id_b) -> similarity_score
        self.communities = {}    # community_id -> [node_ids]
        self.node_to_community = {}  # node_id -> community_id
        self._built = False

    def add_node(self, node_id, text, embedding):
        self.nodes[node_id] = {
            "text": text,
            "embedding": np.asarray(embedding, dtype=np.float32),
        }

    def build_graph(self, block_size=1024):
        """Build edges based on cosine similarity between node embeddings.

        Memory-safe: similarities are computed in row BLOCKS, never a full
        N×N matrix (a 50K-node full matrix is 10GB; blocked it's ~40MB).
        """
        if len(self.nodes) < 2:
            self._built = True
            return

        node_ids = list(self.nodes.keys())
        embeddings = np.array([self.nodes[nid]["embedding"] for nid in node_ids], dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)

        n = len(node_ids)
        self.edges = {}
        k = min(self.max_edges + 1, n)  # +1 to skip self

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            sim_block = embeddings[start:end] @ embeddings.T  # (block, N)

            for i_local in range(end - start):
                i = start + i_local
                sim_scores = sim_block[i_local]
                top_indices = np.argsort(sim_scores)[::-1][:k]
                for j in top_indices:
                    if j == i:
                        continue
                    if sim_scores[j] >= self.threshold:
                        nid_a, nid_b = node_ids[i], node_ids[j]
                        edge = (min(nid_a, nid_b), max(nid_a, nid_b))
                        if edge not in self.edges:
                            self.edges[edge] = float(sim_scores[j])

        self._built = True
        logger.info(f"Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def detect_communities(self):
        """Detect communities using label propagation (simplified Louvain).

        Groups densely connected documents into communities.
        Each community represents a thematic cluster.
        """
        if not self._built:
            self.build_graph()

        # Build adjacency list
        adj = defaultdict(set)
        for (a, b), weight in self.edges.items():
            adj[a].add(b)
            adj[b].add(a)

        # Initialize: each node is its own community
        labels = {nid: i for i, nid in enumerate(self.nodes)}

        # Label propagation (iterative)
        for iteration in range(10):
            changed = False
            for node in self.nodes:
                if node not in adj:
                    continue

                # Count community labels of neighbors
                neighbor_labels = defaultdict(float)
                for neighbor in adj[node]:
                    edge = (min(node, neighbor), max(node, neighbor))
                    weight = self.edges.get(edge, 1.0)
                    neighbor_labels[labels[neighbor]] += weight

                if not neighbor_labels:
                    continue

                # Pick most common neighbor label
                new_label = max(neighbor_labels, key=neighbor_labels.get)
                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True

            if not changed:
                break

        # Group by community
        self.communities = defaultdict(list)
        self.node_to_community = {}
        for nid, label in labels.items():
            self.communities[label].append(nid)
            self.node_to_community[nid] = label

        # Remove empty communities
        self.communities = {k: v for k, v in self.communities.items() if v}

        logger.info(f"Detected {len(self.communities)} communities")
        return self.communities

    def get_community_summary(self, community_id):
        """Get summary info about a community."""
        if community_id not in self.communities:
            return None

        members = self.communities[community_id]
        texts = [self.nodes[nid]["text"] for nid in members if nid in self.nodes]

        # Use first sentence of each document as representative
        sentences = []
        for text in texts:
            first_sent = text.split(". ")[0] if ". " in text else text[:100]
            sentences.append(first_sent)

        return {
            "community_id": community_id,
            "size": len(members),
            "members": members,
            "representative_texts": sentences[:5],
        }


class GraphRetriever:
    """Retrieve using document graph communities.

    Instead of returning individual documents, returns thematic clusters
    of related documents. Better for questions like:
    - "What are the main approaches to X?"
    - "What do researchers say about Y?"
    - "Summarize the debate around Z"

    For factual lookups ("What is the capital of France?"),
    standard retrieval is better.
    """

    def __init__(self, base_retriever, chunk_size=200, chunk_overlap=50,
                 similarity_threshold=0.5):
        self.base_retriever = base_retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph = DocumentGraph(similarity_threshold=similarity_threshold)
        self._indexed = False

    def _chunk_text(self, text):
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(" ".join(tokens[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def build_graph(self, corpus, embeddings=None):
        """Build document graph from corpus.

        Args:
            corpus: dict of {doc_id: {"title": ..., "text": ...}}
            embeddings: optional pre-computed embeddings (numpy array)
        """
        from sentence_transformers import SentenceTransformer

        if embeddings is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = []
            ids = []
            for doc_id, doc in corpus.items():
                full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                chunks = self._chunk_text(full_text)
                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    ids.append(f"{doc_id}::chunk::{i}")

            embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

            for nid, text, emb in zip(ids, texts, embeddings):
                self.graph.add_node(nid, text, emb)
        else:
            # Use provided embeddings
            for i, (doc_id, doc) in enumerate(corpus.items()):
                full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                self.graph.add_node(doc_id, full_text, embeddings[i])

        self.graph.build_graph()
        self.graph.detect_communities()
        self._indexed = True

        return self.graph

    def retrieve(self, query, top_k=10, strategy="hybrid"):
        """Retrieve using graph-enhanced retrieval.

        Strategies:
        - "standard": regular retrieval from base retriever
        - "graph": return community-level results
        - "hybrid": combine standard + graph results
        """
        if strategy == "standard":
            return self.base_retriever.retrieve(query, top_k=top_k)

        # Standard retrieval first
        standard_results = self.base_retriever.retrieve(query, top_k=top_k)

        if strategy == "graph" and self._indexed:
            # Find which communities the top results belong to
            top_doc_ids = [did for did, _ in standard_results[:3]]
            community_ids = set()
            for did in top_doc_ids:
                cid = self.graph.node_to_community.get(did)
                if cid is not None:
                    community_ids.add(cid)

            # Expand results with community members
            expanded = {}
            for doc_id, score in standard_results:
                expanded[doc_id] = score

            for cid in community_ids:
                members = self.graph.communities.get(cid, [])
                for member in members:
                    member_doc = member.split("::")[0]
                    if member_doc not in expanded:
                        # Assign a decayed score based on community membership
                        expanded[member_doc] = 0.5  # Lower than directly retrieved

            sorted_results = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]

        # Hybrid: interleave standard and graph-expanded results
        if self._indexed:
            top_doc_ids = [did for did, _ in standard_results[:3]]
            community_ids = set()
            for did in top_doc_ids:
                cid = self.graph.node_to_community.get(did)
                if cid is not None:
                    community_ids.add(cid)

            expanded = {}
            for doc_id, score in standard_results:
                expanded[doc_id] = score

            for cid in community_ids:
                members = self.graph.communities.get(cid, [])
                for member in members:
                    member_doc = member.split("::")[0]
                    if member_doc not in expanded:
                        expanded[member_doc] = 0.3

            sorted_results = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]

        return standard_results
