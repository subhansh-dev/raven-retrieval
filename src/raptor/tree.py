import numpy as np


class TreeNode:

    def __init__(self, node_id, text, embeddings, level, children=None, parent=None, doc_id=None):
        self.node_id = node_id
        self.text = text
        self.embeddings = embeddings
        self.pooled_embedding = np.mean(embeddings, axis=0) if embeddings is not None else None
        self.level = level
        self.children = children or []
        self.parent = parent
        self.doc_id = doc_id

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class RaptorTree:

    def __init__(self):
        self.nodes = {}
        self.levels = {}
        self.root_ids = []

    def add_node(self, node):
        self.nodes[node.node_id] = node
        if node.level not in self.levels:
            self.levels[node.level] = []
        self.levels[node.level].append(node.node_id)

    def get_level(self, level):
        return self.levels.get(level, [])

    def get_max_level(self):
        if not self.levels:
            return 0
        return max(self.levels.keys())

    def get_all_nodes_flat(self):
        return list(self.nodes.values())

    def get_leaf_nodes(self):
        leaves = []
        for node in self.nodes.values():
            if not node.children:
                leaves.append(node)
        return leaves

    def get_node_path(self, node_id):
        path = []
        current = self.nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def traverse_top_down(self, query_embeddings, scorer, top_k=10):
        if not self.root_ids:
            return []
        selected = []
        current_ids = self.root_ids
        while current_ids:
            scores = []
            for nid in current_ids:
                node = self.nodes[nid]
                score = scorer(query_embeddings, node.embeddings)
                scores.append((nid, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_current = scores[:top_k]
            selected.extend(top_current)
            next_ids = []
            for nid, _ in top_current:
                node = self.nodes[nid]
                for child in node.children:
                    next_ids.append(child.node_id)
            current_ids = next_ids
        return selected

    def retrieve_collapsed(self, query_embeddings, scorer, top_k=10):
        all_nodes = self.get_all_nodes_flat()
        scores = []
        for node in all_nodes:
            score = scorer(query_embeddings, node.embeddings)
            scores.append((node.node_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
