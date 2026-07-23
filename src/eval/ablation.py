"""Ablation helpers.

The critical fix in this module: collect_per_query_scores now returns REAL
per-query nDCG values.

Previously it called BEIR's EvaluateRetrieval.evaluate(), which returns
corpus-AVERAGED floats (see beir/retrieval/evaluation.py — it sums
scores["ndcg_cut_k"] across queries and divides). Treating those floats as
a per-query dict silently produced a list of zeros for every query, which
made every bootstrap/t-test comparison meaningless (zero variance). The
"known BEIR artifact" mentioned in old docs was actually this bug.
"""

import json
import numpy as np

from .metrics import run_beir_evaluation, collect_per_query_ndcg


ABLATION_CONFIGS = [
    "naive_rag",
    "hybrid_rag",
    "late_interaction_flat",
    "raptor_single_vector",
    "raptor_late_collapsed",
    "raptor_late_traversal",
]


def collect_per_query_scores(qrels, results, k=10):
    """Per-query nDCG@k aligned to qrels key order (pure numpy, real values)."""
    return collect_per_query_ndcg(qrels, results, k=k)


class AblationRunner:

    def __init__(self, dataset_name, queries, qrels, corpus):
        self.dataset_name = dataset_name
        self.queries = queries
        self.qrels = qrels
        self.corpus = corpus
        self.all_results = {}

    def run_retriever(self, retriever, top_k=10):
        """Run any retriever and return BEIR-format results dict.

        This replaces the identical run_naive_rag and run_hybrid_rag methods.
        They were doing the exact same thing — the name was misleading
        (the pipeline type depends on the retriever passed in, not the method).
        """
        results = {}
        for qid, query in self.queries.items():
            retrieved = retriever.retrieve(query, top_k=top_k)
            results[qid] = {doc_id: score for doc_id, score in retrieved}
        return results

    # Aliases for backward compatibility
    def run_naive_rag(self, retriever, top_k=10):
        """Alias for run_retriever (backward compatibility)."""
        return self.run_retriever(retriever, top_k=top_k)

    def run_hybrid_rag(self, retriever, top_k=10):
        """Alias for run_retriever (backward compatibility)."""
        return self.run_retriever(retriever, top_k=top_k)

    def run_late_interaction_flat(self, encoder, corpus_embeddings_list, corpus_ids, top_k=10):
        from ..maxsim.brute_force import brute_force_rank
        results = {}
        for qid, query in self.queries.items():
            query_emb = encoder.encode_query(query)
            query_emb_np = query_emb.detach().cpu().numpy()
            if query_emb_np.ndim == 3:
                query_emb_np = query_emb_np.squeeze(0)
            ranked = brute_force_rank(query_emb_np, corpus_embeddings_list, top_k=top_k)
            results[qid] = {corpus_ids[idx]: score for idx, score in ranked}
        return results

    def run_raptor_single_vector(self, tree, query_embeddings_model, top_k=10):
        results = {}
        for qid, query in self.queries.items():
            query_emb = query_embeddings_model.encode([query])
            query_emb = np.array(query_emb).squeeze()
            all_nodes = tree.get_all_nodes_flat()
            scored = []
            for node in all_nodes:
                sim = np.dot(query_emb, node.pooled_embedding)
                scored.append((node.node_id, float(sim)))
            scored.sort(key=lambda x: x[1], reverse=True)
            doc_scores = {}
            for node_id, score in scored[:top_k]:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            results[qid] = doc_scores
        return results

    def run_raptor_late_collapsed(self, combined_system, top_k=10):
        results = {}
        for qid, query in self.queries.items():
            retrieved = combined_system.retrieve(query, strategy="collapsed", top_k=top_k)
            doc_scores = {}
            for node_id, score in retrieved:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            results[qid] = doc_scores
        return results

    def run_raptor_late_traversal(self, combined_system, top_k=10):
        results = {}
        for qid, query in self.queries.items():
            retrieved = combined_system.retrieve(query, strategy="traversal", top_k=top_k)
            doc_scores = {}
            for node_id, score in retrieved:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            results[qid] = doc_scores
        return results

    def run_all(self, pipelines, top_k=10):
        all_metrics = {}
        per_query = {}
        for config_name, results in pipelines.items():
            metrics = run_beir_evaluation(self.qrels, results, k_values=[1, 3, 5, 10, 100])
            all_metrics[config_name] = metrics
            pq = collect_per_query_scores(self.qrels, results, k=top_k)
            per_query[config_name] = pq
        return all_metrics, per_query

    def save_results(self, all_metrics, per_query, output_dir):
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        with open(f"{output_dir}/per_query.json", "w") as f:
            json.dump(per_query, f, indent=2)
