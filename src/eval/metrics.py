"""Evaluation metrics for raven-retrieval.

Two evaluation paths:

1. BEIR/pytrec_eval (default when installed) — the standard IR evaluation,
   used for headline numbers in reports.
2. Internal pure-numpy implementation — same trec_eval conventions
   (linear gain, log2(rank+1) discount), used as:
   - fallback when pytrec_eval is unavailable
   - PER-QUERY scores for significance testing (BEIR's evaluate() only
     returns corpus-averaged floats — using it for per-query scores was
     silently returning all zeros; see BENCHMARK_RESULTS.md history)
"""

import json
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ── Internal trec_eval-compatible metrics (pure numpy) ───────────────

def _ranked_doc_ids(score_dict):
    """Sort {doc_id: score} by score desc, ties broken by doc_id for determinism."""
    return [did for did, _ in sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))]


def ndcg_per_query(qrels, results, k):
    """Per-query nDCG@k (trec_eval convention: linear gain, log2(rank+1) discount).

    Returns {qid: ndcg}. Queries with no relevant docs are skipped
    (matching trec_eval behavior of excluding them from averages).
    """
    scores = {}
    for qid, rels in qrels.items():
        relevant = {d: r for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = _ranked_doc_ids(results.get(qid, {}))[:k]
        dcg = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            rel = relevant.get(doc_id, 0)
            if rel > 0:
                dcg += rel / np.log2(rank + 1)
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal, start=1))
        scores[qid] = dcg / idcg if idcg > 0 else 0.0
    return scores


def recall_per_query(qrels, results, k):
    """Per-query Recall@k: |retrieved ∩ relevant| / |relevant|."""
    scores = {}
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = set(_ranked_doc_ids(results.get(qid, {}))[:k])
        scores[qid] = len(ranked & relevant) / len(relevant)
    return scores


def precision_per_query(qrels, results, k):
    """Per-query Precision@k: |retrieved ∩ relevant| / k."""
    scores = {}
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = set(_ranked_doc_ids(results.get(qid, {}))[:k])
        scores[qid] = len(ranked & relevant) / k
    return scores


def map_per_query(qrels, results, k):
    """Per-query Average Precision@k (trec_eval map_cut: divide by total relevant)."""
    scores = {}
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = _ranked_doc_ids(results.get(qid, {}))[:k]
        hits = 0
        prec_sum = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                hits += 1
                prec_sum += hits / rank
        scores[qid] = prec_sum / len(relevant)
    return scores


def evaluate_internal(qrels, results, k_values=(1, 3, 5, 10, 100)):
    """Corpus-averaged metrics, same shape as BEIR's evaluate() return."""
    ndcg, _map, recall, precision = {}, {}, {}, {}
    for k in k_values:
        n = ndcg_per_query(qrels, results, k)
        m = map_per_query(qrels, results, k)
        r = recall_per_query(qrels, results, k)
        p = precision_per_query(qrels, results, k)
        ndcg[f"NDCG@{k}"] = round(float(np.mean(list(n.values()))), 5) if n else 0.0
        _map[f"MAP@{k}"] = round(float(np.mean(list(m.values()))), 5) if m else 0.0
        recall[f"Recall@{k}"] = round(float(np.mean(list(r.values()))), 5) if r else 0.0
        precision[f"P@{k}"] = round(float(np.mean(list(p.values()))), 5) if p else 0.0
    return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}


# ── BEIR path (preferred) with automatic fallback ────────────────────

def run_beir_evaluation(qrels, results, k_values=None):
    """Evaluate with BEIR's EvaluateRetrieval; fall back to internal metrics.

    Returns {"ndcg": {...}, "map": {...}, "recall": {...}, "precision": {...}}.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 100]
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        logger.warning("beir/pytrec_eval not installed — using internal trec_eval-compatible metrics")
        return evaluate_internal(qrels, results, k_values)
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    return {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }


def collect_per_query_ndcg(qrels, results, k=10):
    """Per-query nDCG@k as an aligned list over qrels keys.

    This is the function significance testing should use. Pure numpy —
    no pytrec_eval dependency, and (unlike the old implementation that
    called BEIR's corpus-averaged evaluate()) it actually returns
    per-query values.
    """
    per_query = ndcg_per_query(qrels, results, k)
    return [per_query.get(qid, 0.0) for qid in qrels]


# ── Formatting & utilities ───────────────────────────────────────────

def format_results_table(all_metrics, pipeline_names):
    header = "| Pipeline | " + " | ".join([f"nDCG@{k}" for k in [1, 3, 5, 10, 100]]) + " |"
    separator = "|" + "|".join(["---"] * 6) + "|"
    rows = [header, separator]
    for name, metrics in zip(pipeline_names, all_metrics):
        ndcg = metrics["ndcg"]
        row = f"| {name} | " + " | ".join([f"{ndcg.get(f'NDCG@{k}', 0.0):.4f}" for k in [1, 3, 5, 10, 100]]) + " |"
        rows.append(row)
    return "\n".join(rows)


def compute_em_f1(predicted_answers, gold_answers):
    em_scores = []
    f1_scores = []
    for pred, gold in zip(predicted_answers, gold_answers):
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            em_scores.append(1.0 if not pred_tokens else 0.0)
            f1_scores.append(1.0 if not pred_tokens else 0.0)
            continue
        common = pred_tokens & gold_tokens
        em = 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
        em_scores.append(em)
        if not pred_tokens:
            f1_scores.append(0.0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
    }


def measure_latency(retriever, queries, method_name="default", top_k=10):
    latencies = []
    for query in queries:
        start = time.perf_counter()
        retriever.retrieve(query, top_k=top_k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
    return {
        "method": method_name,
        "mean_latency_s": float(np.mean(latencies)),
        "median_latency_s": float(np.median(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "p99_latency_s": float(np.percentile(latencies, 99)),
        "total_queries": len(queries),
    }


def save_per_query_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def save_run_metadata(run_id, seeds, hardware_info, package_versions, output_path):
    metadata = {
        "run_id": run_id,
        "seeds": seeds,
        "hardware": hardware_info,
        "package_versions": package_versions,
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
