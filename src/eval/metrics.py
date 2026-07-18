import json
import time
import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval


def run_beir_evaluation(qrels, results, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5, 10, 100]
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    return {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }


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
