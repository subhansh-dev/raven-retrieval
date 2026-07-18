import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.encoder.colbert_encoder import ColbertEncoder
from src.maxsim.brute_force import brute_force_rank, maxsim_score
from src.maxsim.approximate import CentroidIndex, ApproximateMaxSim
from src.baselines.dense import DenseRetriever
from src.baselines.hybrid import HybridRetriever
from src.raptor.builder import RaptorBuilder
from src.combined.late_raptor import LateInteractionRaptor
from src.eval.datasets import load_dataset, get_corpus_texts, get_query_list, subsample_queries
from src.eval.metrics import run_beir_evaluation, measure_latency, format_results_table, save_per_query_results, save_run_metadata
from src.eval.significance import run_all_pairwise_tests, format_significance_table
from src.eval.ablation import AblationRunner


def get_hardware_info():
    info = {"cpu": "unknown", "ram": "unknown", "gpu": "none"}
    try:
        import platform
        info["cpu"] = platform.processor() or platform.machine()
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def get_package_versions():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "scifact", "fiqa"])
    parser.add_argument("--secondary-dataset", default=None, choices=["scifact", "fiqa"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments/runs")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--encoder-model", default="bert-base-uncased")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--summarizer-model", default="facebook/bart-large-cnn")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--skip-raptor", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"{args.dataset}_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Loading dataset: {args.dataset}")
    corpus, queries, qrels = load_dataset(args.dataset, args.data_dir)

    if args.max_queries:
        queries, qrels = subsample_queries(queries, qrels, args.max_queries, seed=args.seed)

    corpus_ids, corpus_texts = get_corpus_texts(corpus)
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    query_ids, query_texts = get_query_list(queries)

    print(f"Corpus size: {len(corpus_ids)}")
    print(f"Query count: {len(query_ids)}")

    all_pipeline_results = {}

    print("\n--- Baseline 1: Naive Dense RAG ---")
    dense_retriever = DenseRetriever(model_name=args.embedding_model, chunk_size=200, chunk_overlap=50)
    dense_retriever.index(corpus)
    dense_results = {}
    for qid, query in queries.items():
        retrieved = dense_retriever.retrieve(query, top_k=args.top_k)
        dense_results[qid] = {doc_id: score for doc_id, score in retrieved}
    all_pipeline_results["naive_rag"] = dense_results

    print("\n--- Baseline 2: Hybrid RAG (BM25 + Dense RRF) ---")
    hybrid_retriever = HybridRetriever(model_name=args.embedding_model, chunk_size=200, chunk_overlap=50, rrf_k=60)
    hybrid_retriever.index(corpus)
    hybrid_results = {}
    for qid, query in queries.items():
        retrieved = hybrid_retriever.retrieve(query, top_k=args.top_k)
        hybrid_results[qid] = {doc_id: score for doc_id, score in retrieved}
    all_pipeline_results["hybrid_rag"] = hybrid_results

    print("\n--- ColBERT Late Interaction (Flat) ---")
    encoder = ColbertEncoder(model_name=args.encoder_model, projection_dim=128)
    encoder.eval()
    with torch.no_grad():
        doc_token_embeddings = []
        for text in corpus_texts:
            emb = encoder.encode_document(text, max_length=256)
            doc_token_embeddings.append(emb.cpu().numpy())
    late_flat_results = {}
    with torch.no_grad():
        for qid, query in queries.items():
            query_emb = encoder.encode_query(query)
            query_emb_np = query_emb.cpu().numpy()
            ranked = brute_force_rank(query_emb_np, doc_token_embeddings, top_k=args.top_k)
            late_flat_results[qid] = {corpus_ids[idx]: score for idx, score in ranked if idx < len(corpus_ids)}
    all_pipeline_results["late_interaction_flat"] = late_flat_results

    if not args.skip_raptor:
        print("\n--- RAPTOR Tree (Single-Vector Retrieval) ---")
        raptor_builder = RaptorBuilder(
            embedding_model_name=args.embedding_model,
            summarizer_model=args.summarizer_model,
            chunk_size=args.chunk_size,
        )
        raptor_tree = raptor_builder.build(corpus)
        raptor_results = {}
        for qid, query in queries.items():
            query_emb = raptor_builder.embedding_model.encode([query])
            query_emb = np.array(query_emb).squeeze()
            all_nodes = raptor_tree.get_all_nodes_flat()
            scored = []
            for node in all_nodes:
                sim = np.dot(query_emb, node.pooled_embedding)
                scored.append((node.node_id, float(sim)))
            scored.sort(key=lambda x: x[1], reverse=True)
            doc_scores = {}
            for node_id, score in scored[:args.top_k]:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            raptor_results[qid] = doc_scores
        all_pipeline_results["raptor_single_vector"] = raptor_results

    if not args.skip_combined and not args.skip_raptor:
        print("\n--- RAPTOR + Late Interaction (Collapsed) ---")
        combined = LateInteractionRaptor(
            colbert_encoder=encoder,
            embedding_model=raptor_builder.embedding_model,
            summarizer_model=args.summarizer_model,
            chunk_size=args.chunk_size,
        )
        combined.tree = raptor_tree
        combined.encoder = encoder

        collapsed_results = {}
        traversal_results = {}
        for qid, query in queries.items():
            retrieved_c = combined.retrieve(query, strategy="collapsed", top_k=args.top_k)
            doc_scores_c = {}
            for node_id, score in retrieved_c:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores_c:
                    doc_scores_c[doc_id] = score
            collapsed_results[qid] = doc_scores_c

            retrieved_t = combined.retrieve(query, strategy="traversal", top_k=args.top_k)
            doc_scores_t = {}
            for node_id, score in retrieved_t:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores_t:
                    doc_scores_t[doc_id] = score
            traversal_results[qid] = doc_scores_t

        all_pipeline_results["raptor_late_collapsed"] = collapsed_results
        all_pipeline_results["raptor_late_traversal"] = traversal_results

    print("\n--- Running BEIR Evaluation ---")
    pipeline_names = list(all_pipeline_results.keys())
    all_metrics = {}
    per_query_scores = {}
    for name in pipeline_names:
        metrics = run_beir_evaluation(qrels, all_pipeline_results[name], k_values=[1, 3, 5, 10, 100])
        all_metrics[name] = metrics
        from src.eval.ablation import collect_per_query_scores
        pq = collect_per_query_scores(qrels, all_pipeline_results[name], k=args.top_k)
        per_query_scores[name] = pq

    print("\n" + format_results_table([all_metrics[n] for n in pipeline_names], pipeline_names))

    print("\n--- Statistical Significance Testing ---")
    comparisons = run_all_pairwise_tests(per_query_scores, pipeline_names, n_resamples=10000)
    print(format_significance_table(comparisons))

    print("\n--- Saving Results ---")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    save_per_query_results(per_query_scores, os.path.join(run_dir, "per_query.json"))
    with open(os.path.join(run_dir, "significance.json"), "w") as f:
        json.dump(comparisons, f, indent=2, default=str)
    save_run_metadata(
        run_id=run_id,
        seeds={"numpy": args.seed, "torch": args.seed},
        hardware_info=get_hardware_info(),
        package_versions=get_package_versions(),
        output_path=os.path.join(run_dir, "metadata.json"),
    )

    print(f"\nResults saved to {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
