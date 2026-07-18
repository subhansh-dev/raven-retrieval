import os, sys, json, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baselines.dense import DenseRetriever
from src.baselines.hybrid import HybridRetriever
from src.encoder.colbert_encoder import ColbertEncoder
from src.maxsim.brute_force import brute_force_rank
from src.raptor.builder import RaptorBuilder
from src.raptor.tree import RaptorTree
from src.raptor.summarizer import LLMSummarizer
from src.eval.datasets import load_dataset, get_corpus_texts, subsample_queries
from src.eval.metrics import run_beir_evaluation, format_results_table, save_per_query_results, save_run_metadata
from src.eval.significance import run_all_pairwise_tests, format_significance_table
from src.eval.ablation import collect_per_query_scores

import subprocess


def get_hardware_info():
    import platform
    info = {"cpu": platform.processor() or platform.machine(), "ram": "unknown", "gpu": "none"}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    info["ram"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def get_package_versions():
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def chunk_corpus_for_eval(corpus, queries, qrels, top_k=10):
    corpus_ids, corpus_texts = get_corpus_texts(corpus)
    return corpus_ids, corpus_texts


def run_dense(corpus, queries, top_k=10):
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50)
    retriever.index(corpus)
    results = {}
    for qid, query in queries.items():
        retrieved = retriever.retrieve(query, top_k=top_k)
        results[qid] = {did: s for did, s in retrieved}
    del retriever
    gc.collect()
    return results


def run_hybrid(corpus, queries, top_k=10):
    retriever = HybridRetriever(model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60)
    retriever.index(corpus)
    results = {}
    for qid, query in queries.items():
        retrieved = retriever.retrieve(query, top_k=top_k)
        results[qid] = {did: s for did, s in retrieved}
    del retriever
    gc.collect()
    return results


def run_late_flat(corpus, queries, encoder, corpus_ids, top_k=10):
    corpus_texts_full = []
    for cid in corpus_ids:
        doc = corpus[cid] if isinstance(corpus, dict) else next(d for d in corpus if d["_id"] == cid)
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        corpus_texts_full.append(full_text)

    print(f"    Encoding {len(corpus_texts_full)} docs with ColBERT encoder...")
    encoder.eval()
    doc_token_embs = []
    with torch.no_grad():
        for i, text in enumerate(corpus_texts_full):
            emb = encoder.encode_document(text, max_length=256)
            doc_token_embs.append(emb.cpu().numpy())
            if (i + 1) % 200 == 0:
                print(f"    Encoded {i+1}/{len(corpus_texts_full)}")

    print(f"    Running MaxSim queries...")
    results = {}
    with torch.no_grad():
        for qid, query in queries.items():
            query_emb = encoder.encode_query(query).cpu().numpy()
            ranked = brute_force_rank(query_emb, doc_token_embs, top_k=top_k)
            results[qid] = {corpus_ids[idx]: score for idx, score in ranked if idx < len(corpus_ids)}
    del doc_token_embs
    gc.collect()
    return results


def run_raptor_single_vector(corpus, queries, embedding_model, chunk_size=100, top_k=10):
    print("    Building RAPTOR tree (single-vector)...")
    builder = RaptorBuilder(
        embedding_model_name="all-MiniLM-L6-v2",
        summarizer_model="facebook/bart-large-cnn",
        chunk_size=chunk_size,
    )
    tree = builder.build(corpus)
    print(f"    Tree built: {len(tree.nodes)} nodes, max level {tree.get_max_level()}")

    print("    Running single-vector retrieval...")
    results = {}
    for qid, query in queries.items():
        query_emb = builder.embedding_model.encode([query])
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
    del tree, builder
    gc.collect()
    return results


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    TOP_K = 10
    MAX_QUERIES_SCI = 100
    MAX_QUERIES_HOTPOT = 50

    run_id = f"full_ablation_{int(time.time())}"
    run_dir = f"./experiments/runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("RAVEN-RETRIEVAL FULL BENCHMARK")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Hardware: {get_hardware_info()}")
    print()

    datasets_to_run = [
        ("scifact", MAX_QUERIES_SCI, True),
        ("hotpotqa", MAX_QUERIES_HOTPOT, False),
    ]

    all_dataset_results = {}

    for dataset_name, max_queries, run_late_and_raptor in datasets_to_run:
        print("=" * 60)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 60)

        corpus, queries, qrels = load_dataset(dataset_name, "./data")
        if max_queries:
            queries, qrels = subsample_queries(queries, qrels, max_queries, seed=42)

        corpus_ids, corpus_texts = get_corpus_texts(corpus)
        print(f"Corpus: {len(corpus_ids)} docs | Queries: {len(queries)}")

        pipeline_results = {}

        print("\n[1/6] Naive Dense RAG...")
        t0 = time.time()
        pipeline_results["naive_rag"] = run_dense(corpus, queries, top_k=TOP_K)
        print(f"  Done in {time.time()-t0:.1f}s")

        print("\n[2/6] Hybrid RAG (BM25 + Dense RRF)...")
        t0 = time.time()
        pipeline_results["hybrid_rag"] = run_hybrid(corpus, queries, top_k=TOP_K)
        print(f"  Done in {time.time()-t0:.1f}s")

        if run_late_and_raptor:
            print("\n[3/6] Late Interaction (Flat, ColBERT)...")
            t0 = time.time()
            encoder = ColbertEncoder(model_name="bert-base-uncased", projection_dim=128)
            pipeline_results["late_interaction_flat"] = run_late_flat(
                corpus, queries, encoder, corpus_ids, top_k=TOP_K
            )
            del encoder
            gc.collect()
            print(f"  Done in {time.time()-t0:.1f}s")

            print("\n[4/6] RAPTOR Tree (Single-Vector Retrieval)...")
            t0 = time.time()
            try:
                pipeline_results["raptor_single_vector"] = run_raptor_single_vector(
                    corpus, queries, None, chunk_size=100, top_k=TOP_K
                )
                print(f"  Done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  SKIPPED: {e}")

        print("\n--- BEIR Evaluation ---")
        pipeline_names = list(pipeline_results.keys())
        all_metrics = {}
        per_query_scores = {}
        for name in pipeline_names:
            metrics = run_beir_evaluation(qrels, pipeline_results[name], k_values=[1, 3, 5, 10, 100])
            all_metrics[name] = metrics
            pq = collect_per_query_scores(qrels, pipeline_results[name], k=TOP_K)
            per_query_scores[name] = pq

        print(format_results_table([all_metrics[n] for n in pipeline_names], pipeline_names))

        if len(pipeline_names) >= 2:
            print("\n--- Statistical Significance ---")
            comparisons = run_all_pairwise_tests(per_query_scores, pipeline_names, n_resamples=10000)
            print(format_significance_table(comparisons))
        else:
            comparisons = []

        ds_dir = f"{run_dir}/{dataset_name}"
        os.makedirs(ds_dir, exist_ok=True)
        with open(f"{ds_dir}/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        save_per_query_results(per_query_scores, f"{ds_dir}/per_query.json")
        if comparisons:
            with open(f"{ds_dir}/significance.json", "w") as f:
                json.dump(comparisons, f, indent=2, default=str)

        all_dataset_results[dataset_name] = {
            "metrics": all_metrics,
            "pipeline_names": pipeline_names,
        }

        del pipeline_results, all_metrics, per_query_scores
        gc.collect()
        print()

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for ds_name, ds_data in all_dataset_results.items():
        print(f"\n{ds_name.upper()}:")
        for pname in ds_data["pipeline_names"]:
            ndcg10 = ds_data["metrics"][pname]["ndcg"].get("NDCG@10", 0.0)
            print(f"  {pname:30s} nDCG@10 = {ndcg10:.4f}")

    save_run_metadata(
        run_id=run_id,
        seeds={"numpy": 42, "torch": 42},
        hardware_info=get_hardware_info(),
        package_versions=get_package_versions(),
        output_path=f"{run_dir}/metadata.json",
    )
    print(f"\nAll results saved to {run_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
