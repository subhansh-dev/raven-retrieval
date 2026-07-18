"""Enhanced benchmark runner with all retrieval pipelines.

Includes:
1. Naive Dense RAG
2. Hybrid RAG (BM25 + Dense RRF)
3. Late Interaction (Flat, ColBERT)
4. HyDE (Hypothetical Document Embeddings)
5. SPLADE (Sparse Lexical and Expansion)
6. Contextual Retrieval (Anthropic-style)
7. Late Chunking (Jina-style)
8. RAPTOR + Late Interaction (collapsed & traversal)
9. Contextual Hybrid (Contextual BM25 + Contextual Dense + RRF)
10. SPLADE + Dense Hybrid

Usage:
    python run_enhanced_benchmark.py --dataset scifact --max-queries 100
    python run_enhanced_benchmark.py --dataset hotpotqa --max-queries 50 --skip-heavy
"""

import os
import sys
import json
import time
import gc
import argparse
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.eval.datasets import load_dataset, get_corpus_texts, subsample_queries
from src.eval.metrics import run_beir_evaluation, format_results_table, save_per_query_results
from src.eval.significance import run_all_pairwise_tests, format_significance_table
from src.eval.ablation import collect_per_query_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(name, retriever_or_fn, queries, top_k=10):
    """Run a retrieval pipeline and return results dict."""
    results = {}
    t0 = time.time()

    if callable(retriever_or_fn) and not hasattr(retriever_or_fn, 'retrieve'):
        # It's a function
        results = retriever_or_fn(queries, top_k)
    else:
        # It's a retriever object
        for qid, query in queries.items():
            retrieved = retriever_or_fn.retrieve(query, top_k=top_k)
            results[qid] = {doc_id: score for doc_id, score in retrieved}

    elapsed = time.time() - t0
    logger.info(f"  [{name}] {elapsed:.1f}s")
    return results, elapsed


def main():
    parser = argparse.ArgumentParser(description="Raven-Retrieval Enhanced Benchmark")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "hotpotqa", "fiqa"])
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="./experiments/runs")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip ColBERT and RAPTOR pipelines")
    parser.add_argument("--skip-colbert", action="store_true", help="Skip ColBERT-based pipelines only")
    parser.add_argument("--skip-raptor", action="store_true", help="Skip RAPTOR pipelines only")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        help="Specific pipelines to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"enhanced_{args.dataset}_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAVEN-RETRIEVAL ENHANCED BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {args.dataset}")

    # Load data
    corpus, queries, qrels = load_dataset(args.dataset, "./data")
    if args.max_queries:
        queries, qrels = subsample_queries(queries, qrels, args.max_queries, seed=args.seed)

    corpus_ids, corpus_texts = get_corpus_texts(corpus)
    logger.info(f"Corpus: {len(corpus_ids)} docs | Queries: {len(queries)}")

    all_results = {}
    timings = {}

    # ---- Pipeline 1: Naive Dense RAG ----
    if args.pipelines is None or "naive_dense" in args.pipelines:
        logger.info("\n[1] Naive Dense RAG")
        from src.baselines.dense import DenseRetriever
        retriever = DenseRetriever(model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50)
        retriever.index(corpus)
        results, elapsed = run_pipeline("naive_dense", retriever, queries, args.top_k)
        all_results["naive_dense"] = results
        timings["naive_dense"] = elapsed
        del retriever; gc.collect()

    # ---- Pipeline 2: Hybrid RAG ----
    if args.pipelines is None or "hybrid_rag" in args.pipelines:
        logger.info("\n[2] Hybrid RAG (BM25 + Dense RRF)")
        from src.baselines.hybrid import HybridRetriever
        retriever = HybridRetriever(model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60)
        retriever.index(corpus)
        results, elapsed = run_pipeline("hybrid_rag", retriever, queries, args.top_k)
        all_results["hybrid_rag"] = results
        timings["hybrid_rag"] = elapsed
        del retriever; gc.collect()

    # ---- Pipeline 3: HyDE ----
    if args.pipelines is None or "hyde" in args.pipelines:
        logger.info("\n[3] HyDE (Hypothetical Document Embeddings)")
        from src.baselines.hyde import HyDERetriever
        retriever = HyDERetriever(model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50)
        retriever.index(corpus)
        results, elapsed = run_pipeline("hyde", retriever, queries, args.top_k)
        all_results["hyde"] = results
        timings["hyde"] = elapsed
        del retriever; gc.collect()

    # ---- Pipeline 4: SPLADE ----
    if args.pipelines is None or "splade" in args.pipelines:
        logger.info("\n[4] SPLADE (Sparse Lexical and Expansion)")
        try:
            from src.baselines.splade import SPLADERetriever
            retriever = SPLADERetriever(model_name="bert-base-uncased", chunk_size=200, chunk_overlap=50)
            retriever.index(corpus)
            results, elapsed = run_pipeline("splade", retriever, queries, args.top_k)
            all_results["splade"] = results
            timings["splade"] = elapsed
            del retriever; gc.collect()
        except Exception as e:
            logger.warning(f"  SPLADE failed: {e}")

    # ---- Pipeline 5: SPLADE + Dense Hybrid ----
    if args.pipelines is None or "splade_hybrid" in args.pipelines:
        logger.info("\n[5] SPLADE + Dense Hybrid")
        try:
            from src.baselines.splade import HybridSPLADERetriever
            retriever = HybridSPLADERetriever(
                splade_model_name="bert-base-uncased",
                dense_model_name="all-MiniLM-L6-v2",
                chunk_size=200, chunk_overlap=50, rrf_k=60,
            )
            retriever.index(corpus)
            results, elapsed = run_pipeline("splade_hybrid", retriever, queries, args.top_k)
            all_results["splade_hybrid"] = results
            timings["splade_hybrid"] = elapsed
            del retriever; gc.collect()
        except Exception as e:
            logger.warning(f"  SPLADE Hybrid failed: {e}")

    # ---- Pipeline 6: BM25 + PRF ----
    if args.pipelines is None or "bm25_prf" in args.pipelines:
        logger.info("\n[6] BM25 + Pseudo-Relevance Feedback")
        from src.baselines.bm25_prf import BM25PRFRetriever
        retriever = BM25PRFRetriever(chunk_size=200, chunk_overlap=50, prf_k=5, expansion_terms=10)
        retriever.index(corpus)
        results, elapsed = run_pipeline("bm25_prf", retriever, queries, args.top_k)
        all_results["bm25_prf"] = results
        timings["bm25_prf"] = elapsed
        del retriever; gc.collect()

    # ---- Pipeline 7: Contextual Retrieval ----
    if args.pipelines is None or "contextual" in args.pipelines:
        logger.info("\n[7] Contextual Hybrid Retrieval")
        from src.baselines.contextual import ContextualHybridRetriever
        retriever = ContextualHybridRetriever(
            model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60
        )
        retriever.index(corpus)
        results, elapsed = run_pipeline("contextual_hybrid", retriever, queries, args.top_k)
        all_results["contextual_hybrid"] = results
        timings["contextual_hybrid"] = elapsed
        del retriever; gc.collect()

    # ---- Pipeline 8: Late Chunking ----
    if args.pipelines is None or "late_chunking" in args.pipelines:
        if not args.skip_heavy:
            logger.info("\n[7] Late Chunking (Jina-style)")
            try:
                from src.baselines.late_chunking import LateChunkingRetriever
                retriever = LateChunkingRetriever(
                    model_name="bert-base-uncased",
                    chunk_size_tokens=128,
                    max_doc_length=2048,
                )
                retriever.index(corpus)
                results, elapsed = run_pipeline("late_chunking", retriever, queries, args.top_k)
                all_results["late_chunking"] = results
                timings["late_chunking"] = elapsed
                del retriever; gc.collect()
            except Exception as e:
                logger.warning(f"  Late Chunking failed: {e}")

    # ---- Pipeline 9: ColBERT Late Interaction (Flat) ----
    if not args.skip_heavy and not args.skip_colbert:
        if args.pipelines is None or "late_interaction" in args.pipelines:
            logger.info("\n[8] ColBERT Late Interaction (Flat)")
            from src.encoder.colbert_encoder import ColbertEncoder
            from src.maxsim.brute_force import brute_force_rank

            encoder = ColbertEncoder(model_name="bert-base-uncased", projection_dim=128)
            encoder.eval()

            doc_token_embs = []
            t0 = time.time()
            with torch.no_grad():
                for text in corpus_texts:
                    emb = encoder.encode_document(text, max_length=256)
                    doc_token_embs.append(emb.cpu().numpy())

            results = {}
            with torch.no_grad():
                for qid, query in queries.items():
                    query_emb = encoder.encode_query(query).cpu().numpy()
                    ranked = brute_force_rank(query_emb, doc_token_embs, top_k=args.top_k)
                    results[qid] = {corpus_ids[idx]: score for idx, score in ranked if idx < len(corpus_ids)}

            elapsed = time.time() - t0
            all_results["late_interaction"] = results
            timings["late_interaction"] = elapsed
            del encoder, doc_token_embs; gc.collect()

    # ---- Pipeline 10: RAPTOR + Late Interaction ----
    if not args.skip_heavy and not args.skip_raptor:
        if args.pipelines is None or "raptor_late" in args.pipelines:
            logger.info("\n[9] RAPTOR + Late Interaction")
            try:
                from src.raptor.builder import RaptorBuilder
                from src.encoder.colbert_encoder import ColbertEncoder
                from src.combined.late_raptor import LateInteractionRaptor

                encoder = ColbertEncoder(model_name="bert-base-uncased", projection_dim=128)
                encoder.eval()

                combined = LateInteractionRaptor(
                    colbert_encoder=encoder,
                    embedding_model=None,  # Will use sentence-transformers internally
                    summarizer_model="facebook/bart-large-cnn",
                    chunk_size=100,
                )

                t0 = time.time()
                tree = combined.build(corpus)

                # Collapsed retrieval
                results_c = {}
                for qid, query in queries.items():
                    retrieved = combined.retrieve(query, strategy="collapsed", top_k=args.top_k)
                    doc_scores = {}
                    for node_id, score in retrieved:
                        doc_id = node_id.split("::")[0]
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = score
                    results_c[qid] = doc_scores

                all_results["raptor_late_collapsed"] = results_c
                timings["raptor_late_collapsed"] = time.time() - t0

                # Traversal retrieval
                t0 = time.time()
                results_t = {}
                for qid, query in queries.items():
                    retrieved = combined.retrieve(query, strategy="traversal", top_k=args.top_k)
                    doc_scores = {}
                    for node_id, score in retrieved:
                        doc_id = node_id.split("::")[0]
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = score
                    results_t[qid] = doc_scores

                all_results["raptor_late_traversal"] = results_t
                timings["raptor_late_traversal"] = time.time() - t0

                del combined, encoder; gc.collect()

            except Exception as e:
                logger.warning(f"  RAPTOR + Late Interaction failed: {e}")

    # ---- Evaluation ----
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    pipeline_names = list(all_results.keys())
    all_metrics = {}
    per_query_scores = {}

    for name in pipeline_names:
        metrics = run_beir_evaluation(qrels, all_results[name], k_values=[1, 3, 5, 10, 100])
        all_metrics[name] = metrics
        pq = collect_per_query_scores(qrels, all_results[name], k=args.top_k)
        per_query_scores[name] = pq

    # Print results table
    print("\n" + format_results_table([all_metrics[n] for n in pipeline_names], pipeline_names))

    # Statistical significance
    if len(pipeline_names) >= 2:
        logger.info("\n--- Statistical Significance ---")
        comparisons = run_all_pairwise_tests(per_query_scores, pipeline_names, n_resamples=10000)
        print(format_significance_table(comparisons))
    else:
        comparisons = []

    # ---- Save results ----
    logger.info(f"\nSaving to {run_dir}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    save_per_query_results(per_query_scores, os.path.join(run_dir, "per_query.json"))
    if comparisons:
        with open(os.path.join(run_dir, "significance.json"), "w") as f:
            json.dump(comparisons, f, indent=2, default=str)
    with open(os.path.join(run_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    # Generate dashboard
    try:
        from src.eval.visualize import generate_dashboard
        dashboard_path = os.path.join(run_dir, "dashboard.html")
        generate_dashboard(all_metrics, timings, comparisons, output_path=dashboard_path)
        logger.info(f"Dashboard: {dashboard_path}")
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name in pipeline_names:
        ndcg10 = all_metrics[name]["ndcg"].get("NDCG@10", 0.0)
        t = timings.get(name, 0)
        print(f"  {name:30s}  nDCG@10 = {ndcg10:.4f}  ({t:.1f}s)")

    print(f"\nResults: {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
