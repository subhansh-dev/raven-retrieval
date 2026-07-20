"""Raven-Retrieval — Unified Benchmark Runner.

Single entry point for all benchmarks (replaces the old run_benchmark.py /
run_full_benchmark.py which had diverged).

Design:
1. SHARED MODELS: one SBERT, one ColBERT, one BART across all pipelines
   (avoids OOM on 4-8GB machines).
2. SEPARATE TIMING: index and query time tracked independently.
3. MEMORY SAFETY: gc between pipelines, batched encoding, optional
   --max-docs subsampling (judged docs always preserved).
4. ERROR CAPTURE: each pipeline runs in try/except; failures go to
   errors.json and the summary.

Usage:
    python run_enhanced_benchmark.py --dataset scifact --max-queries 100
    python run_enhanced_benchmark.py --dataset hotpotqa --max-queries 50 --max-docs 2000
    python run_enhanced_benchmark.py --pipelines naive_dense hybrid_rag --skip-heavy
    python run_enhanced_benchmark.py --colbert-checkpoint checkpoints/final_model.pt
"""

import os
import sys
import json
import time
import gc
import argparse
import logging
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import PipelineTimer, iter_corpus, full_doc_text
from src.eval.datasets import load_dataset, get_corpus_texts, subsample_queries, subsample_corpus
from src.eval.metrics import (
    run_beir_evaluation, format_results_table, save_per_query_results,
    save_run_metadata, collect_per_query_ndcg,
)
from src.eval.significance import run_all_pairwise_tests, format_significance_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Pipeline registry ────────────────────────────────────────────────
# name -> {"heavy": bool, "colbert": bool, "raptor": bool, "runner": callable}
# runner(ctx) fills ctx["results"][name], ctx["timings"][name]
# ctx keys: corpus, queries, qrels, args, sbert, colbert, summarizer

def _run_standard(ctx, name, retriever):
    """Standard path: index (timed) → retrieve all queries (timed)."""
    timer = PipelineTimer(name)
    with timer.index_phase():
        retriever.index(ctx["corpus"])
    results = {}
    with timer.query_phase():
        for qid, query in ctx["queries"].items():
            retrieved = retriever.retrieve(query, top_k=ctx["args"].top_k)
            results[qid] = {doc_id: score for doc_id, score in retrieved}
    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")
    return retriever


def run_naive_dense(ctx):
    from src.baselines.dense import DenseRetriever
    r = DenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())
    _run_standard(ctx, "naive_dense", r)


def run_hybrid(ctx):
    from src.baselines.hybrid import HybridRetriever
    r = HybridRetriever(chunk_size=200, chunk_overlap=50, rrf_k=60, model=ctx["sbert"]())
    _run_standard(ctx, "hybrid_rag", r)


def run_hyde(ctx):
    from src.baselines.hyde import HyDERetriever
    r = HyDERetriever(chunk_size=200, chunk_overlap=50, use_llm=ctx["args"].hyde_llm,
                      model=ctx["sbert"]())
    _run_standard(ctx, "hyde", r)


def run_splade(ctx):
    from src.baselines.splade import SPLADERetriever
    r = SPLADERetriever(model_name="bert-base-uncased", chunk_size=200, chunk_overlap=50)
    _run_standard(ctx, "splade", r)


def run_splade_hybrid(ctx):
    from src.baselines.splade import HybridSPLADERetriever
    r = HybridSPLADERetriever(splade_model_name="bert-base-uncased",
                              chunk_size=200, chunk_overlap=50, rrf_k=60,
                              dense_model=ctx["sbert"]())
    _run_standard(ctx, "splade_hybrid", r)


def run_bm25_prf(ctx):
    from src.baselines.bm25_prf import BM25PRFRetriever
    r = BM25PRFRetriever(chunk_size=200, chunk_overlap=50, prf_k=5, expansion_terms=10)
    _run_standard(ctx, "bm25_prf", r)


def run_contextual_dense(ctx):
    from src.baselines.contextual import ContextualDenseRetriever
    r = ContextualDenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())
    _run_standard(ctx, "contextual_dense", r)


def run_contextual_bm25(ctx):
    from src.baselines.contextual import ContextualBM25Retriever
    r = ContextualBM25Retriever(chunk_size=200, chunk_overlap=50)
    _run_standard(ctx, "contextual_bm25", r)


def run_contextual_hybrid(ctx):
    from src.baselines.contextual import ContextualHybridRetriever
    r = ContextualHybridRetriever(chunk_size=200, chunk_overlap=50, rrf_k=60, model=ctx["sbert"]())
    _run_standard(ctx, "contextual_hybrid", r)


def run_late_chunking(ctx):
    from src.baselines.late_chunking import LateChunkingRetriever
    r = LateChunkingRetriever(model_name="bert-base-uncased", chunk_size_tokens=128,
                              max_doc_length=2048)
    _run_standard(ctx, "late_chunking", r)


def _encode_corpus_colbert(ctx, max_length=256):
    """Encode corpus with the shared ColBERT encoder (batched, mask-trimmed)."""
    corpus_ids, corpus_texts = get_corpus_texts(ctx["corpus"])
    encoder = ctx["colbert"]()
    doc_embs = encoder.encode_documents(
        corpus_texts, max_length=max_length,
        batch_size=ctx["args"].encode_batch_size, show_progress=True,
    )
    return corpus_ids, doc_embs


def run_late_interaction(ctx):
    from src.maxsim.brute_force import pack_doc_embeddings, brute_force_rank_fast
    import torch

    name = "late_interaction"
    timer = PipelineTimer(name)
    encoder = ctx["colbert"]()

    with timer.index_phase():
        corpus_ids, doc_embs = _encode_corpus_colbert(ctx)
        flat, lengths = pack_doc_embeddings(doc_embs)

    results = {}
    with timer.query_phase():
        with torch.no_grad():
            for qid, query in ctx["queries"].items():
                q_emb = encoder.encode_query(query).detach().cpu().numpy()
                if q_emb.ndim == 3:
                    q_emb = q_emb.squeeze(0)
                ranked = brute_force_rank_fast(q_emb, flat, lengths, top_k=ctx["args"].top_k)
                results[qid] = {corpus_ids[idx]: score for idx, score in ranked}

    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


def run_late_interaction_approx(ctx):
    from src.maxsim.brute_force import pack_doc_embeddings
    from src.maxsim.approximate import CentroidIndex, ApproximateMaxSim
    import torch

    name = "late_interaction_approx"
    timer = PipelineTimer(name)
    encoder = ctx["colbert"]()

    with timer.index_phase():
        corpus_ids, doc_embs = _encode_corpus_colbert(ctx)
        flat, lengths = pack_doc_embeddings(doc_embs)
        index = CentroidIndex(num_centroids=min(256, max(4, flat.shape[0] // 10)))
        index.build(flat)
        approx = ApproximateMaxSim(index, prune_ratio=0.2)
        approx.index_documents(doc_embs)

    results = {}
    with timer.query_phase():
        with torch.no_grad():
            for qid, query in ctx["queries"].items():
                q_emb = encoder.encode_query(query).detach().cpu().numpy()
                if q_emb.ndim == 3:
                    q_emb = q_emb.squeeze(0)
                ranked = approx.retrieve(q_emb, top_k=ctx["args"].top_k)
                results[qid] = {corpus_ids[idx]: score for idx, score in ranked}

    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


def _build_raptor_tree(ctx):
    """Build (and cache) the single-vector RAPTOR tree."""
    if "raptor_tree" in ctx:
        return ctx["raptor_tree"], ctx["raptor_builder"]
    from src.raptor.builder import RaptorBuilder
    builder = RaptorBuilder(
        chunk_size=100,
        embedding_model=ctx["sbert"](),
        summarizer=ctx["summarizer"](),
    )
    tree = builder.build(ctx["corpus"])
    logger.info(f"  RAPTOR tree: {len(tree.nodes)} nodes, max level {tree.get_max_level()}")
    ctx["raptor_tree"] = tree
    ctx["raptor_builder"] = builder
    return tree, builder


def run_raptor_single_vector(ctx):
    name = "raptor_single_vector"
    timer = PipelineTimer(name)
    with timer.index_phase():
        tree, builder = _build_raptor_tree(ctx)

    results = {}
    with timer.query_phase():
        for qid, query in ctx["queries"].items():
            q_emb = np.array(ctx["sbert"]().encode([query])).squeeze()
            scored = []
            for node in tree.get_all_nodes_flat():
                scored.append((node.node_id, float(np.dot(q_emb, node.pooled_embedding))))
            scored.sort(key=lambda x: x[1], reverse=True)
            doc_scores = {}
            for node_id, score in scored[:ctx["args"].top_k]:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            results[qid] = doc_scores

    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


def _build_late_raptor(ctx):
    """Build (and cache) the LateInteractionRaptor tree."""
    if "late_raptor" in ctx:
        return ctx["late_raptor"]
    from src.combined.late_raptor import LateInteractionRaptor
    combined = LateInteractionRaptor(
        colbert_encoder=ctx["colbert"](),
        embedding_model=ctx["sbert"](),
        summarizer=ctx["summarizer"](),
        chunk_size=100,
        encode_batch_size=ctx["args"].encode_batch_size,
    )
    combined.build(ctx["corpus"])
    ctx["late_raptor"] = combined
    return combined


def run_raptor_late(ctx, strategy, name):
    timer = PipelineTimer(name)
    with timer.index_phase():
        combined = _build_late_raptor(ctx)

    results = {}
    with timer.query_phase():
        for qid, query in ctx["queries"].items():
            retrieved = combined.retrieve(query, strategy=strategy, top_k=ctx["args"].top_k)
            doc_scores = {}
            for node_id, score in retrieved:
                doc_id = node_id.split("::")[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
            results[qid] = doc_scores

    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


def run_two_stage(ctx):
    from src.baselines.dense import DenseRetriever
    from src.baselines.reranker import CrossEncoderReranker, TwoStageRetriever
    dense = DenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())
    two_stage = TwoStageRetriever(dense, CrossEncoderReranker(),
                                  first_stage_k=100, final_k=ctx["args"].top_k)
    two_stage.set_corpus(ctx["corpus"] if isinstance(ctx["corpus"], dict) else {
        d["_id"]: d for d in ctx["corpus"]
    })
    _run_standard(ctx, "two_stage_dense", two_stage)


def run_agentic_multihop(ctx):
    from src.baselines.dense import DenseRetriever
    from src.baselines.agentic import MultiHopRetriever
    dense = DenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())
    multi = MultiHopRetriever(dense, top_k=ctx["args"].top_k)

    # MultiHop wraps an indexed dense retriever; index once, delegate retrieval
    timer = PipelineTimer("agentic_multihop")
    with timer.index_phase():
        dense.index(ctx["corpus"])
    results = {}
    with timer.query_phase():
        for qid, query in ctx["queries"].items():
            retrieved = multi.retrieve(query, top_k=ctx["args"].top_k)
            results[qid] = {doc_id: score for doc_id, score in retrieved}
    ctx["results"]["agentic_multihop"] = results
    ctx["timings"]["agentic_multihop"] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [agentic_multihop] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


def run_reflection(ctx):
    from src.baselines.dense import DenseRetriever
    from src.baselines.agentic import ReflectionRetriever
    dense = DenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())

    # text_lookup over full document text (needed for real sufficiency evaluation)
    doc_text_map = {doc_id: full_doc_text(doc) for doc_id, doc in iter_corpus(ctx["corpus"])}
    reflection = ReflectionRetriever(dense, text_lookup=doc_text_map.get,
                                     max_iterations=2, top_k=ctx["args"].top_k)
    _run_standard(ctx, "reflection", reflection)


def run_graph(ctx):
    from src.baselines.dense import DenseRetriever
    from src.baselines.graph_retrieval import GraphRetriever
    dense = DenseRetriever(chunk_size=200, chunk_overlap=50, model=ctx["sbert"]())
    graph = GraphRetriever(dense, chunk_size=200, chunk_overlap=50)

    name = "graph"
    timer = PipelineTimer(name)
    with timer.index_phase():
        dense.index(ctx["corpus"])
        corpus_dict = ctx["corpus"] if isinstance(ctx["corpus"], dict) else {
            d["_id"]: d for d in ctx["corpus"]
        }
        graph.build_graph(corpus_dict)

    results = {}
    with timer.query_phase():
        for qid, query in ctx["queries"].items():
            retrieved = graph.retrieve(query, top_k=ctx["args"].top_k, strategy="hybrid")
            results[qid] = {doc_id: score for doc_id, score in retrieved}
    ctx["results"][name] = results
    ctx["timings"][name] = timer.record(n_queries=len(ctx["queries"]))
    logger.info(f"  [{name}] index={timer.index_s:.1f}s query={timer.query_s:.1f}s")


PIPELINE_REGISTRY = {
    "naive_dense":              {"heavy": False, "colbert": False, "raptor": False, "fn": run_naive_dense},
    "hybrid_rag":               {"heavy": False, "colbert": False, "raptor": False, "fn": run_hybrid},
    "hyde":                     {"heavy": False, "colbert": False, "raptor": False, "fn": run_hyde},
    "splade":                   {"heavy": True,  "colbert": False, "raptor": False, "fn": run_splade},
    "splade_hybrid":            {"heavy": True,  "colbert": False, "raptor": False, "fn": run_splade_hybrid},
    "bm25_prf":                 {"heavy": False, "colbert": False, "raptor": False, "fn": run_bm25_prf},
    "contextual_dense":         {"heavy": False, "colbert": False, "raptor": False, "fn": run_contextual_dense},
    "contextual_bm25":          {"heavy": False, "colbert": False, "raptor": False, "fn": run_contextual_bm25},
    "contextual_hybrid":        {"heavy": False, "colbert": False, "raptor": False, "fn": run_contextual_hybrid},
    "late_chunking":            {"heavy": True,  "colbert": False, "raptor": False, "fn": run_late_chunking},
    "late_interaction":         {"heavy": True,  "colbert": True,  "raptor": False, "fn": run_late_interaction},
    "late_interaction_approx":  {"heavy": True,  "colbert": True,  "raptor": False, "fn": run_late_interaction_approx},
    "raptor_single_vector":     {"heavy": True,  "colbert": False, "raptor": True,  "fn": run_raptor_single_vector},
    "raptor_late_collapsed":    {"heavy": True,  "colbert": True,  "raptor": True,
                                 "fn": lambda ctx: run_raptor_late(ctx, "collapsed", "raptor_late_collapsed")},
    "raptor_late_traversal":    {"heavy": True,  "colbert": True,  "raptor": True,
                                 "fn": lambda ctx: run_raptor_late(ctx, "traversal", "raptor_late_traversal")},
    "two_stage_dense":          {"heavy": False, "colbert": False, "raptor": False, "fn": run_two_stage},
    "agentic_multihop":         {"heavy": False, "colbert": False, "raptor": False, "fn": run_agentic_multihop},
    "reflection":               {"heavy": False, "colbert": False, "raptor": False, "fn": run_reflection},
    "graph":                    {"heavy": False, "colbert": False, "raptor": False, "fn": run_graph},
}

DEFAULT_PIPELINES = [
    "naive_dense", "hybrid_rag", "hyde", "splade", "splade_hybrid",
    "bm25_prf", "contextual_hybrid", "late_chunking",
    "late_interaction", "raptor_single_vector",
    "raptor_late_collapsed", "raptor_late_traversal",
]


# ── Hardware / environment metadata ──────────────────────────────────

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
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            memory = ctypes.c_ulonglong()
            kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(memory))
            info["ram"] = f"{memory.value // 1024} kB"
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def get_package_versions():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=60)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Raven-Retrieval Unified Benchmark")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "hotpotqa", "fiqa"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Subsample corpus to N docs (judged docs always kept). "
                             "Essential for HotpotQA on low-RAM machines.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="./experiments/runs")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip SPLADE/ColBERT/RAPTOR pipelines")
    parser.add_argument("--skip-colbert", action="store_true", help="Skip ColBERT-based pipelines")
    parser.add_argument("--skip-raptor", action="store_true", help="Skip RAPTOR pipelines")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        help=f"Pipelines to run. Available: {sorted(PIPELINE_REGISTRY.keys())}")
    parser.add_argument("--colbert-checkpoint", default=None,
                        help="Path to trained ColBERT checkpoint (from train_colbert.py)")
    parser.add_argument("--encode-batch-size", type=int, default=16,
                        help="Batch size for ColBERT document encoding (lower for less RAM)")
    parser.add_argument("--hyde-llm", action="store_true",
                        help="Use a real LLM for HyDE generation (default: template mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    # Resolve pipeline list
    if args.pipelines is not None:
        unknown = set(args.pipelines) - set(PIPELINE_REGISTRY.keys())
        if unknown:
            parser.error(f"Unknown pipelines: {unknown}. Available: {sorted(PIPELINE_REGISTRY.keys())}")
        selected = list(args.pipelines)
    else:
        selected = list(DEFAULT_PIPELINES)

    def is_enabled(name):
        spec = PIPELINE_REGISTRY[name]
        if args.skip_heavy and spec["heavy"]:
            return False
        if args.skip_colbert and spec["colbert"]:
            return False
        if args.skip_raptor and spec["raptor"]:
            return False
        return True

    selected = [p for p in selected if is_enabled(p)]
    if not selected:
        parser.error("No pipelines enabled after applying --skip flags.")

    run_id = f"enhanced_{args.dataset}_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAVEN-RETRIEVAL UNIFIED BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {args.dataset} | Pipelines: {len(selected)}")

    # ── Load data ────────────────────────────────────────────────────
    corpus, queries, qrels = load_dataset(args.dataset, args.data_dir)
    if args.max_queries:
        queries, qrels = subsample_queries(queries, qrels, args.max_queries, seed=args.seed)
    if args.max_docs:
        corpus = subsample_corpus(corpus, qrels, args.max_docs, seed=args.seed)
    logger.info(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")

    # ── Shared model instances (created lazily, once) ────────────────
    _shared = {}

    def get_sbert():
        if "sbert" not in _shared:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading shared SBERT (all-MiniLM-L6-v2)...")
            _shared["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
        return _shared["sbert"]

    def get_colbert():
        if "colbert" not in _shared:
            from src.encoder.colbert_encoder import ColbertEncoder
            logger.info("Loading shared ColBERT encoder (bert-base-uncased)...")
            enc = ColbertEncoder(model_name="bert-base-uncased", projection_dim=128)
            if args.colbert_checkpoint:
                enc.load_checkpoint(args.colbert_checkpoint)
                logger.info(f"  + trained checkpoint: {args.colbert_checkpoint}")
            else:
                logger.info("  + UNTRAINED projection head (pass --colbert-checkpoint for trained)")
            enc.eval()
            _shared["colbert"] = enc
        return _shared["colbert"]

    def get_summarizer():
        if "summarizer" not in _shared:
            from src.raptor.summarizer import LLMSummarizer
            _shared["summarizer"] = LLMSummarizer(
                model_name="facebook/bart-large-cnn", fallback_to_extractive=True
            )
        return _shared["summarizer"]

    ctx = {
        "corpus": corpus, "queries": queries, "qrels": qrels, "args": args,
        "sbert": get_sbert, "colbert": get_colbert, "summarizer": get_summarizer,
        "results": {}, "timings": {},
    }
    errors = {}

    # ── Run pipelines ────────────────────────────────────────────────
    for i, name in enumerate(selected, 1):
        logger.info(f"\n[{i}/{len(selected)}] {name}")
        try:
            PIPELINE_REGISTRY[name]["fn"](ctx)
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}", exc_info=True)
            errors[name] = str(e)
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    all_results = ctx["results"]
    timings = ctx["timings"]

    if not all_results:
        logger.error("All pipelines failed — nothing to evaluate. See errors.json.")
        with open(os.path.join(run_dir, "errors.json"), "w") as f:
            json.dump(errors, f, indent=2)
        sys.exit(1)

    # ── Evaluation ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    pipeline_names = list(all_results.keys())
    all_metrics = {}
    per_query_scores = {}
    for name in pipeline_names:
        all_metrics[name] = run_beir_evaluation(qrels, all_results[name], k_values=[1, 3, 5, 10, 100])
        per_query_scores[name] = collect_per_query_ndcg(qrels, all_results[name], k=args.top_k)

    print("\n" + format_results_table([all_metrics[n] for n in pipeline_names], pipeline_names))

    if len(pipeline_names) >= 2:
        logger.info("\n--- Statistical Significance (paired bootstrap, per-query nDCG) ---")
        comparisons = run_all_pairwise_tests(per_query_scores, pipeline_names, n_resamples=10000)
        print(format_significance_table(comparisons))
    else:
        comparisons = []

    # ── Save everything ──────────────────────────────────────────────
    logger.info(f"\nSaving to {run_dir}")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    save_per_query_results(per_query_scores, os.path.join(run_dir, "per_query.json"))
    if comparisons:
        with open(os.path.join(run_dir, "significance.json"), "w") as f:
            json.dump(comparisons, f, indent=2, default=str)
    with open(os.path.join(run_dir, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    if errors:
        with open(os.path.join(run_dir, "errors.json"), "w") as f:
            json.dump(errors, f, indent=2)
    save_run_metadata(
        run_id=run_id,
        seeds={"numpy": args.seed, "torch": args.seed},
        hardware_info=get_hardware_info(),
        package_versions=get_package_versions(),
        output_path=os.path.join(run_dir, "metadata.json"),
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "dataset": args.dataset, "max_queries": args.max_queries,
            "max_docs": args.max_docs, "top_k": args.top_k,
            "pipelines": pipeline_names,
            "colbert_checkpoint": args.colbert_checkpoint,
            "corpus_size": len(corpus), "n_queries": len(queries),
        }, f, indent=2)

    # Dashboard (flatten timings for the scatter plot)
    try:
        from src.eval.visualize import generate_dashboard
        flat_timings = {n: t["total_s"] for n, t in timings.items()}
        dashboard_path = os.path.join(run_dir, "dashboard.html")
        generate_dashboard(all_metrics, flat_timings, comparisons, output_path=dashboard_path)
        logger.info(f"Dashboard: {dashboard_path}")
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name in pipeline_names:
        ndcg10 = all_metrics[name]["ndcg"].get("NDCG@10", 0.0)
        t = timings.get(name, {})
        print(f"  {name:28s} nDCG@10 = {ndcg10:.4f}  "
              f"(index {t.get('index_s', 0):.1f}s, query {t.get('query_s', 0):.1f}s)")
    if errors:
        print(f"\n  FAILED pipelines ({len(errors)}): {', '.join(errors.keys())}")
        print("  See errors.json for details.")

    print(f"\nResults: {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
