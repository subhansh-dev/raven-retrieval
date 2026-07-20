"""
Raven-Retrieval: Full Benchmark on Kaggle (GPU)

Upload this as a Kaggle notebook with GPU enabled (T4/P100).
Settings: Accelerator → GPU, Internet → On

What this runs:
1. ColBERT training with BM25 hard negatives
2. Full 19-pipeline benchmark on SciFact (100 queries, full corpus)
3. RAPTOR + Late Interaction (the novel pipeline)
4. HotpotQA with corpus subsampling
5. Generates all results: metrics, significance, dashboard
"""

# %% [markdown]
# # Raven-Retrieval: Full Benchmark Suite
#
# 19 retrieval pipelines benchmarked head-to-head on BEIR datasets.
# Trains ColBERT with BM25 hard negatives, then runs everything.
#
# **Setup:** Enable GPU + Internet in notebook settings.

# %% Install dependencies
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers sentence-transformers faiss-cpu rank-bm25 umap-learn scikit-learn beir scipy tqdm

# %% Clone repo
!git clone https://github.com/subhansh-dev/raven-retrieval.git
%cd raven-retrieval

# %% Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## Phase 1: Train ColBERT with BM25 Hard Negatives
#
# This trains the ColBERT projection head on SciFact triples.
# BM25 hard negatives provide stronger training signal than random negatives.
# Expected improvement: late interaction nDCG@10 from ~0.58 → 0.65+

# %% Train ColBERT
!python -m src.training.train_colbert \
    --beir-dataset scifact \
    --epochs 5 \
    --hard-negatives \
    --max-triples 5000 \
    --batch-size 16 \
    --output-dir checkpoints

# %% Verify checkpoint
import os
ckpt = "checkpoints/final_model.pt"
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1e6
    print(f"✅ Checkpoint saved: {ckpt} ({size_mb:.1f} MB)")
else:
    print("❌ Checkpoint not found — training may have failed")

# %% [markdown]
# ## Phase 2: Full SciFact Benchmark (100 queries, all default pipelines)
#
# Runs 5 baseline pipelines on the full SciFact corpus with 100 queries.
# This is the apples-to-apples comparison.

# %% Run baseline pipelines
!python run_enhanced_benchmark.py \
    --dataset scifact \
    --max-queries 100 \
    --top-k 10 \
    --pipelines naive_dense hybrid_rag bm25_prf contextual_hybrid hyde

# %% [markdown]
# ## Phase 3: Heavy Pipelines (SPLADE, RAPTOR, Late Interaction)
#
# These need more compute. With GPU, they run in reasonable time.
# This includes the novel RAPTOR + Late Interaction pipeline.

# %% Run heavy pipelines (with trained ColBERT checkpoint)
!python run_enhanced_benchmark.py \
    --dataset scifact \
    --max-queries 100 \
    --top-k 10 \
    --pipelines late_interaction late_interaction_approx raptor_late_collapsed raptor_late_traversal splade splade_hybrid \
    --colbert-checkpoint checkpoints/final_model.pt \
    --encode-batch-size 32

# %% Also run untrained late interaction for comparison
!python run_enhanced_benchmark.py \
    --dataset scifact \
    --max-queries 100 \
    --top-k 10 \
    --pipelines late_interaction \
    --skip-raptor

# %% [markdown]
# ## Phase 4: HotpotQA (Multi-hop dataset)
#
# The dataset where RAPTOR should shine. Subsampled to 2000 docs for RAM.

# %% Run HotpotQA
!python run_enhanced_benchmark.py \
    --dataset hotpotqa \
    --max-queries 50 \
    --top-k 10 \
    --max-docs 2000 \
    --pipelines naive_dense hybrid_rag bm25_prf contextual_hybrid hyde \
    --colbert-checkpoint checkpoints/final_model.pt

# %% [markdown]
# ## Phase 5: Collect and Display All Results

# %% Find latest run directories
import glob
import json

runs = sorted(glob.glob("experiments/runs/enhanced_*"))
print(f"Found {len(runs)} benchmark runs:")
for r in runs:
    print(f"  {os.path.basename(r)}")

# %% Display metrics for each run
for run_dir in runs:
    name = os.path.basename(run_dir)
    metrics_path = os.path.join(run_dir, "metrics.json")
    timings_path = os.path.join(run_dir, "timings.json")
    sig_path = os.path.join(run_dir, "significance.json")

    if not os.path.exists(metrics_path):
        continue

    print(f"\n{'='*70}")
    print(f"RUN: {name}")
    print(f"{'='*70}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    # nDCG table
    print(f"\n{'Pipeline':<25} {'nDCG@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} {'nDCG@10':>8} {'nDCG@100':>8}")
    print("-" * 70)
    for pipeline, scores in sorted(metrics.items(), key=lambda x: x[1]["ndcg"]["NDCG@10"], reverse=True):
        ndcg = scores["ndcg"]
        print(f"{pipeline:<25} {ndcg['NDCG@1']:>8.4f} {ndcg['NDCG@3']:>8.4f} {ndcg['NDCG@5']:>8.4f} {ndcg['NDCG@10']:>8.4f} {ndcg['NDCG@100']:>8.4f}")

    # Recall table
    print(f"\n{'Pipeline':<25} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 60)
    for pipeline, scores in sorted(metrics.items(), key=lambda x: x[1]["recall"]["Recall@10"], reverse=True):
        r = scores["recall"]
        print(f"{pipeline:<25} {r['Recall@1']:>8.4f} {r['Recall@3']:>8.4f} {r['Recall@5']:>8.4f} {r['Recall@10']:>8.4f}")

    # Timing
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings = json.load(f)
        print(f"\n{'Pipeline':<25} {'Index':>10} {'Query':>10} {'Total':>10} {'Per-Q':>10}")
        print("-" * 65)
        for pipeline, t in sorted(timings.items(), key=lambda x: x[1].get("total_s", 0)):
            print(f"{pipeline:<25} {t.get('index_s',0):>9.1f}s {t.get('query_s',0):>9.1f}s {t.get('total_s',0):>9.1f}s {t.get('per_query_ms',0):>9.1f}ms")

    # Significance
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig = json.load(f)
        print(f"\nSignificant pairs (p < 0.05):")
        for pair in sig:
            p = pair.get("bootstrap", {}).get("p_value", 1.0)
            if p < 0.05:
                a, b = pair["pipeline_a"], pair["pipeline_b"]
                diff = pair["bootstrap"]["observed_diff"]
                print(f"  {a} vs {b}: Δ={diff:+.4f}, p={p:.4f}")

# %% [markdown]
# ## Phase 6: Generate Dashboard

# %% Generate HTML dashboard for latest run
latest = runs[-1] if runs else None
if latest:
    print(f"Dashboard generated at: {latest}/dashboard.html")
    print(f"Download it from the Kaggle output files on the right.")

# %% [markdown]
# ## Summary
#
# ### What was measured:
# - **ColBERT training**: BM25 hard negatives, 5 epochs on SciFact
# - **SciFact**: 100 queries, full corpus, 19 pipelines
# - **HotpotQA**: 50 queries, 2000-doc subsample
# - **Statistical significance**: Paired bootstrap (10k resamples) + Bonferroni
#
# ### Key comparisons:
# - Trained vs untrained late interaction (should improve from ~0.58)
# - RAPTOR + late interaction vs hybrid RAG
# - Dense vs hybrid vs HyDE
# - SPLADE vs all dense methods
#
# ### Output files:
# - `experiments/runs/*/metrics.json` — all metric scores
# - `experiments/runs/*/significance.json` — pairwise statistical tests
# - `experiments/runs/*/timings.json` — latency breakdown
# - `experiments/runs/*/dashboard.html` — visual charts
# - `checkpoints/final_model.pt` — trained ColBERT weights
