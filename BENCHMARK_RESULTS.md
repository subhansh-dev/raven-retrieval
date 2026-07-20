# Raven-Retrieval Benchmark Results

## v0.2.0 — Initial run (historical)

**Run ID:** `full_ablation_1784328531`
**Date:** 2026-07-18
**Hardware:** x86_64 CPU, 6GB RAM, no GPU
**Python:** 3.12.3, PyTorch 2.13.0+cpu

### SciFact (BEIR) — 100 queries, top-10

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|---|
| **Naive Dense RAG** | 0.5100 | 0.6349 | 0.6741 | **0.6964** | 0.6964 |
| **Hybrid RAG (BM25+Dense)** | 0.5100 | 0.6122 | 0.6290 | 0.6668 | 0.6668 |
| **Late Interaction (Flat)** | 0.4900 | 0.5562 | 0.5737 | 0.5801 | 0.5801 |

### Known issues at the time

1. **RAPTOR + Late Interaction couldn't run** — the BART summarizer was incompatible with transformers 5.x. The code now uses `AutoModelForSeq2SeqLM` directly and degrades to extractive summarization with a `_load_failed` flag (no infinite retry).

2. **HotpotQA OOM'd** — the full 5.2M-document corpus exceeds 6GB RAM when encoded with SBERT. The runner now supports `--max-docs` corpus subsampling that *always preserves judged documents* (so metrics stay valid).

3. **"Zero variance" significance results** — *this was misdiagnosed in the original writeup as a "BEIR artifact."* The actual cause (found in v0.3): `collect_per_query_scores` called BEIR's `EvaluateRetrieval.evaluate()`, which returns **corpus-averaged floats**, not a per-query dict. The code then read per-query keys off a float, got nothing, and appended `0.0` for every query. **Every significance test in v0.2 ran on all-zero arrays — the p-values were meaningless, not "artifact-limited."** Fixed in v0.3 with a pure-numpy per-query nDCG implementation (pinned against trec_eval semantics by hand-computed tests).

### Honest interpretation (v0.2 numbers were real; significance was not)

- **Dense > Hybrid on SciFact** is a real, expected result — SciFact is single-hop claim verification where semantic similarity alone is strong; BM25's lexical matching adds noise.
- **Late Interaction < Dense** is a real, expected result — the ColBERT encoder used an untrained (Xavier-init) projection head against fully-trained SBERT. The code supports fine-tuning; it just hadn't been run.
- **No significance claim from v0.2 is valid** — re-run on v0.3 to get real p-values.

---

## v0.3.0 — Engineering pass (this release)

v0.3.0 did not re-run the full benchmark (no suitable hardware in scope). It is the **make-it-actually-work** release: 6 critical bugs fixed, 7 logic bugs fixed, and the novel pipeline (RAPTOR + Late Interaction) is now runnable for the first time. The research results are TBD pending a run on capable hardware.

### What changed (affects future numbers)

| Before | After |
|---|---|
| HyDE crashed on launch (`model_name` ≠ `embedding_model_name`) | HyDE runs; generator LLM actually loads when `use_llm=True`, template mode logged explicitly otherwise |
| Late Interaction matched against **[PAD] tokens** (padded to 256) | Attention-mask-trimmed token embeddings — no pad pollution in scoring or pooling |
| Level-0 RAPTOR clustering used mean-of-ColBERT-tokens (different space than summary levels) | All levels cluster in SBERT space (consistent; the computed-but-discarded SBERT vectors are now used) |
| Per-query nDCG for significance = all zeros | Real per-query nDCG (numpy, trec_eval-compatible) |
| Late Chunking / SPLADE encoded one text at a time | Batched encoding (`encode_documents`, `batch_size=`) |
| BM25+PRF re-scored the full corpus per expansion term (~20 passes/query) | Two-stage PRF: full-corpus first stage, weighted rerank of top-100 candidates only |
| Index time and query time mixed across pipelines | Reported separately for every pipeline (`timings.json`: `index_s`, `query_s`, `per_query_ms`) |
| Approximate MaxSim / compression used per-token Python loops (~256x slow) | Vectorized centroid assignment |
| ReflectionRetriever scored keyword coverage over **doc ID strings** | Scores over real document text (`text_lookup`) |

### Recommended re-run command

```bash
# Fits on a 4-8GB RAM machine now:
python run_enhanced_benchmark.py --dataset scifact --max-queries 100

# Multi-hop dataset (the actual target of the research question):
python run_enhanced_benchmark.py --dataset hotpotqa --max-queries 50 --max-docs 2000

# With a trained ColBERT encoder (the fix for "Late Interaction < Dense"):
python -m src.training.train_colbert --beir-dataset scifact --epochs 3 --hard-negatives
python run_enhanced_benchmark.py --dataset scifact --max-queries 100 \
    --colbert-checkpoint checkpoints/final_model.pt
```

### Pre-registered expectations (still valid — see experiments/preregistration/template.md)

- **H1:** RAPTOR + late interaction (collapsed) > hybrid RAG on nDCG@10 for HotpotQA
- **H2:** RAPTOR + late interaction > RAPTOR single-vector on nDCG@10
- **H3:** Late interaction (flat) > naive dense RAG on nDCG@10 *(only with trained projection)*
- **H4:** RAPTOR + late interaction shows minimal improvement on SciFact (single-hop control)
- **H5:** Untrained late interaction underperforms dense on SciFact

H3 and H5 directly depend on the (now-wired) `--colbert-checkpoint` flag.

---

## What's needed to produce v0.3 numbers

1. A machine with ≥8GB RAM (or use `--max-docs 2000` for HotpotQA on smaller machines)
2. Run baselines first (`--skip-heavy`) to confirm the pipeline works end-to-end
3. Train ColBERT (`make train-colbert`) — short run is fine for a quality signal
4. Full benchmark with `--colbert-checkpoint`
5. The runner auto-generates: `metrics.json`, `per_query.json`, `significance.json` (now real p-values), `timings.json` (index/query split), `errors.json` (any pipeline failures, no longer silent), `dashboard.html`, `REPORT.md` (via `make report`)
---

## v0.3.0 — Actual Benchmark Run ✅

**Run ID:** `enhanced_scifact_1784579644`
**Date:** 2026-07-21
**Hardware:** x86_64 CPU, ~6GB RAM, no GPU
**Python:** 3.12.3 | **PyTorch:** 2.13.0+cpu | **Transformers:** 5.14.1
**Dataset:** SciFact (BEIR) — 500 docs subsampled (judged docs preserved), 10 queries, top-10
**Seed:** 42 (numpy + torch)

### Test Suite Results

- **Core tests:** 42/42 passed (numpy-only, no torch required)
- **Full pytest:** 55/55 passed in 85.48s
- All 7 test modules pass: integration, maxsim, metrics, pipelines smoke, raptor, significance, utils

### nDCG@10 Results

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|---|
| **HyDE** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Naive Dense RAG** | 0.9000 | 0.9631 | 0.9631 | 0.9631 | 0.9631 |
| **Contextual Hybrid** | 0.8000 | 0.8000 | 0.8000 | 0.8690 | 0.8690 |
| **Hybrid RAG** | 0.8000 | 0.8000 | 0.8000 | 0.8672 | 0.8672 |
| **BM25 + Rocchio PRF** | 0.6000 | 0.6631 | 0.7018 | 0.7018 | 0.7018 |

### Latency

| Pipeline | Index Time | Query Time | Total | Per-Query |
|---|---|---|---|---|
| **BM25 + Rocchio PRF** | 0.07s | 0.04s | 0.10s | 3.7ms |
| **HyDE** | 25.64s | 0.17s | 25.82s | 17.4ms |
| **Hybrid RAG** | 26.89s | 0.17s | 27.06s | 17.0ms |
| **Contextual Hybrid** | 26.17s | 0.22s | 26.39s | 22.4ms |
| **Naive Dense RAG** | 27.37s | 0.13s | 27.49s | 12.6ms |

### Per-Query nDCG@10 Detail

| Query # | Naive Dense | Hybrid RAG | BM25 PRF | Contextual | HyDE |
|---|---|---|---|---|---|
| Q1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q3 | 1.0000 | 1.0000 | 0.6309 | 1.0000 | 1.0000 |
| Q4 | 1.0000 | 0.3562 | 0.0000 | 0.3333 | 1.0000 |
| Q5 | 1.0000 | 1.0000 | 0.3869 | 1.0000 | 1.0000 |
| Q6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q8 | 0.6309 | 0.3155 | 0.0000 | 0.3333 | 1.0000 |
| Q9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Q10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Mean** | **0.9631** | **0.8672** | **0.7018** | **0.8690** | **1.0000** |

### Statistical Significance (10,000 bootstrap resamples, Bonferroni-corrected)

| Comparison | Δ nDCG@10 | Bootstrap p | t-test p | Sig? |
|---|---|---|---|---|
| Dense vs BM25 PRF | +0.261 | 0.007 | 0.052 | ❌ |
| Hybrid vs BM25 PRF | +0.165 | 0.007 | 0.047 | ❌ |
| BM25 PRF vs Contextual | −0.165 | 0.007 | 0.047 | ❌ |
| BM25 PRF vs HyDE | −0.298 | 0.007 | 0.053 | ❌ |
| Dense vs Hybrid | +0.096 | 0.107 | 0.195 | ❌ |
| Dense vs Contextual | +0.096 | 0.107 | 0.201 | ❌ |
| Hybrid vs Contextual | +0.001 | 0.458 | 0.873 | ❌ |
| Dense vs HyDE | −0.037 | 0.359 | 0.343 | ❌ |
| Hybrid vs HyDE | −0.133 | 0.107 | 0.168 | ❌ |
| Contextual vs HyDE | −0.133 | 0.107 | 0.168 | ❌ |

> With 10 queries, no comparison reaches Bonferroni significance (α=0.005). BM25 PRF vs Dense/Hybrid (p=0.007) is near threshold. Full 100-query run needed for proper power.

### Interpretation

1. **HyDE perfect score** — Hypothetical document generation bridged the semantic gap perfectly on these queries. Partly small-sample, but directionally consistent.
2. **Naive Dense strong (0.96)** — SciFact's single-hop scientific claims are ideal for SBERT.
3. **Contextual ≈ Hybrid (0.87)** — Context prefixes didn't help on already self-contained abstracts.
4. **BM25 PRF weakest (0.70)** — TF-IDF query expansion added noise on precise scientific terminology.
5. **BM25 PRF fastest** — 260x faster indexing, 3-6x faster querying. Valid when speed > accuracy.
