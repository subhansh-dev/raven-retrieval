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

## v0.3.0 — Bug fix pass

v0.3.0 fixed 13 bugs (6 critical, 7 logic) and got the RAPTOR + Late Interaction pipeline running for the first time. No full re-run yet — that needs more RAM than was available.

### What changed

- **HyDE**: was crashing on launch (model_name kwarg mismatch) — now runs
- **Late Interaction**: was matching against [PAD] tokens — now attention-mask-trimmed
- **RAPTOR clustering**: Level-0 was using ColBERT token means (wrong space) — now uses SBERT consistently
- **Per-query nDCG**: was silently all zeros (BEIR returns averaged floats, not per-query dict) — now computed in numpy
- **Encoding**: Late Chunking/SPLADE were encoding one text at a time — now batched
- **BM25+PRF**: was re-scoring full corpus per expansion term — now two-stage (first pass + rerank top-100)
- **Timing**: index and query time now reported separately
- **MaxSim/compression**: were using per-token Python loops — now vectorized
- **Reflection**: was scoring keyword coverage over doc ID strings — now scores over actual document text

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

### Pre-registered hypotheses

- **H1:** RAPTOR + late interaction > hybrid on HotpotQA
- **H2:** RAPTOR + late interaction > RAPTOR single-vector
- **H3:** Late interaction > dense (only with trained projection)
- **H4:** RAPTOR + late interaction ≈ hybrid on SciFact
- **H5:** Untrained late interaction < dense on SciFact

H3 and H5 need `--colbert-checkpoint` to test properly.

---

## What's needed to produce v0.3 numbers

1. A machine with ≥8GB RAM (or use `--max-docs 2000` for HotpotQA on smaller machines)
2. Run baselines first (`--skip-heavy`) to confirm the pipeline works end-to-end
3. Train ColBERT (`make train-colbert`) — short run is fine for a quality signal
4. Full benchmark with `--colbert-checkpoint`
5. The runner auto-generates: `metrics.json`, `per_query.json`, `significance.json` (now real p-values), `timings.json` (index/query split), `errors.json` (any pipeline failures, no longer silent), `dashboard.html`, `REPORT.md` (via `make report`)
---

## v0.3.0 — Full 100-Query Benchmark on Google Colab T4 ✅

**Run ID:** `enhanced_scifact_1784635462`
**Date:** 2026-07-21
**Hardware:** Google Colab T4 GPU (15.6 GB VRAM)
**Python:** 3.12 | **PyTorch:** 2.11.0+cu128 | **CUDA:** 12.8
**Dataset:** SciFact (BEIR) — full corpus, 100 queries, top-10
**Seed:** 42 (numpy + torch)
**Total runtime:** 2.8 minutes

### nDCG@10 Results

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|---|
| **🥇 HyDE** | 0.5300 | 0.6536 | 0.6855 | **0.7119** | 0.7119 |
| **🥈 Naive Dense** | 0.5100 | 0.6349 | 0.6741 | 0.6964 | 0.6964 |
| **🥉 Contextual Hybrid** | 0.5500 | 0.6223 | 0.6501 | 0.6823 | 0.6823 |
| **Hybrid RAG** | 0.5100 | 0.6122 | 0.6290 | 0.6668 | 0.6668 |
| **BM25 + PRF** | 0.3400 | 0.4549 | 0.5092 | 0.5285 | 0.5285 |

### Latency

| Pipeline | Index | Query (total) | Per-Query |
|---|---|---|---|
| **BM25 + PRF** | 1.2s | 4.3s | 43ms |
| **Naive Dense** | 24.4s | 3.2s | 32ms |
| **HyDE** | 27.0s | 1.9s | 19ms |
| **Hybrid RAG** | 25.2s | 8.7s | 87ms |
| **Contextual Hybrid** | 28.3s | 9.3s | 93ms |

### Statistical Significance (100 queries, Bonferroni-corrected)

| Comparison | Δ nDCG@10 | Bootstrap p | Bonferroni Sig? |
|---|---|---|---|
| Dense vs BM25 PRF | +0.168 | 0.0000 | ✅ **Yes** |
| Hybrid vs BM25 PRF | +0.141 | 0.0000 | ✅ **Yes** |
| BM25 PRF vs Contextual | −0.170 | 0.0000 | ✅ **Yes** |
| BM25 PRF vs HyDE | −0.183 | 0.0000 | ✅ **Yes** |
| Dense vs Hybrid | +0.027 | 0.1423 | ❌ No |
| Dense vs Contextual | −0.002 | 0.4634 | ❌ No |
| Dense vs HyDE | −0.016 | 0.1253 | ❌ No |
| Hybrid vs Contextual | −0.029 | 0.0193 | ❌ No |
| Hybrid vs HyDE | −0.042 | 0.0601 | ❌ No |
| Contextual vs HyDE | −0.014 | 0.3186 | ❌ No |

### Interpretation

1. **HyDE wins (0.712)** — but only by ~1.5 points over Dense. The10-query run's perfect1.0 was small-sample luck.
2. **Top 4 are statistically indistinguishable** — HyDE, Dense, Contextual, and Hybrid all overlap within noise.
3. **BM25 PRF is significantly worse** (p≈0.000 vs everything else) — TF-IDF expansion hurts on scientific terminology.
4. **Latency trade-off real** — BM25 indexes in1.2s vs25-28s for neural pipelines.
5. **Contextual ≈ Hybrid ≈ Dense** — context prefixes don't help on already self-contained abstracts.


### Historical Note: v0.2 Results (100 queries, CPU, historical)

The v0.2 run on CPU with Dense, Hybrid, and Late Interaction showed:

| Pipeline | nDCG@10 | Total Time |
|---|---|---|
| Naive Dense RAG | 0.6964 | 265s |
| Hybrid RAG | 0.6668 | 263s |
| Late Interaction (untrained) | 0.5801 | 1181s |

These numbers are consistent with the v0.3 Colab run — Dense at 0.696 and Hybrid at 0.667 match exactly. The significance tests from v0.2 were invalid (per-query scores were all zeros due to a BEIR API bug).
