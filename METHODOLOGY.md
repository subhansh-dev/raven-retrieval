# Methodology

## Problem

RAPTOR builds hierarchical summary trees but retrieves with single-vector similarity. ColBERT uses per-token MaxSim scoring but operates on flat documents. This project combines them: ColBERT-style scoring at every level of a RAPTOR tree.

## RAPTOR Tree Construction

Documents → ~100-token chunks (Level 0). Then iterate:
1. UMAP dimensionality reduction (two-step: global + local)
2. GMM clustering (BIC-selected cluster count, soft assignment threshold 0.1)
3. Summarize each cluster (BART abstractive or TF-IDF extractive fallback)
4. Re-embed summaries, repeat until clusters are too small

Every node stores both pooled embeddings (for single-vector) and per-token ColBERT embeddings (for late interaction).

## ColBERT Encoder

`bert-base-uncased` → `Linear(768 → 128)` → L2 normalize. The projection head is NOT retrieval-trained by default — this is the main quality bottleneck for late interaction. Pass `--colbert-checkpoint` after running `make train-colbert` to fix this.

## Scoring

MaxSim: for each query token, find the most similar document token (cosine), sum across query tokens. Two retrieval strategies on the tree:
- **Collapsed**: flatten all nodes, MaxSim top-k
- **Traversal**: root → top-k → children → repeat until leaves

## Baselines

| Pipeline | What it does |
|---|---|
| Naive Dense | SBERT (all-MiniLM-L6-v2) + cosine, 200-token chunks |
| Hybrid | BM25 + Dense + Reciprocal Rank Fusion (k=60) |
| HyDE | Generate hypothetical answer → embed → retrieve |
| SPLADE | BERT MLM logits → sparse vectors with term expansion |
| BM25+PRF | BM25 + Rocchio pseudo-relevance feedback |
| Contextual | Document title/sentence prepended to each chunk before indexing |
| Late Chunking | Encode full doc first, then split token embeddings by chunk |

## Evaluation

**Datasets:** SciFact (5,183 docs, single-hop), HotpotQA (5.2M docs, multi-hop — use `--max-docs` for low RAM)

**Metrics:** nDCG@k, Recall@k, MAP@k, Precision@k via BEIR + pytrec_eval

**Significance:** Paired bootstrap (10k resamples) + t-test, Bonferroni-corrected. Per-query nDCG computed in numpy (BEIR's evaluate() returns corpus-averaged floats, not per-query — the original code silently got all-zero arrays).

## Hypotheses

- **H1:** RAPTOR + late interaction > hybrid on HotpotQA (multi-hop needs hierarchy)
- **H2:** RAPTOR + late interaction > RAPTOR single-vector (MaxSim > cosine)
- **H3:** Late interaction > dense on nDCG@10 (only with trained projection)
- **H4:** RAPTOR + late interaction ≈ hybrid on SciFact (single-hop doesn't need hierarchy)
- **H5:** Untrained late interaction < dense on SciFact (untrained projection hurts)

## Current Results (v0.3.0, 7 runs, seed=42)

### SciFact Baselines (100 queries, 5,183 docs)

| Pipeline | nDCG@10 | Per-Query |
|---|---|---|
| 🏆 HyDE | 0.7119 | 18.6ms |
| Naive Dense | 0.6964 | 31.8ms |
| Contextual Hybrid | 0.6823 | 93.1ms |
| Hybrid RAG | 0.6668 | 86.8ms |
| BM25 PRF | 0.5285 | 43.1ms |

BM25 PRF is significantly worse than all others (p≈0.000, Bonferroni-corrected). Top 4 are statistically indistinguishable.

### HotpotQA Multi-Hop (50 queries, 2,000 docs)

| Pipeline | nDCG@10 | Per-Query |
|---|---|---|
| 🏆 Hybrid RAG | 0.9249 | 22.5ms |
| Contextual Hybrid | 0.9233 | 22.3ms |
| Naive Dense | 0.9056 | 11.7ms |
| HyDE | 0.8916 | 25.6ms |
| BM25 PRF | 0.8685 | 8.1ms |

Hybrid vs BM25 PRF (p=0.0045) and BM25 PRF vs Contextual (p=0.0031) are Bonferroni-significant. All other pairs: not significant.

### Agentic + Graph (SciFact, 100 queries)

| Pipeline | nDCG@10 | Per-Query |
|---|---|---|
| Graph Retrieval | 0.6964 | 20.7ms |
| Agentic Multi-Hop | 0.6783 | 21.6ms |

Not statistically different (p=0.0934). Two-Stage Dense + Reranker and Reflection Retriever both failed.

### Cross-Dataset Insight

Ranking flips: HyDE wins SciFact (single-hop), Hybrid RAG wins HotpotQA (multi-hop). No single pipeline dominates across both tasks.

### Uncompleted

- Heavy pipelines (ColBERT, SPLADE, Late Chunking): Cell 3 timed out
- RAPTOR pipelines: UMAP bug (fixed in v0.3.1, re-run needed)
- Reranker + Reflection: runtime crashes

## Limitations

1. ColBERT projection not trained (fix: `make train-colbert`)
2. Soft-clustering ≈ hard on short chunks
3. CPU-only — ColBERT encoding is slow without GPU
4. BART may fall back to extractive summarization (logged)
5. Single dataset per run
