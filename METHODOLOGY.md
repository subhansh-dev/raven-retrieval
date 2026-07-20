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

## Limitations

1. ColBERT projection not trained (fix: `make train-colbert`)
2. Soft-clustering ≈ hard on short chunks
3. CPU-only — ColBERT encoding is slow without GPU
4. BART may fall back to extractive summarization (logged)
5. Single dataset per run
