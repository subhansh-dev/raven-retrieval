# raven-retrieval

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.14-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Tests](https://img.shields.io/badge/Tests-50%2B%20Passing-brightgreen)
![Pipelines](https://img.shields.io/badge/Pipelines-19-blueviolet)
![ColBERT](https://img.shields.io/badge/ColBERT-Late%20Interaction-orange)
![RAPTOR](https://img.shields.io/badge/RAPTOR-Hierarchical%20Tree-purple)
![SPLADE](https://img.shields.io/badge/SPLADE-Sparse%20Expansion-green)
![HyDE](https://img.shields.io/badge/HyDE-Hypothetical%20Docs-yellow)
![FAISS](https://img.shields.io/badge/FAISS-CPU-005C84?logo=meta&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black?logo=githubactions)

**19 retrieval pipelines. Real benchmarks. Honest results.**

*Late interaction, hierarchical trees, sparse expansion, hypothetical documents, contextual chunking, agentic decomposition, reflection, graph retrieval — all benchmarked head-to-head on BEIR datasets with proper statistical significance testing.*

> **v0.3.0** — engineering pass: 6 critical bugs fixed (incl. the novel RAPTOR+MaxSim pipeline that previously couldn't run, and per-query significance scores that were silently all-zero), 7 logic bugs fixed, batched/memory-safe encoding, `--max-docs` corpus subsampling for low-RAM machines, `--colbert-checkpoint` wiring. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the honest before/after.

</div>

---

## What Is This

This is a retrieval research framework. Not a production search engine, not a RAG template — a tool for actually comparing retrieval methods against each other on equal footing.

Most retrieval blog posts tell you "Method X is better" without showing you the full picture. This project implements 14 different pipelines, runs them all on the same datasets with the same chunking, same evaluation metrics, and same statistical tests. The results are what they are. No cherry-picking.

The core question it answers: **does applying ColBERT-style late interaction scoring at every level of a RAPTOR hierarchical tree actually improve retrieval quality over simpler approaches?** Everything else is scaffolding to answer that question properly.

### The 19 Pipelines (registry in `run_enhanced_benchmark.py`)

#### Core Pipelines

| # | Pipeline | What It Does | Reference |
|---|---|---|---|
| 1 | **Naive Dense RAG** | Fixed-size chunks (200 tokens, 50 overlap) → SBERT (all-MiniLM-L6-v2) → cosine similarity → aggregate by doc (max chunk score) | Standard baseline |
| 2 | **Hybrid RAG** | BM25Okapi + SBERT dense, fused with Reciprocal Rank Fusion (k=60) | Robertson et al. |
| 3 | **ColBERT Late Interaction** | Per-token BERT embeddings (768→128 projection) → brute-force MaxSim scoring | Khattab & Zaharia, SIGIR 2020 |
| 4 | **RAPTOR + Late Interaction** | Hierarchical summary tree with ColBERT per-token MaxSim at every node (leaf chunks AND summary nodes) | Sarthi et al., ICLR 2024 + novel combination |

#### Research Pipelines (2024-2025)

| # | Pipeline | What It Does | Reference |
|---|---|---|---|
| 5 | **HyDE** | Generate hypothetical answer (TinyLlama or template fallback) → embed that → retrieve. Bridges query-document semantic gap | Gao et al., ACL 2023 |
| 6 | **SPLADE** | BERT MLM logits → ReLU + log(1+x) → max over positions → sparse vector. Terms not in original text get activated (semantic expansion). IDF re-weighting on query | Formal et al., SIGIR 2021 |
| 7 | **SPLADE + Dense Hybrid** | SPLADE sparse + SBERT dense + RRF fusion. Best of both lexical and semantic worlds | Hybrid approach |
| 8 | **BM25 + Rocchio PRF** | BM25 initial retrieval → extract TF*IDF-weighted terms from top-5 docs → expand query → re-retrieve. Zero neural networks needed | Rocchio, 1971 |
| 9 | **Contextual Retrieval** | Prepend document title or first sentence as context prefix to every chunk before embedding/BM25 indexing. Addresses "lost in the middle" problem | Anthropic, 2024 |
| 10 | **Contextual Hybrid** | Contextual BM25 + Contextual Dense + RRF. Both retrieval channels get the context enrichment | Extended Anthropic |
| 11 | **Late Chunking** | Embed entire document through BERT first, THEN split token embeddings into chunks and pool. Preserves cross-chunk context that normal chunking destroys | Jina AI, 2024 |
| 12 | **RAPTOR + Late Traversal** | Top-down greedy traversal: score root nodes with MaxSim, pick top-k, descend to children, repeat until leaves. More efficient than collapsed | Novel combination |
| 13 | **Agentic Multi-Hop** | Template or LLM-based query decomposition → retrieve for each sub-query → aggregate scores across sub-queries | Agentic RAG, 2025 |
| 14 | **Reflection Retriever** | Retrieve → evaluate context sufficiency (keyword coverage over **real document text**, not doc IDs) → if insufficient, reformulate query → re-retrieve (up to 2 iterations). Self-RAG pattern | Self-RAG, Asai et al. 2023 |
| 15 | **Graph Retrieval** | Build cosine-similarity document graph → label-propagation community detection → expand results with community members. Memory-safe blocked edge construction | LightRAG-style, 2024 |
| 16 | **Two-Stage Dense + Reranker** | Dense top-100 → cross-encoder (ms-marco-MiniLM) rerank to top-10. The standard production architecture | Nogueira & Cho, 2019 |
| 17 | **Approximate Late Interaction** | PLAID-style: FAISS K-means centroids → centroid-overlap candidate pruning → full MaxSim rerank. `compute_fidelity()` vs brute force | PLAID, SIGMOD 2023 |
| 18 | **Contextual Dense** | Contextual chunking (Anthropic) → dense only | Anthropic, 2024 |
| 19 | **Contextual BM25** | Contextual chunking → BM25 only | Anthropic, 2024 |

#### Composition Modules

These aren't standalone pipelines — they wrap other retrievers to add capabilities.

| Module | What It Does |
|---|---|
| **Cross-Encoder Reranker** | Second-stage reranking with cross-attention (cross-encoder/ms-marco-MiniLM-L-6-v2). Bi-encoder top-100 → cross-encoder top-10 |
| **Two-Stage Retriever** | Wraps any first-stage retriever + any reranker into a complete pipeline |
| **Document Graph** | Build cosine similarity graph → label propagation community detection → expand results with community members. Good for "what are the main themes" questions |
| **Residual Compressor** | ColBERTv2-style compression: K-means centroids + quantized residuals. ~30x storage reduction (512 bytes → ~17 bytes per token) |

---

## How It Actually Works

### The Encoder

Everything starts with the encoder. Two options:

**SBERT** (for dense/hybrid baselines): `all-MiniLM-L6-v2`, produces a single 384-dim vector per chunk. Fast, trained on millions of retrieval pairs, works out of the box.

**ColBERT** (for late interaction): `bert-base-uncased` backbone → `Linear(768 → 128)` projection → L2 normalize per token. Produces a matrix of per-token embeddings instead of a single vector. The projection head is Xavier-initialized but NOT trained on retrieval triples in the default setup. This is a known limitation — the ColBERT encoder needs fine-tuning to reach its potential. The code supports training via `ColbertContrastiveEncoder` with contrastive loss or InfoNCE.

The encoder lives in `src/encoder/colbert_encoder.py`. Key methods:
- `encode_query(text, max_length=64)` → tensor of shape (n_query_tokens, 128)
- `encode_document(text, max_length=256)` → tensor of shape (n_doc_tokens, 128)
- `encode_pooled(texts, strategy="mean"|"max"|"cls")` → single vector per text (for clustering/GMM)
- `get_temperature()` → learned temperature parameter (exp(log_tau))

### MaxSim: The Core Scoring Algorithm

ColBERT's scoring function. For query Q = [q₁, ..., qₘ] and document D = [d₁, ..., dₙ]:

```
MaxSim(Q, D) = Σᵢ maxⱼ sim(qᵢ, dⱼ)
```

For each query token, find the most similar document token (cosine similarity), then sum those maximums across all query tokens. This captures fine-grained token-level relevance that single-vector cosine misses.

Implemented in `src/maxsim/brute_force.py` — works entirely in numpy, no torch required for scoring. The `maxsim_score_batch()` function handles concatenated document embeddings with offset tracking for efficiency.

### RAPTOR Tree Construction

The hierarchical tree is built offline (one-time cost):

1. **Chunk** the corpus into ~100-token segments (Level 0 leaf nodes)
2. **Embed** each chunk (SBERT for pooled embeddings, ColBERT for token embeddings)
3. **Cluster** using two-step UMAP + GMM:
   - Global step: `n_neighbors ≈ sqrt(N-1)` preserves corpus-wide structure
   - Local step: `n_neighbors ≈ 10` within each global cluster captures fine-grained structure
   - Cluster count selected via BIC (Bayesian Information Criterion) to penalize overfitting
   - Soft assignment: nodes with P(cluster) > 0.1 go to multiple clusters
4. **Summarize** each cluster (BART abstractive, or extractive TF-IDF fallback if model fails)
5. **Embed** the summaries, add as parent nodes
6. **Repeat** steps 3-5 until clusters have ≤ `min_cluster_size` members (default: 3)

The tree ends up with levels of increasing abstraction — raw text at the bottom, meta-summaries at the top. Every node stores both pooled embeddings (for single-vector retrieval) AND per-token ColBERT embeddings (for late interaction).

Built in `src/raptor/builder.py`. The `LateInteractionRaptor` in `src/combined/late_raptor.py` does the same thing but specifically stores ColBERT token embeddings at every node.

### Two Retrieval Strategies on the Tree

**Collapsed**: Flatten all nodes (all levels) into a single pool. MaxSim query against every node. Take top-k. Simple, finds the best matching node regardless of level.

**Traversal**: Start at root nodes. MaxSim score each, take top-k. Descend to their children. Repeat until you hit leaf nodes. More efficient, naturally moves from abstract to specific.

### Approximate MaxSim (PLAID-style)

For when brute-force over millions of documents is too slow:

1. **K-means** all token embeddings into centroids (FAISS)
2. **Encode** each document as centroid IDs + residuals
3. **Query**: find top centroids closest to query tokens → candidate documents via centroid overlap → decompress residuals → full MaxSim reranking

Implemented in `src/maxsim/approximate.py`. The `ApproximateMaxSim` class also has `compute_fidelity()` to measure how much quality you lose vs brute-force.

### Residual Compression (ColBERTv2-style)

Standard ColBERT stores 128 floats per token = 512 bytes. ColBERTv2 compresses this:

1. Learn centroids (K-means)
2. For each token: store centroid ID (1 byte) + quantized residual (2 bits per dim)
3. Total: ~17 bytes per token instead of 512. That's ~30x compression.

At query time: centroid interaction for candidate generation, decompress residuals for full MaxSim. Implemented in `src/maxsim/compression.py` with `ResidualCompressor` and `CompressedCorpusIndex`.

### SPLADE: Learned Sparse Retrieval

Instead of dense vectors, SPLADE produces sparse vectors over the BERT vocabulary (~30k dimensions). The trick: use BERT's MLM (Masked Language Model) head, apply ReLU + log(1+x) activation, take max across sequence positions.

This naturally expands terms — "cat" might activate "feline", "kitten", "pet" in the vocabulary. The result is a sparse vector that combines lexical matching (like BM25) with semantic expansion (like dense retrieval).

Implemented in `src/baselines/splade.py`. The `get_expansion_terms()` method shows you exactly which terms SPLADE would add for a given input — useful for interpretability.

### HyDE: Hypothetical Document Embeddings

Instead of embedding the query directly:
1. Generate a hypothetical answer to the query (using TinyLlama or a template fallback)
2. Embed the hypothetical answer
3. Retrieve real documents similar to the hypothetical

The insight: a hypothetical answer is semantically closer to real documents than the raw question. The embedding model's "dense bottleneck" filters out hallucinated details.

Implemented in `src/baselines/hyde.py`. Falls back to 3 rotating templates based on query hash if no LLM is available.

### Contextual Retrieval (Anthropic's Approach)

When you chunk a document, each chunk loses its document-level context. "The company's revenue grew by 3%" — which company? What quarter?

Solution: prepend a short context prefix to each chunk before embedding. Something like "Document: ACME Corp Q2 2023 SEC filing. The company's revenue grew by 3%..."

Three flavors in `src/baselines/contextual.py`:
- **ContextualDenseRetriever**: embed context-enriched chunks
- **ContextualBM25Retriever**: BM25 over context-enriched chunks
- **ContextualHybridRetriever**: both + RRF

Anthropic reported 35-49% reduction in failed retrievals with this technique.

### Late Chunking (Jina's Approach)

Standard pipeline: chunk text → embed each chunk independently. Problem: each chunk's embedding has no awareness of surrounding context.

Late chunking flips this: embed the entire document first → then split the resulting token embeddings by position → pool per chunk.

The key: when the transformer processes the full document, each token's embedding is contextualized by ALL other tokens, including those in other chunks. So chunk embeddings naturally preserve cross-chunk context.

Implemented in `src/baselines/late_chunking.py`. Limited by BERT's 512-token context window in the default setup — needs a long-context model (like Jina-embeddings-v2 with 8K tokens) to really shine.

### Agentic Patterns

**Query Decomposition** (`QueryDecomposer` in `src/baselines/agentic.py`):
- Detects multi-hop queries via conjunctions ("and", "but also"), comparison patterns ("compare", "vs"), and relative clauses ("that", "which")
- Splits into sub-queries, retrieves for each, aggregates scores
- Also supports LLM-based decomposition (optional)

**Reflection** (`ReflectionRetriever`):
- After initial retrieval, evaluates context sufficiency using keyword coverage
- If <50% of query keywords appear in retrieved context, reformulates and re-retrieves
- Up to 2 iterations, each reformulation gets more specific or broader

**Multi-Hop** (`MultiHopRetriever`):
- Combines decomposition + per-sub-query retrieval + score aggregation
- Documents appearing in results for multiple sub-queries get higher aggregate scores

---

## The Evaluation System

This is where the framework takes itself seriously.

### Metrics

All computed via BEIR's `EvaluateRetrieval` wrapping `pytrec_eval` (the standard IR evaluation tool):
- **nDCG@k**: Normalized Discounted Cumulative Gain at k ∈ {1, 3, 5, 10, 100}
- **MAP@k**: Mean Average Precision
- **Recall@k**: Fraction of relevant documents retrieved
- **Precision@k**: Fraction of retrieved documents that are relevant

### Statistical Significance

Not just "pipeline A scored higher than pipeline B." Actual statistical tests:

- **Paired bootstrap test**: 10,000 resamples per comparison. Computes observed difference, p-value, and 95% confidence interval
- **Paired t-test**: Cross-check with parametric test (scipy `ttest_rel`)
- **Bonferroni correction**: Adjusts significance threshold for multiple comparisons (6 pairwise tests across 4 pipelines → α/6)

All implemented in `src/eval/significance.py`. The `run_all_pairwise_tests()` function handles all C(n,2) pairs automatically.

### Dashboard and Reports

**HTML Dashboard** (`src/eval/visualize.py`): Self-contained HTML file with embedded SVG charts:
- Bar chart: nDCG@10 by pipeline
- Grouped bar chart: nDCG@k breakdown
- Radar chart: multi-metric comparison
- Scatter plot: latency vs quality tradeoff
- Significance table with color-coded results

**Markdown Report** (`src/eval/report.py`): Auto-generated with:
- nDCG table (best highlighted in bold)
- Recall table
- Latency table with relative speedup
- Statistical significance table
- Key findings analysis
- Production and research recommendations

### Pre-registered Hypotheses

Expectations are written BEFORE the benchmark run (see `experiments/preregistration/template.md`). Results are compared against stated expectations, including where the system was wrong. This prevents HARKing (Hypothesizing After Results are Known).

---

## Benchmark Results

### Full Test Suite Results

**Core Tests (no torch required):**
```
==================================================
Raven-Retrieval Core Test Suite
==================================================
  ✅ chunker_basic
  ✅ chunker_overlap
  ✅ chunker_corpus
  ✅ tree_ops
  ✅ maxsim_basic
  ✅ maxsim_ranking
  ✅ maxsim_batch
  ✅ umap_reduction
  ✅ cluster_count
  ✅ soft_assignment
  ✅ extractive_summarizer
  ✅ extractive_empty
  ✅ compression_roundtrip
  ✅ compression_ratio
  ✅ compression_save_load
  ✅ bootstrap_known_diff
  ✅ bootstrap_no_diff
  ✅ bonferroni
  ✅ pairwise
  ✅ dashboard
  ✅ report
  ✅ query_decomposition
  ✅ centroid_index

Results: 42 passed, 0 failed, 42 total assertions
🎉 ALL TESTS PASSED
```

**Full pytest Suite (torch required):**
```
55 passed, 2 warnings in 85.48s

Tests by module:
  test_integration.py     3/3 passed  (pipeline wiring, tree flat retrieval, MaxSim end-to-end)
  test_maxsim.py          6/6 passed  (scoring, ranking, batch, centroid index, approximate, fidelity)
  test_metrics.py         9/9 passed  (per-query nDCG, recall, precision, MAP, trec_eval alignment)
  test_pipelines_smoke.py 11/11 passed (DI-based contract tests for every retriever)
  test_raptor.py          6/6 passed  (chunker, UMAP, cluster count, soft cluster, tree ops)
  test_significance.py    5/5 passed  (bootstrap, t-test, Bonferroni, pairwise)
  test_utils.py           15/15 passed (chunking, aggregation, RRF, timer, masking, L2 normalize)
```

### SciFact (BEIR) — v0.3.0 Benchmark Run

**Run ID:** `enhanced_scifact_1784579644`
**Date:** 2026-07-21
**Hardware:** x86_64 CPU, ~6GB RAM, no GPU
**Python:** 3.12.3 | **PyTorch:** 2.13.0+cpu | **Transformers:** 5.14.1
**Dataset:** SciFact (BEIR) — 500 docs subsampled (judged docs preserved), 10 queries, top-10
**Seed:** 42 (numpy + torch)

#### nDCG Scores (Normalized Discounted Cumulative Gain)

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|---|
| **🥇 HyDE** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **🥈 Naive Dense RAG** | 0.9000 | 0.9631 | 0.9631 | 0.9631 | 0.9631 |
| **🥉 Contextual Hybrid** | 0.8000 | 0.8000 | 0.8000 | 0.8690 | 0.8690 |
| **Hybrid RAG (BM25+Dense)** | 0.8000 | 0.8000 | 0.8000 | 0.8672 | 0.8672 |
| **BM25 + Rocchio PRF** | 0.6000 | 0.6631 | 0.7018 | 0.7018 | 0.7018 |

#### MAP Scores (Mean Average Precision)

| Pipeline | MAP@1 | MAP@3 | MAP@5 | MAP@10 | MAP@100 |
|---|---|---|---|---|---|
| **HyDE** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Naive Dense RAG** | 0.9000 | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| **Contextual Hybrid** | 0.8000 | 0.8000 | 0.8000 | 0.8310 | 0.8310 |
| **Hybrid RAG** | 0.8000 | 0.8000 | 0.8000 | 0.8292 | 0.8292 |
| **BM25 + Rocchio PRF** | 0.6000 | 0.6500 | 0.6700 | 0.6700 | 0.6700 |

#### Recall Scores

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@100 |
|---|---|---|---|---|---|
| **HyDE** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Naive Dense RAG** | 0.9000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Contextual Hybrid** | 0.8000 | 0.8000 | 0.8000 | **1.0000** | **1.0000** |
| **Hybrid RAG** | 0.8000 | 0.8000 | 0.8000 | **1.0000** | **1.0000** |
| **BM25 + Rocchio PRF** | 0.6000 | 0.7000 | 0.8000 | 0.8000 | 0.8000 |

#### Precision Scores

| Pipeline | P@1 | P@3 | P@5 | P@10 | P@100 |
|---|---|---|---|---|---|
| **HyDE** | **1.0000** | 0.3333 | 0.2000 | 0.1000 | 0.0100 |
| **Naive Dense RAG** | 0.9000 | 0.3333 | 0.2000 | 0.1000 | 0.0100 |
| **Contextual Hybrid** | 0.8000 | 0.2667 | 0.1600 | 0.1000 | 0.0100 |
| **Hybrid RAG** | 0.8000 | 0.2667 | 0.1600 | 0.1000 | 0.0100 |
| **BM25 + Rocchio PRF** | 0.6000 | 0.2333 | 0.1600 | 0.0800 | 0.0080 |

#### Latency (Index + Query Time)

| Pipeline | Index Time | Query Time | Total Time | Per-Query Latency |
|---|---|---|---|---|
| **BM25 + Rocchio PRF** | **0.07s** | **0.04s** | **0.10s** | **3.7ms** |
| **HyDE** | 25.64s | 0.17s | 25.82s | 17.4ms |
| **Contextual Hybrid** | 26.17s | 0.22s | 26.39s | 22.4ms |
| **Hybrid RAG** | 26.89s | 0.17s | 27.06s | 17.0ms |
| **Naive Dense RAG** | 27.37s | 0.13s | 27.49s | 12.6ms |

#### Per-Query nDCG@10 Detail

| Query # | Naive Dense | Hybrid RAG | BM25 PRF | Contextual Hybrid | HyDE |
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

#### Statistical Significance (Paired Bootstrap + t-test, Bonferroni-corrected)

| Comparison | Observed Δ | Bootstrap p | t-test p | Bonferroni Sig? |
|---|---|---|---|---|
| Naive Dense vs Hybrid RAG | +0.096 | 0.107 | 0.195 | ❌ No |
| Naive Dense vs BM25 PRF | +0.261 | 0.007 | 0.052 | ❌ No |
| Naive Dense vs Contextual Hybrid | +0.096 | 0.107 | 0.201 | ❌ No |
| Naive Dense vs HyDE | −0.037 | 0.359 | 0.343 | ❌ No |
| Hybrid RAG vs BM25 PRF | +0.165 | 0.007 | 0.047 | ❌ No |
| Hybrid RAG vs Contextual Hybrid | +0.001 | 0.458 | 0.873 | ❌ No |
| Hybrid RAG vs HyDE | −0.133 | 0.107 | 0.168 | ❌ No |
| BM25 PRF vs Contextual Hybrid | −0.165 | 0.007 | 0.047 | ❌ No |
| BM25 PRF vs HyDE | −0.298 | 0.007 | 0.053 | ❌ No |
| Contextual Hybrid vs HyDE | −0.133 | 0.107 | 0.168 | ❌ No |

> **Note:** With only 10 queries, statistical power is limited. After Bonferroni correction (α=0.05/10=0.005), no pairwise comparison reaches significance. The BM25 PRF vs Dense/Hybrid differences (bootstrap p=0.007) are near the threshold. A full 100-query run would provide more power.

### What These Numbers Mean

**HyDE achieved perfect scores (nDCG@10 = 1.0).** On these 10 SciFact queries, generating a hypothetical answer before retrieval bridged the semantic gap perfectly — every relevant document was ranked at the top. This is partly a small-sample effect (10 queries), but the direction is consistent with HyDE's design thesis.

**Naive Dense RAG came in strong (nDCG@10 = 0.96).** SciFact is single-hop scientific claim verification — exactly the kind of task where SBERT's semantic similarity shines. Dense retrieval is hard to beat on well-formed scientific queries.

**Contextual Hybrid ≈ Hybrid RAG (0.87 vs 0.87).** Adding document context prefixes to chunks before BM25+dense fusion had minimal impact on SciFact. Makes sense — scientific abstracts are already self-contained; the "lost in the middle" problem context enrichment addresses is more relevant for long documents.

**BM25 + Rocchio PRF was weakest (nDCG@10 = 0.70).** Pseudo-relevance feedback with query expansion via TF-IDF terms added noise on these scientific queries. BM25's lexical matching without semantic understanding struggles with the precise terminology of scientific claims.

**Late Interaction couldn't run in this batch** (ColBERT encoder not included in --skip-heavy). The untrained projection head is the #1 quality bottleneck — `make train-colbert` + `--colbert-checkpoint` is the fix.

**RAPTOR pipelines couldn't run in this batch** (also skipped as heavy). They need BART summarizer + UMAP clustering, which requires more time and RAM.

**Latency tells a different story.** BM25 PRF is 260x faster to index (no neural encoding) and 3-6x faster to query. For applications where speed matters more than accuracy, it's a valid choice.

### Historical Benchmark (v0.2.0 — 100 queries, full corpus)

For comparison, here are the v0.2 results on the full SciFact corpus with 100 queries:

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 | Index+Query Time |
|---|---|---|---|---|---|---|
| **Naive Dense RAG** | 0.5100 | 0.6349 | 0.6741 | **0.6964** | 0.6964 | 265.0s |
| **Hybrid RAG (BM25+Dense)** | 0.5100 | 0.6122 | 0.6290 | 0.6668 | 0.6668 | 263.3s |
| **Late Interaction (Flat)** | 0.4900 | 0.5562 | 0.5737 | 0.5801 | 0.5801 | 1181.4s |

> v0.2 significance tests were invalid (per-query scores were all zeros due to a BEIR API bug). v0.3 results above use real per-query nDCG computation.

### Known Limitations

1. **Encoder not trained on retrieval triples (addressable).** The ColBERT projection head is Xavier-initialized by default, which hurts late interaction quality vs fully-trained SBERT. **Fix:** `make train-colbert` (add `--hard-negatives` for BM25-mined hard negatives), then `--colbert-checkpoint checkpoints/final_model.pt` to the benchmark runner. This is the single biggest quality lever for late interaction.

2. **Soft-clustering ≈ hard assignment on short docs.** With short chunks (~100 tokens), the GMM soft assignments tend to converge to near-hard assignments. Measured via `compute_soft_assignment_rate()`. Matches the Stanford CS224N RAPTOR reproduction.

3. **RAPTOR summarizer may use extractive fallback.** If BART fails to load (transformers version issues, memory constraints), falls back to TF-IDF extractive summarization. A `_load_failed` flag prevents the infinite-retry bug; the fallback is logged so tree quality is honestly attributable.

4. **SPLADE is slow on CPU.** Each chunk requires a BERT MLM forward pass (now batched). Use GPU, or accept that indexing takes longer than SBERT.

5. **Late Chunking limited by context window.** Standard BERT caps at 512 tokens, so the "full document" is actually truncated. Needs a long-context embedding model (Jina-embeddings-v2, etc.) to realize the full benefit.

6. **HotpotQA needs >6GB RAM (addressable).** The 5.2M-document corpus exceeds 6GB when encoded with SBERT. **Fix:** `--max-docs 2000` subsamples the corpus while *always preserving judged documents* so every metric stays valid. The unified runner also reuses one SBERT/ColBERT/BART instance across all pipelines (critical on a 4-8GB machine).

7. **Per-query significance requires the numpy implementation.** BEIR's `EvaluateRetrieval.evaluate()` returns corpus-averaged floats, not a per-query dict — using it for per-query scores (the v0.2 bug) silently produces all-zero arrays and meaningless p-values. v0.3 computes per-query nDCG in pure numpy (trec_eval-compatible), pinned by `tests/test_metrics.py`.

---

## Project Structure

```
raven-retrieval/
├── .github/workflows/
│   └── tests.yml               # 3-job CI: core tests (3.10/3.11/3.12 matrix) → full pytest → fast benchmark
├── configs/
│   ├── scifact.yaml             # 5 pipelines, SciFact defaults
│   ├── hotpotqa.yaml            # 4 pipelines, HotpotQA defaults
│   └── full_ablation.yaml       # Full registry, comprehensive
├── experiments/
│   └── preregistration/
│       └── template.md          # Pre-registered hypotheses template
├── src/
│   ├── __init__.py
│   ├── utils.py                 # SHARED: chunking, doc-score aggregation, RRF, timing, masking (kills 8x duplication)
│   ├── config.py                # Central config: 7 nested dataclasses, YAML/JSON, dot-notation overrides, --max-docs/--colbert-checkpoint
│   ├── encoder/
│   │   ├── __init__.py
│   │   └── colbert_encoder.py   # ColbertEncoder + ColbertContrastiveEncoder (contrastive/InfoNCE) + batched masked encoding + checkpoint I/O
│   ├── maxsim/
│   │   ├── __init__.py
│   │   ├── brute_force.py       # maxsim_score, brute_force_rank, maxsim_score_batch, brute_force_rank_fast (1 matmul), pack_doc_embeddings
│   │   ├── approximate.py       # CentroidIndex (FAISS K-means) + ApproximateMaxSim (PLAID-style, vectorized)
│   │   └── compression.py       # ResidualCompressor (30x, vectorized) + CompressedCorpusIndex
│   ├── raptor/
│   │   ├── __init__.py
│   │   ├── chunker.py           # TextChunker: fixed-size word chunks with overlap
│   │   ├── clustering.py        # Two-step UMAP + GMM: global_local_cluster, soft_cluster, BIC selection
│   │   ├── summarizer.py        # LLMSummarizer (BART/T5 + extractive fallback, no infinite retry) + ExtractiveSummarizer (TF-IDF)
│   │   ├── tree.py              # TreeNode (token embeddings + pooled) + RaptorTree (traverse/collapse)
│   │   └── builder.py           # RaptorBuilder: chunk → embed → cluster → summarize → repeat (accepts shared model/summarizer)
│   ├── combined/
│   │   ├── __init__.py
│   │   └── late_raptor.py       # LateInteractionRaptor: novel RAPTOR + ColBERT MaxSim at every node (batched, mask-trimmed, fix)
│   ├── baselines/
│   │   ├── __init__.py          # Lazy imports via __getattr__ (avoids pulling torch when not needed)
│   │   ├── dense.py             # DenseRetriever: SBERT → cosine, aggregate by doc (model injection for tests)
│   │   ├── hybrid.py            # HybridRetriever: BM25 + Dense + RRF
│   │   ├── hyde.py              # HyDERetriever: TinyLlama (actually loads now) or template fallback → embed hypothetical → retrieve
│   │   ├── splade.py            # SPLADERetriever (MLM logits → sparse, inverted index, batched) + HybridSPLADERetriever
│   │   ├── bm25_prf.py          # BM25PRFRetriever: two-stage Rocchio PRF (weighted rerank of candidates only)
│   │   ├── contextual.py        # ContextualChunker + Contextual(Dense|BM25|Hybrid)Retriever
│   │   ├── late_chunking.py     # LateChunkingEncoder (embed full doc → split → pool) + LateChunkingRetriever
│   │   ├── agentic.py           # QueryDecomposer + ReflectionRetriever (real text_lookup!) + MultiHopRetriever
│   │   ├── reranker.py          # CrossEncoderReranker (ms-marco-MiniLM) + TwoStageRetriever
│   │   └── graph_retrieval.py   # DocumentGraph (cosine edges + label propagation, memory-safe blocked) + GraphRetriever
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── datasets.py          # BEIR download/load/subsample + subsample_corpus (judged-doc-preserving)
│   │   ├── metrics.py           # run_beir_evaluation + REAL per-query nDCG/recall/precision/map (numpy, trec_eval-compatible)
│   │   ├── significance.py      # paired_bootstrap_test, paired_t_test, bonferroni_correction, run_all_pairwise_tests
│   │   ├── ablation.py          # AblationRunner + collect_per_query_scores (now returns real values)
│   │   ├── visualize.py         # SVG charts (bar, grouped bar, radar, scatter) + generate_dashboard (HTML)
│   │   └── report.py            # generate_report (Markdown) with CLI entry point
│   ├── training/
│   │   └── train_colbert.py     # generate_triples_from_beir (--hard-negatives via BM25) + train_colbert_encoder (InfoNCE)
│   └── validation/
│       └── cross_check.py       # ReferenceCrossValidator: our ColBERT vs ColBERTv2 (ragatouille), Spearman correlation
├── tests/
│   ├── run_core_tests.py        # Core suite: numpy-only, runs in CI without torch
│   ├── test_utils.py            # Shared utilities (chunking, aggregation, RRF, timer, masking)
│   ├── test_metrics.py          # Per-query nDCG/recall/precision/map pinned against trec_eval semantics
│   ├── test_pipelines_smoke.py  # DI-based contract tests for every retriever (run without torch/transformers)
│   ├── test_maxsim.py           # scoring, ranking, batch, centroid index, approximate, fidelity (faiss-guarded)
│   ├── test_raptor.py           # chunker, UMAP, cluster count, soft cluster, tree ops, assignment rate (umap-guarded)
│   ├── test_significance.py     # bootstrap known/no diff, t-test agreement, Bonferroni, pairwise
│   └── test_integration.py      # pipeline wiring, tree flat retrieval, MaxSim end-to-end
├── run_enhanced_benchmark.py    # THE unified runner: 19-pipeline registry, shared models, honest timing, error capture
├── Makefile                     # make test, test-core, benchmark, benchmark-fast, benchmark-baselines, benchmark-trained, train-colbert, report, clean
├── setup.py                     # pip install raven-retrieval[full,dev] with entry points (v0.3.0)
├── requirements.txt             # torch, transformers, sentence-transformers, faiss-cpu, rank-bm25, umap-learn, scikit-learn, beir, scipy, numpy, tqdm
├── requirements-dev.txt         # pytest, flake8, mypy
├── requirements-validation.txt  # + ragatouille (for reference ColBERT cross-check)
├── README.md                    # This file
├── METHODOLOGY.md               # Research paper methodology: hypotheses, architecture, baselines, evaluation protocol
├── RESEARCH_NOTES.md            # Deep research on 11 topics: ColBERTv2/PLAID, SPLATE, MUVERA, RAPTOR improvements, SPLADE, HyDE, GraphRAG, Agentic RAG, Contextual Retrieval, Late Chunking, Token Pooling
├── BENCHMARK_RESULTS.md         # Actual results + analysis + what went wrong + next steps (honestly revised in v0.3)
└── CONTRIBUTING.md              # How to add a new pipeline, code style, commit messages
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/subhansh-dev/raven-retrieval.git
cd raven-retrieval

# Core only (no torch, runs anywhere)
pip install numpy scipy scikit-learn rank-bm25 umap-learn tqdm

# Full (all pipelines)
pip install -r requirements.txt

# Dev (testing, linting)
pip install -r requirements.txt -r requirements-dev.txt

# Or install as a package
pip install -e ".[full,dev]"
```

### Running Tests

```bash
# Core tests — no torch required, runs in seconds
make test-core
# or: python tests/run_core_tests.py

# Full test suite (needs torch)
make test
# or: python -m pytest tests/ -v
```

### Running Benchmarks

```bash
# Fast benchmark (skip ColBERT/SPLADE/RAPTOR — good for CI or quick checks)
make benchmark-fast
# or: python run_enhanced_benchmark.py --dataset scifact --max-queries 100 --skip-heavy

# Baselines only (dense, hybrid, BM25+PRF, contextual, hyde)
make benchmark-baselines

# Full benchmark (all default pipelines)
make benchmark
# or: python run_enhanced_benchmark.py --dataset scifact --max-queries 100

# Specific pipelines only (from the 19-pipeline registry)
python run_enhanced_benchmark.py --pipelines naive_dense hybrid_rag hyde bm25_prf

# HotpotQA on a low-RAM machine (corpus subsampled to 2000 docs;
# judged docs are ALWAYS preserved so metrics stay valid)
python run_enhanced_benchmark.py --dataset hotpotqa --max-queries 50 --max-docs 2000

# Tune ColBERT encoding batch size for your RAM
python run_enhanced_benchmark.py --dataset scifact --encode-batch-size 8

# List every available pipeline
python run_enhanced_benchmark.py --help   # see the --pipelines choices

# Generate a Markdown report from the latest run
make report
```

> The two old divergent runners (`run_benchmark.py`, `run_full_benchmark.py`)
> were removed in v0.3.0 — one of them mixed single-vector tree embeddings
> with MaxSim scoring and could only crash. Everything routes through
> `run_enhanced_benchmark.py` now (one runner, one shared SBERT/ColBERT/BART
> instance across all pipelines — see "Memory" notes in the runner docstring).

### Training ColBERT

```bash
# Generate triples from BEIR and train (untrained projection is the #1
# reason late interaction underperforms dense — this fixes it)
make train-colbert
# or: python -m src.training.train_colbert --beir-dataset scifact --epochs 3

# Hard-negative mining via BM25 (stronger training signal than random negatives)
python -m src.training.train_colbert --beir-dataset scifact --epochs 5 --hard-negatives

# From pre-generated triples
python -m src.training.train_colbert --triples data/triples.jsonl --epochs 5

# Then benchmark WITH the trained encoder:
make benchmark-trained
# or:
python run_enhanced_benchmark.py --dataset scifact --max-queries 100 \
    --colbert-checkpoint checkpoints/final_model.pt
```

### Generating Reports

```bash
# Auto-generate report from latest benchmark run
make report
# or: python -m src.eval.report experiments/runs/<run_dir>

# Specific run
python -m src.eval.report experiments/runs/enhanced_scifact_1234567890
```

### Using YAML Configs

```bash
# Run with specific config
python run_enhanced_benchmark.py --config configs/scifact.yaml

# Override config values
python run_enhanced_benchmark.py --config configs/full_ablation.yaml --max-queries 50
```

---

## Configuration System

Centralized in `src/config.py`. Six nested dataclasses:

```python
from src.config import ExperimentConfig

# Defaults
config = ExperimentConfig.defaults(dataset="scifact")

# With overrides (dot notation)
config = ExperimentConfig.defaults(
    dataset="hotpotqa",
    max_queries=50,
    **{"encoder.projection_dim": 256, "chunking.chunk_size": 100}
)

# From YAML
config = ExperimentConfig.from_yaml("configs/full_ablation.yaml")

# From JSON
config = ExperimentConfig.from_json("experiment_config.json")

# Validate
config.validate()  # Checks pipeline names, top_k >= 1, etc.
```

Pre-built configs: `SCIFACT_DEFAULTS`, `HOTPOTQA_DEFAULTS`, `FULL_ABLATION`.

---

## CI/CD

Three-job GitHub Actions pipeline in `.github/workflows/tests.yml`:

1. **test-core**: Python 3.10/3.11/3.12 matrix, installs only numpy/scipy/sklearn/rank-bm25/umap/tqdm, runs `tests/run_core_tests.py` (no torch)
2. **test-full**: Python 3.12, installs full requirements, runs `pytest tests/ -v`
3. **benchmark**: Python 3.12, runs fast benchmark on SciFact (20 queries, skip heavy), uploads results as artifact

---

## Key Algorithms In Detail

### Reciprocal Rank Fusion (RRF)

Used to combine rankings from multiple retrieval systems:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where `k` is typically 60. Each document's score is the sum of reciprocal ranks across all input rankings. Simple, effective, no training needed.

### BIC Cluster Count Selection

For choosing how many clusters in GMM:

```
BIC = -2 * log(likelihood) + p * log(n)
```

Where `p` is number of parameters and `n` is number of data points. Penalizes overfitting — BIC prefers simpler models unless the data strongly supports more clusters.

### Label Propagation (Graph Community Detection)

In `src/baselines/graph_retrieval.py`:
1. Initialize each node as its own community
2. Each node adopts the most common community label among its neighbors (weighted by edge strength)
3. Repeat until convergence (or 10 iterations)
4. Groups of nodes with the same label form communities

### TF-IDF Extractive Summarization

Fallback when BART can't load:
1. Split text into sentences
2. Score each sentence: `0.5 * tf_score + 0.3 * position_score + 0.2 * length_score`
3. `tf_score` = average IDF-weighted word frequency across all sentences
4. `position_score` = 1/(1 + i*0.2) (decay for later sentences)
5. `length_score` = min(word_count/15, 1.0)
6. Pick top-k sentences, return in original order

---

## Research References

| Paper | Year | Key Contribution |
|---|---|---|
| ColBERT | 2020 | Late interaction via per-token MaxSim |
| ColBERTv2 | 2022 | Residual compression for storage efficiency |
| PLAID | 2023 | Centroid interaction pruning engine |
| RAPTOR | 2024 | Hierarchical tree retrieval |
| SPLADE | 2021 | Learned sparse retrieval with term expansion |
| HyDE | 2023 | Hypothetical document embeddings |
| Contextual Retrieval | 2024 | LLM-enriched chunk prefixes |
| Late Chunking | 2024 | Post-transformer chunk embeddings |
| SPLATE | 2024 | Sparse candidate generation for ColBERT |
| MUVERA | 2024 | Multi-vector → single-vector via FDEs |
| Agentic RAG | 2025 | Agent-integrated retrieval patterns |
| GraphRAG | 2024 | Graph-based retrieval for global queries |

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for detailed technical analysis of 11 research topics with paper links, implementability assessments, and concrete improvement ideas.

---

## Known Limitations

1. **Encoder not trained on retrieval triples (addressable).** The ColBERT projection head is Xavier-initialized by default, which hurts late interaction quality vs fully-trained SBERT. **Fix:** `make train-colbert` (add `--hard-negatives` for BM21-mined hard negatives), then `--colbert-checkpoint checkpoints/final_model.pt` to the benchmark runner. This is the single biggest quality lever for late interaction.

2. **Soft-clustering ≈ hard assignment on short docs.** With short chunks (~100 tokens), the GMM soft assignments tend to converge to near-hard assignments. Measured via `compute_soft_assignment_rate()`. Matches the Stanford CS224N RAPTOR reproduction.

3. **RAPTOR summarizer may use extractive fallback.** If BART fails to load (transformers version issues, memory constraints), falls back to TF-IDF extractive summarization. A `_load_failed` flag prevents the infinite-retry bug; the fallback is logged so tree quality is honestly attributable.

4. **SPLADE is slow on CPU.** Each chunk requires a BERT MLM forward pass (now batched). Use GPU, or accept that indexing takes longer than SBERT.

5. **Late Chunking limited by context window.** Standard BERT caps at 512 tokens, so the "full document" is actually truncated. Needs a long-context embedding model (Jina-embeddings-v2, etc.) to realize the full benefit.

6. **HotpotQA needs >6GB RAM (addressable).** The 5.2M-document corpus exceeds 6GB when encoded with SBERT. **Fix:** `--max-docs 2000` subsamples the corpus while *always preserving judged documents* so every metric stays valid. The unified runner also reuses one SBERT/ColBERT/BART instance across all pipelines (critical on a 4-8GB machine).

7. **Per-query significance requires the numpy implementation.** BEIR's `EvaluateRetrieval.evaluate()` returns corpus-averaged floats, not a per-query dict — using it for per-query scores (the v0.2 bug) silently produces all-zero arrays and meaningless p-values. v0.3 computes per-query nDCG in pure numpy (trec_eval-compatible), pinned by `tests/test_metrics.py`.

---

## What Would Contradict Expectations

- If RAPTOR + late interaction underperforms hybrid on HotpotQA → tree construction isn't capturing useful hierarchical structure, or late interaction scoring isn't effective on summary nodes
- If late interaction significantly outperforms dense on SciFact → the method has broader applicability than hypothesized (would need trained projection)
- If SPLADE significantly outperforms all dense methods → sparse retrieval is underrated in the current RAG discourse

---

## License

MIT
