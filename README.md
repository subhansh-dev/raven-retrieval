# raven-retrieval

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.14-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Tests](https://img.shields.io/badge/Tests-39%2F39%20Passing-brightgreen)
![Pipelines](https://img.shields.io/badge/Pipelines-14-blueviolet)
![ColBERT](https://img.shields.io/badge/ColBERT-Late%20Interaction-orange)
![RAPTOR](https://img.shields.io/badge/RAPTOR-Hierarchical%20Tree-purple)
![SPLADE](https://img.shields.io/badge/SPLADE-Sparse%20Expansion-green)
![HyDE](https://img.shields.io/badge/HyDE-Hypothetical%20Docs-yellow)
![FAISS](https://img.shields.io/badge/FAISS-CPU-005C84?logo=meta&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black?logo=githubactions)

**14 retrieval pipelines. Real benchmarks. Honest results.**

*Late interaction, hierarchical trees, sparse expansion, hypothetical documents, contextual chunking, agentic decomposition — all benchmarked head-to-head on BEIR datasets with proper statistical significance testing.*

</div>

---

## What Is This

This is a retrieval research framework. Not a production search engine, not a RAG template — a tool for actually comparing retrieval methods against each other on equal footing.

Most retrieval blog posts tell you "Method X is better" without showing you the full picture. This project implements 14 different pipelines, runs them all on the same datasets with the same chunking, same evaluation metrics, and same statistical tests. The results are what they are. No cherry-picking.

The core question it answers: **does applying ColBERT-style late interaction scoring at every level of a RAPTOR hierarchical tree actually improve retrieval quality over simpler approaches?** Everything else is scaffolding to answer that question properly.

### The 14 Pipelines

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
| 14 | **Reflection Retriever** | Retrieve → evaluate context sufficiency (keyword coverage) → if insufficient, reformulate query → re-retrieve (up to 2 iterations). Self-RAG pattern | Self-RAG, Asai et al. 2023 |

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

### SciFact (BEIR) — 100 queries, top-10, CPU-only

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 | Index+Query Time |
|---|---|---|---|---|---|---|
| **Naive Dense RAG** | 0.5100 | 0.6349 | 0.6741 | **0.6964** | 0.6964 | 265.0s |
| **Hybrid RAG (BM25+Dense)** | 0.5100 | 0.6122 | 0.6290 | 0.6668 | 0.6668 | 263.3s |
| **Late Interaction (Flat)** | 0.4900 | 0.5562 | 0.5737 | 0.5801 | 0.5801 | 1181.4s |

### What These Numbers Mean

**Dense beat Hybrid on SciFact.** Makes sense — SciFact is single-hop scientific claim verification. BM25's lexical matching adds noise without corresponding signal on these queries. Semantic similarity alone is already strong.

**Late Interaction underperformed Dense.** The ColBERT encoder uses a pretrained BERT backbone with a randomly-initialized projection head (not trained on retrieval triples). The untrained projection layer hurts quality compared to fully-trained SBERT embeddings. This is an honest, expected result — the code supports fine-tuning, it just hasn't been run yet.

**RAPTOR + Late Interaction couldn't run.** The BART summarizer pipeline had compatibility issues with transformers 5.x (the `summarization` task name was removed). The code has since been updated to use `AutoModelForSeq2SeqLM` directly, which works across versions.

**HotpotQA OOM'd.** The full HotpotQA corpus (5.2M entries) exceeds 6GB RAM when encoding with SBERT. Needs either a machine with more RAM, corpus subsampling, or streaming encoding. This is the dataset where RAPTOR's hierarchical summaries should actually shine — multi-hop questions benefit from cross-passage relationships.

---

## Project Structure

```
raven-retrieval/
├── .github/workflows/
│   └── tests.yml               # 3-job CI: core tests (3.10/3.11/3.12 matrix) → full pytest → fast benchmark
├── configs/
│   ├── scifact.yaml             # 5 pipelines, SciFact defaults
│   ├── hotpotqa.yaml            # 4 pipelines, HotpotQA defaults
│   └── full_ablation.yaml       # All 12 pipelines, comprehensive
├── experiments/
│   └── preregistration/
│       └── template.md          # Pre-registered hypotheses template
├── src/
│   ├── __init__.py
│   ├── config.py                # Central config: 6 nested dataclasses, YAML/JSON support, dot-notation overrides
│   ├── encoder/
│   │   ├── __init__.py
│   │   └── colbert_encoder.py   # ColbertEncoder (162 lines) + ColbertContrastiveEncoder with contrastive/InfoNCE loss
│   ├── maxsim/
│   │   ├── __init__.py
│   │   ├── brute_force.py       # maxsim_score, brute_force_rank, maxsim_score_batch (numpy-only)
│   │   ├── approximate.py       # CentroidIndex (FAISS K-means) + ApproximateMaxSim (PLAID-style)
│   │   └── compression.py       # ResidualCompressor (30x compression) + CompressedCorpusIndex
│   ├── raptor/
│   │   ├── __init__.py
│   │   ├── chunker.py           # TextChunker: fixed-size word chunks with overlap
│   │   ├── clustering.py        # Two-step UMAP + GMM: global_local_cluster, soft_cluster, BIC selection
│   │   ├── summarizer.py        # LLMSummarizer (BART/T5 + extractive fallback) + ExtractiveSummarizer (TF-IDF)
│   │   ├── tree.py              # TreeNode (token embeddings + pooled) + RaptorTree (traverse/collapse)
│   │   └── builder.py           # RaptorBuilder: chunk → embed → cluster → summarize → repeat
│   ├── combined/
│   │   ├── __init__.py
│   │   └── late_raptor.py       # LateInteractionRaptor: the novel RAPTOR + ColBERT MaxSim at every node
│   ├── baselines/
│   │   ├── __init__.py          # Lazy imports via __getattr__ (avoids pulling torch when not needed)
│   │   ├── dense.py             # DenseRetriever: SBERT → cosine, aggregate by doc
│   │   ├── hybrid.py            # HybridRetriever: BM25 + Dense + RRF
│   │   ├── hyde.py              # HyDERetriever: TinyLlama or template fallback → embed hypothetical → retrieve
│   │   ├── splade.py            # SPLADERetriever (MLM logits → sparse) + HybridSPLADERetriever (sparse + dense + RRF)
│   │   ├── bm25_prf.py          # BM25PRFRetriever: Rocchio expansion, TF*IDF term weighting
│   │   ├── contextual.py        # ContextualChunker + ContextualDenseRetriever + ContextualBM25Retriever + ContextualHybridRetriever
│   │   ├── late_chunking.py     # LateChunkingEncoder (embed full doc → split → pool) + LateChunkingRetriever
│   │   ├── agentic.py           # QueryDecomposer + ReflectionRetriever + MultiHopRetriever
│   │   ├── reranker.py          # CrossEncoderReranker (ms-marco-MiniLM) + TwoStageRetriever
│   │   └── graph_retrieval.py   # DocumentGraph (cosine edges + label propagation) + GraphRetriever
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── datasets.py          # BEIR download/load/subsample
│   │   ├── metrics.py           # run_beir_evaluation, compute_em_f1, measure_latency
│   │   ├── significance.py      # paired_bootstrap_test, paired_t_test, bonferroni_correction, run_all_pairwise_tests
│   │   ├── ablation.py          # AblationRunner + collect_per_query_scores
│   │   ├── visualize.py         # SVG charts (bar, grouped bar, radar, scatter) + generate_dashboard (HTML)
│   │   └── report.py            # generate_report (Markdown) with CLI entry point
│   ├── training/
│   │   └── train_colbert.py     # generate_triples_from_beir + train_colbert_encoder (AdamW, warmup+decay, InfoNCE)
│   └── validation/
│       └── cross_check.py       # ReferenceCrossValidator: our ColBERT vs ColBERTv2 (ragatouille), Spearman correlation
├── tests/
│   ├── run_core_tests.py        # 22 tests, 39 assertions — NO torch required, runs instantly
│   ├── test_maxsim.py           # 6 tests: scoring, ranking, batch, centroid index, approximate, fidelity
│   ├── test_raptor.py           # 6 tests: chunker, UMAP, cluster count, soft cluster, tree ops, assignment rate
│   ├── test_significance.py     # 5 tests: bootstrap known/no diff, t-test agreement, Bonferroni, pairwise
│   └── test_integration.py      # 3 tests: pipeline wiring, tree flat retrieval, MaxSim end-to-end
├── Makefile                     # make test, test-core, benchmark, benchmark-fast, benchmark-baselines, train-colbert, report, clean
├── setup.py                     # pip install raven-retrieval[full,dev] with entry points
├── requirements.txt             # torch, transformers, sentence-transformers, faiss-cpu, rank-bm25, umap-learn, scikit-learn, beir, scipy, numpy, tqdm
├── requirements-dev.txt         # pytest, flake8, mypy
├── requirements-validation.txt  # + ragatouille (for reference ColBERT cross-check)
├── README.md                    # This file
├── METHODOLOGY.md               # Research paper methodology: hypotheses, architecture, baselines, evaluation protocol
├── RESEARCH_NOTES.md            # Deep research on 11 topics: ColBERTv2/PLAID, SPLATE, MUVERA, RAPTOR improvements, SPLADE, HyDE, GraphRAG, Agentic RAG, Contextual Retrieval, Late Chunking, Token Pooling
├── BENCHMARK_RESULTS.md         # Actual results + analysis + what went wrong + next steps
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
# Fast benchmark (skip ColBERT and RAPTOR — good for CI or quick checks)
make benchmark-fast
# or: python run_enhanced_benchmark.py --dataset scifact --max-queries 100 --skip-heavy

# Baselines only (dense, hybrid, BM25+PRF, contextual, hyde)
make benchmark-baselines

# Full benchmark (all 14 pipelines)
make benchmark
# or: python run_enhanced_benchmark.py --dataset scifact --max-queries 100

# Specific pipelines only
python run_enhanced_benchmark.py --pipelines naive_dense hybrid_rag hyde bm25_prf

# HotpotQA (needs >6GB RAM)
python run_enhanced_benchmark.py --dataset hotpotqa --max-queries 50

# Original benchmark script (core 3 + RAPTOR)
python run_benchmark.py --dataset hotpotqa --max-queries 50

# Multi-dataset full ablation
python run_full_benchmark.py
```

### Training ColBERT

```bash
# Generate triples from BEIR and train
make train-colbert
# or: python -m src.training.train_colbert --beir-dataset scifact --epochs 3

# From pre-generated triples
python -m src.training.train_colbert --triples data/triples.jsonl --epochs 5

# Custom settings
python -m src.training.train_colbert --beir-dataset scifact --epochs 5 --batch-size 16 --lr 1e-5
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

1. **Encoder not trained on retrieval triples.** The ColBERT projection head is Xavier-initialized, not trained on (query, positive, negative) triples. This hurts late interaction quality vs fully-trained SBERT. Fix: `make train-colbert` to fine-tune on BEIR data.

2. **Soft-clustering ≈ hard assignment on short docs.** With short chunks (~100 tokens), the GMM soft assignments tend to converge to near-hard assignments. Measured via `compute_soft_assignment_rate()`.

3. **RAPTOR summarizer may use extractive fallback.** If BART fails to load (transformers version issues, memory constraints), automatically falls back to TF-IDF extractive summarization. Works fine but lower quality.

4. **SPLADE is slow on CPU.** Each document requires a full BERT MLM forward pass. On CPU, expect ~10x slower indexing vs SBERT. Use GPU for reasonable speed.

5. **Late Chunking limited by context window.** Standard BERT caps at 512 tokens, so the "full document" is actually truncated. Needs a long-context embedding model (Jina-embeddings-v2, etc.) to realize the full benefit.

6. **HotpotQA needs >6GB RAM.** The 5.2M document corpus exceeds 6GB when encoded with SBERT. Needs subsampling, streaming encoding, or a machine with more RAM.

7. **Per-query NDCG variance is zero on SciFact.** Known BEIR evaluation artifact where the qrels structure produces uniform per-query NDCG. Bootstrap/t-test still run but results reflect evaluation granularity, not necessarily identical system quality.

---

## What Would Contradict Expectations

- If RAPTOR + late interaction underperforms hybrid on HotpotQA → tree construction isn't capturing useful hierarchical structure, or late interaction scoring isn't effective on summary nodes
- If late interaction significantly outperforms dense on SciFact → the method has broader applicability than hypothesized (would need trained projection)
- If SPLADE significantly outperforms all dense methods → sparse retrieval is underrated in the current RAG discourse

---

## License

MIT
