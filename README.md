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

A comprehensive retrieval research framework that implements and benchmarks **14 retrieval pipelines** from simple baselines to cutting-edge 2024-2025 research. Built for researchers who want to compare retrieval methods rigorously, not just read blog posts about them.

### Core Pipelines

| # | Pipeline | What It Does | Reference |
|---|---|---|---|
| 1 | **Naive Dense RAG** | Fixed-size chunks → SBERT → cosine similarity | Standard baseline |
| 2 | **Hybrid RAG** | BM25 + Dense, fused with Reciprocal Rank Fusion (k=60) | Robertson et al. |
| 3 | **ColBERT Late Interaction** | Per-token BERT embeddings → MaxSim scoring | Khattab & Zaharia, SIGIR 2020 |
| 4 | **RAPTOR + Late Interaction** | Hierarchical tree with MaxSim at every node | Sarthi et al., ICLR 2024 |

### Research Pipelines (2024-2025)

| # | Pipeline | What It Does | Reference |
|---|---|---|---|
| 5 | **HyDE** | Generate hypothetical answer → embed → retrieve | Gao et al., ACL 2023 |
| 6 | **SPLADE** | Learned sparse retrieval with term expansion via MLM logits | Formal et al., SIGIR 2021 |
| 7 | **SPLADE + Dense Hybrid** | Sparse expansion + dense embeddings + RRF fusion | Hybrid approach |
| 8 | **BM25 + Rocchio PRF** | BM25 with pseudo-relevance feedback query expansion | Rocchio, 1971 |
| 9 | **Contextual Retrieval** | LLM-enriched chunk prefixes before embedding | Anthropic, 2024 |
| 10 | **Contextual Hybrid** | Contextual BM25 + Contextual Dense + RRF | Extended Anthropic |
| 11 | **Late Chunking** | Embed full doc first, chunk the embeddings after transformer | Jina AI, 2024 |
| 12 | **RAPTOR + Late Traversal** | Top-down tree traversal with MaxSim at each level | Novel combination |
| 13 | **Agentic Multi-Hop** | Query decomposition → retrieve per sub-query → aggregate | Agentic RAG, 2025 |
| 14 | **Reflection Retriever** | Evaluate context sufficiency → reformulate → re-retrieve | Self-RAG pattern |

### Composition Modules

| Module | What It Does |
|---|---|
| **Cross-Encoder Reranker** | Second-stage reranking with cross-attention (ms-marco-MiniLM) |
| **Two-Stage Retriever** | Bi-encoder (fast, top-100) → Cross-encoder (accurate, top-10) |
| **Document Graph** | Community detection for thematic retrieval |
| **Residual Compression** | ColBERTv2-style 30x storage compression |

---

## Architecture

```
Query
  │
  ├─ [Optional] Query Decomposition (Agentic RAG)
  │   └─ Complex query → sub-queries → retrieve each → aggregate
  │
  ├─ [Optional] HyDE (Hypothetical Documents)
  │   └─ Generate hypothetical answer → embed that → retrieve
  │
  ├─ Encode with BERT → per-token embeddings (128-dim projection)
  │
  ├─ RAPTOR Tree (built offline)
  │   ├─ Level 0: corpus chunks (~100 tokens each)
  │   ├─ Level 1: UMAP + GMM clusters → LLM/extractive summaries
  │   ├─ Level 2: re-cluster summaries → meta-summaries
  │   └─ ... until clustering can't subdivide further
  │
  ├─ Retrieval strategies:
  │   ├─ Dense: SBERT cosine similarity
  │   ├─ Hybrid: BM25 + Dense + RRF
  │   ├─ SPLADE: Learned sparse with term expansion
  │   ├─ BM25+PRF: Rocchio pseudo-relevance feedback
  │   ├─ ColBERT: Per-token MaxSim (flat or tree)
  │   ├─ Contextual: Context-enriched chunk embeddings
  │   ├─ Late Chunking: Full-doc embeddings, post-transformer chunking
  │   └─ Collapsed/Traversal: RAPTOR tree retrieval strategies
  │
  ├─ [Optional] Cross-Encoder Reranking
  │   └─ Bi-encoder top-100 → Cross-encoder top-10
  │
  └─ [Optional] Reflection Loop
      └─ Evaluate context → reformulate → re-retrieve if insufficient
```

---

## Results

### SciFact (BEIR) — 100 queries, top-10, official `pytrec_eval` scoring

| Pipeline | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|---|
| Naive Dense RAG | 0.5100 | 0.6349 | 0.6741 | **0.6964** | 0.6964 |
| Hybrid RAG (BM25+Dense) | 0.5100 | 0.6122 | 0.6290 | 0.6668 | 0.6668 |
| Late Interaction (Flat) | 0.4900 | 0.5562 | 0.5737 | 0.5801 | 0.5801 |

*New pipelines (HyDE, SPLADE, BM25+PRF, Contextual, Late Chunking, Agentic) ready to benchmark — run `make benchmark`*

---

## Encoder Details

**Backbone:** `bert-base-uncased` (pretrained)
**Projection:** `Linear(768 → 128)`, Xavier-initialized
**Training support:** `ColbertContrastiveEncoder` with:
- Contrastive loss for (query, positive, negative) triples
- InfoNCE loss with in-batch negatives
- Learnable temperature parameter
- Pooled encoding (mean/max/cls) for single-vector fallback
- Flash Attention 2 support when hardware allows

---

## Clustering Details

**Two-step UMAP + GMM** (matching RAPTOR's paper):
- Global step: `n_neighbors ≈ sqrt(N-1)`, preserves corpus-wide structure
- Local step: `n_neighbors ≈ 10`, captures fine-grained structure
- Cluster count: selected via BIC (penalizes overfitting)
- Soft assignment threshold: 0.1

---

## Summarization

**Primary:** Abstractive via `AutoModelForSeq2SeqLM` (BART, T5, etc.)
- Compatible with transformers 4.x AND 5.x (no deprecated pipeline API)
- Beam search with length penalty

**Fallback:** Extractive summarization (no model required)
- TF-IDF sentence scoring with position weighting
- Zero dependencies — works on any machine

---

## Getting Started

```bash
git clone https://github.com/subhansh-dev/raven-retrieval.git
cd raven-retrieval
pip install -r requirements.txt

# Run core tests (no torch required, instant)
make test-core

# Run fast benchmark (skip heavy pipelines)
make benchmark-fast

# Run all pipelines
make benchmark

# Train ColBERT encoder
make train-colbert

# Generate report from latest run
make report
```

### Using Configs

```bash
# Run with specific config
python run_enhanced_benchmark.py --config configs/scifact.yaml

# Override config values
python run_enhanced_benchmark.py --config configs/full_ablation.yaml --max-queries 50
```

---

## Project Structure

```
raven-retrieval/
├── .github/workflows/          # CI/CD (GitHub Actions)
├── configs/                    # YAML experiment configs
│   ├── scifact.yaml            # SciFact defaults
│   ├── hotpotqa.yaml           # HotpotQA defaults
│   └── full_ablation.yaml      # All 12 pipelines
├── src/
│   ├── config.py               # Experiment configuration system
│   ├── encoder/
│   │   └── colbert_encoder.py  # ColBERT + contrastive training
│   ├── maxsim/
│   │   ├── brute_force.py      # MaxSim scoring (numpy/torch)
│   │   ├── approximate.py      # PLAID-style centroid pruning
│   │   └── compression.py      # ColBERTv2 residual compression
│   ├── raptor/
│   │   ├── chunker.py          # Text chunking
│   │   ├── clustering.py       # UMAP + GMM two-step clustering
│   │   ├── summarizer.py       # Abstractive + extractive fallback
│   │   ├── tree.py             # RAPTOR tree data structure
│   │   └── builder.py          # Tree construction
│   ├── combined/
│   │   └── late_raptor.py      # RAPTOR + Late Interaction (novel)
│   ├── baselines/
│   │   ├── dense.py            # Naive Dense RAG
│   │   ├── hybrid.py           # BM25 + Dense + RRF
│   │   ├── hyde.py             # HyDE
│   │   ├── splade.py           # SPLADE + SPLADE-Dense Hybrid
│   │   ├── bm25_prf.py         # BM25 + Rocchio PRF
│   │   ├── contextual.py       # Contextual Retrieval
│   │   ├── late_chunking.py    # Late Chunking
│   │   ├── agentic.py          # Query Decomposition + Reflection
│   │   ├── reranker.py         # Cross-Encoder Reranker
│   │   └── graph_retrieval.py  # GraphRAG-lite
│   ├── eval/
│   │   ├── datasets.py         # BEIR data loading
│   │   ├── metrics.py          # nDCG, MAP, Recall, Precision
│   │   ├── significance.py     # Bootstrap + t-test + Bonferroni
│   │   ├── ablation.py         # Ablation runner
│   │   ├── visualize.py        # HTML dashboard with SVG charts
│   │   └── report.py           # Auto Markdown report generator
│   ├── training/
│   │   └── train_colbert.py    # ColBERT contrastive training
│   └── validation/
│       └── cross_check.py      # Spearman correlation
├── tests/
│   ├── run_core_tests.py       # 22 tests, 39 assertions (no torch!)
│   ├── test_maxsim.py
│   ├── test_raptor.py
│   ├── test_significance.py
│   └── test_integration.py
├── experiments/
│   └── preregistration/
│       └── template.md         # Pre-registered hypotheses
├── Makefile                    # make test, benchmark, train, report
├── setup.py                    # pip install .[full,dev]
├── README.md                   # This file
├── METHODOLOGY.md              # Research paper methodology
├── CONTRIBUTING.md             # Contribution guide
├── RESEARCH_NOTES.md           # Deep research (10 topics)
└── BENCHMARK_RESULTS.md        # Benchmark analysis
```

---

## Evaluation

- **Metrics:** nDCG@k, MAP@k, Recall@k, Precision@k at k ∈ {1, 3, 5, 10, 100}
- **Significance:** Paired bootstrap (10,000 resamples) + paired t-test
- **Multiple comparisons:** Bonferroni correction (6 pairwise across 4 pipelines)
- **Confidence intervals:** 95% CI from bootstrap resamples
- **Latency:** Wall-clock per query, same hardware
- **Dashboard:** Auto-generated HTML with SVG charts (bar, radar, scatter)
- **Reports:** Auto-generated Markdown with tables and analysis

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

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for detailed technical analysis of each.

---

## Known Limitations

1. **Encoder not trained on retrieval triples.** Uses pretrained BERT with Xavier-initialized projection. Fine-tuning supported via `ColbertContrastiveEncoder`.
2. **Soft-clustering ≈ hard assignment on short docs.** Measured and reported.
3. **RAPTOR summarizer may use extractive fallback** if BART fails to load. Automatic.
4. **SPLADE is slow on CPU.** Each document requires BERT MLM forward pass. Use GPU for reasonable speed.
5. **Late Chunking requires long-context model.** Standard BERT limits to 512 tokens.

---

## License

MIT
