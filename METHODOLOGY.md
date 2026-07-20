# Methodology

## 1. Problem Statement

Current retrieval-augmented generation (RAG) systems typically use one of two approaches:

1. **Single-vector retrieval**: Encode documents and queries into fixed-dimensional vectors, retrieve via cosine similarity. Fast but loses token-level information.

2. **Late interaction retrieval**: Encode documents and queries as sequences of per-token vectors, score via MaxSim (ColBERT). More expressive but expensive.

Neither approach fully leverages hierarchical document structure. RAPTOR (Sarthi et al., 2024) builds recursive summary trees but retrieves with single-vector similarity. We hypothesize that applying late interaction scoring at every level of a RAPTOR tree will improve retrieval quality, especially for multi-hop queries.

## 2. System Architecture

### 2.1 RAPTOR Tree Construction

Documents are chunked into ~100-token segments (Level 0). At each subsequent level:

1. **Dimensionality reduction**: UMAP reduces chunk embeddings to 10 dimensions
2. **Clustering**: Gaussian Mixture Model (GMM) with BIC-selected cluster count
3. **Soft assignment**: Nodes with P(cluster) > 0.1 are assigned to multiple clusters
4. **Summarization**: Each cluster's texts are summarized (abstractive or extractive)
5. **Re-embedding**: Summary nodes are embedded and re-clustered

This continues until clusters have fewer than `min_cluster_size` members.

**Two-step UMAP** (matching RAPTOR's paper):
- Global: `n_neighbors ≈ sqrt(N-1)` — preserves corpus-wide structure
- Local: `n_neighbors ≈ 10` — captures fine-grained structure

### 2.2 ColBERT Encoder

**Backbone**: `bert-base-uncased` (pretrained, 768-dim)
**Projection**: `Linear(768 → 128)`, Xavier-initialized
**Output**: L2-normalized per-token embeddings

The encoder produces a matrix of token embeddings for each text. These are stored at every tree node (leaf chunks AND summary nodes).

**Important**: The projection head is NOT trained on retrieval triples in the baseline experiments. This is a known limitation — the encoder uses pretrained BERT representations with a random projection, not a retrieval-trained model. Results reflect this.

### 2.3 MaxSim Scoring

For query Q = [q₁, ..., qₘ] and document D = [d₁, ..., dₙ]:

```
MaxSim(Q, D) = Σᵢ maxⱼ sim(qᵢ, dⱼ)
```

Where sim is cosine similarity. For each query token, we find its best match in the document, then sum across all query tokens.

### 2.4 Retrieval Strategies

**Collapsed**: Flatten all tree nodes into a single pool, MaxSim top-k.

**Traversal**: Start at root nodes, MaxSim to select top-k, descend to their children, repeat until leaves.

## 3. Baselines

| Pipeline | Method | Components |
|---|---|---|
| Naive Dense | SBERT + cosine | `all-MiniLM-L6-v2`, 200-token chunks, 50-token overlap |
| Hybrid | BM25 + Dense + RRF | Reciprocal Rank Fusion (k=60) |
| HyDE | Hypothetical docs + dense | Template-based hypothetical document generation |
| SPLADE | Learned sparse | BERT MLM logits, ReLU + log(1+x) activation |
| BM25+PRF | BM25 + pseudo-relevance feedback | Rocchio expansion, top-5 docs, 10 expansion terms |
| Contextual | Context-enriched chunks | Document title/first-sentence prefix on each chunk |
| Late Chunking | Post-transformer chunking | Full document encoding, then chunk token embeddings |

## 4. Evaluation Protocol

### 4.1 Datasets

- **SciFact** (BEIR): 5,183 documents, 300 queries — single-hop scientific claim verification
- **HotpotQA** (BEIR): 5.2M documents, 7,405 queries — multi-hop question answering

### 4.2 Metrics

- **nDCG@k**: Normalized Discounted Cumulative Gain at k ∈ {1, 3, 5, 10, 100}
- **Recall@k**: Fraction of relevant documents retrieved
- **MAP@k**: Mean Average Precision
- **Precision@k**: Fraction of retrieved documents that are relevant

All computed via BEIR's `EvaluateRetrieval` wrapping `pytrec_eval`.

### 4.3 Statistical Significance

- **Paired bootstrap test**: 10,000 resamples per comparison
- **Paired t-test**: Cross-check with parametric test
- **Bonferroni correction**: Adjusted α for all C(n,2) pairwise comparisons
- **95% confidence intervals**: From bootstrap distribution

Per-query nDCG is computed by a pure-numpy implementation (trec_eval
convention: linear gain, log2(rank+1) discount) in `src/eval/metrics.py`.
This is required because BEIR's `EvaluateRetrieval.evaluate()` returns
**corpus-averaged** floats, not a per-query dict — using it for per-query
scores (the original implementation) silently produces all-zero arrays and
meaningless p-values. The numpy implementation is pinned against
hand-computed values in `tests/test_metrics.py`.

### 4.4 Pre-registration

Expectations are written BEFORE the final benchmark run (see `experiments/preregistration/template.md`). Results are compared against stated expectations, including where the system was wrong.

## 5. Expected Results

### Hypotheses (Pre-registered)

**H1**: RAPTOR + late interaction (collapsed) > hybrid RAG on nDCG@10 for HotpotQA.
*Reasoning*: Multi-hop questions benefit from hierarchical summaries + fine-grained token matching.

**H2**: RAPTOR + late interaction > RAPTOR single-vector on nDCG@10.
*Reasoning*: MaxSim is strictly more expressive than cosine similarity.

**H3**: Late interaction (flat) > naive dense RAG on nDCG@10.
*Reasoning*: Per-token matching captures more nuance than single-vector.

**H4**: RAPTOR + late interaction shows minimal improvement on SciFact.
*Reasoning*: Single-hop queries don't benefit from hierarchical structure.

**H5**: Late interaction underperforms dense on SciFact.
*Reasoning*: Untrained projection head hurts quality vs. fully-trained SBERT.

### What Would Contradict Expectations

- If RAPTOR + late interaction underperforms hybrid on HotpotQA → tree construction is not useful
- If late interaction outperforms dense on SciFact → the method has broader applicability than expected
- If SPLADE significantly outperforms all dense methods → sparse retrieval is underrated

## 6. Known Limitations

1. **Encoder not trained** (addressable): The projection head starts Xavier-initialized. Pass `--colbert-checkpoint checkpoints/final_model.pt` to the benchmark runner (after `make train-colbert`, which supports `--hard-negatives` for stronger training signal). Without a trained checkpoint, H5 (late interaction < dense) is expected to hold.
2. **Soft-clustering ≈ hard assignment on short docs**: Per Stanford CS224N reproduction. Measured via `compute_soft_assignment_rate()`.
3. **CPU-only benchmarking**: ColBERT encoding is ~20x slower without GPU. Batched encoding (`--encode-batch-size`) mitigates this; `--max-docs` enables HotpotQA on small machines by subsampling the corpus (judged docs always preserved so metrics stay valid).
4. **Summarizer may degrade to extractive**: If BART fails to load (transformers 5.x compatibility, memory), the summarizer falls back to TF-IDF extractive summarization. A `_load_failed` flag prevents the infinite-retry bug; the fallback is logged so tree quality is honestly attributable.
5. **Single dataset per run**: Cross-dataset generalization not yet measured; use `run_full` shells or run the runner twice with different `--dataset`.

## 7. Reproducibility

- All random seeds fixed (numpy: 42, torch: 42)
- Package versions logged in `metadata.json`
- Hardware profile recorded
- Pre-registered expectations documented before runs
- Code and data available at: `https://github.com/subhansh-dev/raven-retrieval`
