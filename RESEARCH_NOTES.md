# Research Notes

Quick reference for the retrieval techniques implemented or considered in this project.

## ColBERTv2 / PLAID

- **Residual compression**: Store centroid ID + quantized residuals instead of full 128-dim vectors (~30x compression)
- **PLAID pruning**: Centroid interaction for candidate generation, then full MaxSim rerank
- **Key paper**: Santhanam et al., 2022 — "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"
- **Implemented in**: `src/maxsim/compression.py`, `src/maxsim/approximate.py`

## RAPTOR

- Hierarchical tree: chunk → cluster → summarize → re-embed → repeat
- Two-step UMAP (global + local) + GMM with BIC cluster selection
- **Key paper**: Sarthi et al., 2024 — "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- **Implemented in**: `src/raptor/`

## SPLADE

- BERT MLM logits → ReLU + log(1+x) → max over positions → sparse vector
- Activates terms not in original text (semantic expansion)
- **Key paper**: Formal et al., 2021 — "From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Information Retrieval"
- **Implemented in**: `src/baselines/splade.py`

## HyDE

- Generate hypothetical answer → embed that → retrieve real docs
- Bridges query-document semantic gap
- **Key paper**: Gao et al., 2023 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- **Implemented in**: `src/baselines/hyde.py`

### Benchmark Results

- **SciFact (100 queries):** 🏆 nDCG@10 = 0.7119 — best overall. Hypothetical documents genuinely bridge the query-document semantic gap, even with simple templates instead of an LLM. Only ~2 points above Naive Dense, not a game-changer on single-hop claims.
- **HotpotQA (50 queries):** nDCG@10 = 0.8916 — drops to 4th place. A single generated passage can't bridge two separate reasoning steps needed for multi-hop questions.
- **Statistical significance:** Top 4 baselines on SciFact are statistically indistinguishable (p > 0.05 after Bonferroni). HyDE's lead over Dense is not significant.
- **Latency:** 18.6ms per-query on SciFact (fastest neural pipeline). 25.6ms on HotpotQA.

## Contextual Retrieval (Anthropic)

- Prepend document context to each chunk before embedding
- Reported 35-49% reduction in failed retrievals
- **Source**: Anthropic blog, 2024
- **Implemented in**: `src/baselines/contextual.py`

## Late Chunking (Jina)

- Embed entire document first, THEN split token embeddings into chunks
- Preserves cross-chunk context that normal chunking destroys
- Limited by BERT's 512-token window
- **Source**: Jina AI, 2024
- **Implemented in**: `src/baselines/late_chunking.py`

## Agentic RAG

- Query decomposition → retrieve per sub-query → aggregate
- Reflection: retrieve → evaluate sufficiency → reformulate if needed
- **Implemented in**: `src/baselines/agentic.py`

### Benchmark Results

- **Agentic Multi-Hop (SciFact, 100 queries):** nDCG@10 = 0.6783 — slightly behind Graph Retrieval (0.6964). Template-based query decomposition doesn't help on single-hop SciFact claims — these queries don't need decomposition. Statistically indistinguishable from Graph (p=0.0934).
- **Reflection Retriever:** ❌ Failed on SciFact (runtime crash).
- **Two-Stage Dense + Reranker:** ❌ Failed on SciFact (runtime crash).

## GraphRAG

- Build cosine-similarity document graph → label propagation community detection
- Expand results with community members
- **Implemented in**: `src/baselines/graph_retrieval.py`

### Benchmark Results

- **SciFact (100 queries):** nDCG@10 = 0.6964 — matches Naive Dense exactly. Community expansion added zero value on SciFact because scientific abstracts are self-contained, so cosine-sim document graphs don't capture useful thematic structure. 2.4x slower index time (59s vs 25s) buys nothing here.
- **Not tested on HotpotQA** (needs multi-hop corpus structure where community expansion may help).
- **Statistical significance:** Graph vs Agentic Multi-Hop not significantly different (p=0.0934).

## Key Papers

| Paper | Year | Key Idea |
|---|---|---|
| ColBERT | 2020 | Per-token MaxSim late interaction |
| ColBERTv2 | 2022 | Residual compression |
| PLAID | 2023 | Centroid pruning engine |
| RAPTOR | 2024 | Hierarchical tree retrieval |
| SPLADE | 2021 | Learned sparse retrieval |
| HyDE | 2023 | Hypothetical document embeddings |

## Glossary

- **MaxSim**: Σᵢ maxⱼ sim(qᵢ, dⱼ) — ColBERT's scoring function
- **RRF**: Reciprocal Rank Fusion — combining rankings from multiple systems
- **BIC**: Bayesian Information Criterion — penalizes overfitting in cluster count selection
- **UMAP**: Uniform Manifold Approximation — dimensionality reduction before clustering
- **PRF**: Pseudo-Relevance Feedback — query expansion using top retrieved docs
