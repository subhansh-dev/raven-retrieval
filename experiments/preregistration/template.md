# Pre-Registration: Expected Results

**Date:** [YYYY-MM-DD]
**Run ID:** [auto-generated]
**Dataset(s):** [HotpotQA / SciFact / etc.]

## Expected Outcomes

### Primary Dataset: HotpotQA (Multi-hop)

- **H1:** RAPTOR + late interaction (collapsed) > hybrid RAG on nDCG@10
  - Reasoning: Multi-hop questions benefit from hierarchical summaries that capture cross-passage relationships, and late interaction provides finer-grained token-level matching against those summaries.
- **H2:** RAPTOR + late interaction > RAPTOR with single-vector retrieval on nDCG@10
  - Reasoning: Late interaction's per-token MaxSim should capture more nuanced matches than single-vector cosine, especially for summary nodes that aggregate multiple passages.
- **H3:** Late interaction (flat) > naive dense RAG on nDCG@10
  - Reasoning: ColBERT-style scoring is strictly more expressive than single-vector cosine for the same encoder backbone.

### Secondary Dataset: SciFact (Single-hop control)

- **H4:** RAPTOR + late interaction shows minimal or no improvement over hybrid RAG
  - Reasoning: Single-hop queries don't benefit from hierarchical tree structure; the added complexity may introduce noise without corresponding signal.
- **H5:** Late interaction (flat) performs comparably to hybrid RAG
  - Reasoning: For simple lookups, the BM25 component of hybrid provides strong lexical matching that late interaction may not substantially improve upon.

### Expected Tradeoffs

- **Latency:** Late interaction methods will be slower than dense retrieval. Expected ~3-10x latency increase for brute-force MaxSim vs cosine similarity.
- **Storage:** Per-token embeddings require significantly more memory than single-vector representations.

## What Would Contradict Expectations

- If RAPTOR + late interaction underperforms hybrid RAG on HotpotQA, it would suggest either (a) the tree construction is not capturing useful hierarchical structure, or (b) the late interaction scoring is not effective on summary nodes.
- If late interaction significantly outperforms hybrid RAG on SciFact, it would suggest the method has broader applicability than hypothesized.
