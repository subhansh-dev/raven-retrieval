# Pre-Registration: Expected Results

**Date:** [YYYY-MM-DD]
**Run ID:** [auto-generated]
**Dataset(s):** [HotpotQA / SciFact / etc.]

## Hypotheses

### HotpotQA (Multi-hop)

- **H1:** RAPTOR + late interaction (collapsed) > hybrid RAG on nDCG@10
  - Multi-hop questions should benefit from hierarchical summaries + token-level matching
- **H2:** RAPTOR + late interaction > RAPTOR single-vector on nDCG@10
  - MaxSim is more expressive than cosine for matching against summary nodes
- **H3:** Late interaction (flat) > naive dense on nDCG@10
  - Only with a trained projection head — untrained won't beat SBERT

### SciFact (Single-hop control)

- **H4:** RAPTOR + late interaction ≈ hybrid RAG (minimal difference)
  - Single-hop doesn't need hierarchical structure
- **H5:** Untrained late interaction < dense on SciFact
  - The projection head isn't trained, so it'll underperform SBERT

## Expected Tradeoffs

- Late interaction: ~3-10x slower than dense (brute-force MaxSim)
- Per-token embeddings use significantly more memory than single-vector

## What Would Be Surprising

- RAPTOR + late interaction badly underperforming hybrid on HotpotQA → tree construction isn't working
- Late interaction crushing dense on SciFact without training → broader applicability than expected
