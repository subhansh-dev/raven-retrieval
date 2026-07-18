# Neural Information Retrieval & RAG Systems — Deep Research Notes

*Last updated: 2026-07-18*

---

## Table of Contents

1. [ColBERTv2 / PLAID Improvements](#1-colbertv2--plaid-improvements)
2. [Late Interaction Models Beyond ColBERT](#2-late-interaction-models-beyond-colbert)
3. [RAPTOR Improvements](#3-raptor-improvements)
4. [SPLADE & Sparse-Dense Hybrid Approaches](#4-splade--sparse-dense-hybrid-approaches)
5. [HyDE (Hypothetical Document Embeddings)](#5-hyde-hypothetical-document-embeddings)
6. [GraphRAG](#6-graphrag)
7. [Agentic RAG](#7-agentic-rag)
8. [Contextual Retrieval (Anthropic)](#8-contextual-retrieval-anthropic)
9. [Late Chunking (Jina)](#9-late-chunking-jina)
10. [ColBERT Token Pooling Strategies](#10-colbert-token-pooling-strategies)
11. [Concrete Ideas for raven-retrieval](#11-concrete-ideas-for-raven-retrieval)

---

## 1. ColBERTv2 / PLAID Improvements

### Core Architecture

**ColBERTv2** (Santhanam et al., 2022) introduced **residual compression** to address the main drawback of ColBERT — massive storage requirements from storing per-token embeddings. The key ideas:

- **Residual compression**: Instead of storing full 128-dim token embeddings, store residuals (differences) relative to a set of learned centroids. Each token embedding is mapped to its nearest centroid, and only the residual vector is stored (compressed via quantization).
- **Centroid interaction**: During candidate generation, PLAID computes similarity between query tokens and centroids first (cheap), then only decompresses full embeddings for documents whose centroids score highly.
- **Centroid pruning**: A mechanism to skip documents whose centroid interactions are below a threshold — dramatically reduces the number of full MaxSim computations needed.

**PLAID Engine** (ACM SIGMOD 2023): The production engine that operationalizes these ideas.
- Pipeline: centroid scoring → pruning → residual decompression → full MaxSim reranking
- Achieves sub-10ms query latency on CPU for top-10 retrieval
- Key insight: centroid interaction acts as a cheap upper-bound proxy for full document relevance

### What's New Since 2022

**SPLATE** (SIGIR 2024 — Formal, Clinchant, Déjean, Lassance):
- Paper: https://arxiv.org/abs/2404.13950
- Replaces PLAID's centroid-based candidate generation with **sparse retrieval** (SPLADE-style)
- Learns an "MLM adapter" that maps ColBERTv2's frozen token embeddings to sparse vocabulary space
- Candidate generation becomes a standard inverted index lookup — runs entirely on CPU
- Achieves same effectiveness as PLAID ColBERTv2 by reranking 50 documents, retrieved in <10ms
- **Key advantage**: No need for centroid computation or product quantization infrastructure

**EMVB** (ECIR 2024 — Nardini, Rulli, Venturini):
- Paper: https://arxiv.org/abs/2404.02805
- Code: https://github.com/cosimorulli/emvb
- Uses bit vectors for fast candidate filtering in multi-vector retrieval
- Product Quantization (PQ) for memory reduction
- Three-stage pipeline: bit-vector filtering → PQ-based scoring → full MaxSim reranking

**MUVERA** (NeurIPS 2024 — Google):
- Paper: https://arxiv.org/abs/2405.19504
- Converts multi-vector retrieval into **single-vector MIPS** via Fixed Dimensional Encodings (FDEs)
- Provably preserves the Chamfer similarity (ColBERT's MaxSim generalization)
- Enables use of standard ANN indexes (FAISS, ScaNN) for multi-vector models
- **Major implication**: Could make ColBERT-style retrieval as infrastructure-light as dense retrieval

**Enhancing ColBERT (PACLIC 2024)**:
- Paper: https://aclanthology.org/2024.paclic-1.79.pdf
- Introduces a small network that assigns importance weights to tokens for pruning
- Reduces the number of stored embeddings while preserving quality

### Implementability

| Technique | Compute Needed | Implementable? |
|-----------|---------------|----------------|
| ColBERTv2 residual compression | Moderate (training) | ✅ Use pretrained models |
| PLAID centroid pruning | Low (inference) | ✅ Reference impl available |
| SPLATE sparse candidate gen | Low | ✅ Just add SPLADE adapter |
| EMVB bit vectors | Low | ✅ Open source |
| MUVERA FDEs | Moderate | ⚠️ Need to train FDE projection |

---

## 2. Late Interaction Models Beyond ColBERT

### ColBERT-PRF (Pseudo-Relevance Feedback)

- Paper: https://dl.acm.org/doi/10.1145/3572405
- Applies pseudo-relevance feedback (PRF) to multi-vector retrieval
- Takes top-K results from initial ColBERT retrieval, aggregates their token embeddings to expand the query representation
- **PLAID-PRF** (SIGIR 2024): Extends this within the PLAID engine using centroid-like tokens
  - Paper: https://dl.acm.org/doi/10.1145/3805712.3809690
  - More efficient than naive ColBERT-PRF by operating in centroid space

### Jina-ColBERT-v2

- Paper: https://aclanthology.org/2024.mrl-1.11.pdf
- General-purpose **multilingual** late interaction model
- Key improvements over vanilla ColBERTv2:
  - Better multilingual support (trained on diverse multilingual data)
  - Improved token-level representations
  - Works across 89 languages
- Available on HuggingFace as `jinaai/jina-colbert-v2`

### PyLate

- Paper: https://arxiv.org/abs/2508.03555 (Aug 2025)
- A **library/framework** for flexible training and retrieval with late interaction models
- Standardizes training pipelines for ColBERT-family models
- Supports various pooling strategies, loss functions, and retrieval backends
- Makes it easier to experiment with late interaction architectures

### Sparsified Late Interaction (SLIM)

- Paper: https://dl.acm.org/doi/10.1145/3539618.3591977
- Combines sparse retrieval with late interaction scoring
- Uses inverted index structure for efficient multi-vector retrieval
- Bridges the gap between sparse methods and ColBERT-style models

### Constant-Space Multi-Vector Retrieval

- Paper: https://arxiv.org/abs/2504.01818 (2025)
- Addresses the storage problem from a different angle
- Achieves multi-vector quality with constant additional space per document
- Key insight: don't store all token embeddings — reconstruct on-the-fly from compressed representations

### IGP: Proximity Graph Index for Multi-Vector Retrieval

- Paper: SIGIR 2025
- Uses proximity graphs (like HNSW) adapted for multi-vector retrieval
- Alternative to PLAID's centroid-based approach
- Potentially better for very large-scale collections

### Key Takeaway

The late interaction space is maturing rapidly. The trend is toward:
1. **Reducing infrastructure complexity** (MUVERA → single-vector MIPS)
2. **Reducing storage** (residual compression, constant-space methods)
3. **Better candidate generation** (SPLATE → sparse, EMVB → bit vectors)
4. **Multilingual support** (Jina-ColBERT-v2)

---

## 3. RAPTOR Improvements

### Original RAPTOR (ICLR 2024)

- Paper: https://arxiv.org/abs/2401.18059
- **Recursive Abstractive Processing for Tree-Organized Retrieval**
- Builds a hierarchical tree of document summaries:
  1. Chunk documents into leaf nodes
  2. Cluster similar chunks (Gaussian Mixture Models)
  3. Summarize each cluster → parent nodes
  4. Repeat recursively until tree is built
- At query time: retrieve from any level of the tree (not just leaves)
- Enables both fine-grained and abstract-level retrieval

### Follow-Up Work

**Dynamic Tree Memory Representation for LLMs** (ICLR 2025):
- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/
- Extends RAPTOR with dynamic tree updates as new knowledge arrives
- Maintains tree structure incrementally rather than rebuilding from scratch

**An Abstract Bridge Tree Based RAG** (2026):
- Paper: https://arxiv.org/abs/2603.26668
- Combines RAPTOR-style hierarchical retrieval with multi-step reasoning
- Uses intermediate reasoning to formulate follow-up queries at different tree levels

**Beyond Chunking: Discourse-Aware Hierarchical Retrieval** (ACL 2026):
- Paper: https://aclanthology.org/2026.acl-long.829.pdf
- Integrates discourse structure (section headers, paragraph boundaries) into the hierarchical tree
- More linguistically-informed clustering than RAPTOR's pure embedding-based approach

### Implementability

RAPTOR is one of the **most implementable** advanced RAG techniques:
- Open source: https://github.com/parthsarthi03/raptor
- Works with any LLM for summarization
- Tree building is offline (one-time cost)
- Query time: just retrieve from the tree (can use standard vector search at each level)
- **Key limitation**: Summarization cost scales with document count; needs LLM calls for tree construction

---

## 4. SPLADE & Sparse-Dense Hybrid Approaches

### SPLADE Overview

**SPLADE** (Sparse Lexical and Expansion Model):
- Uses a pretrained language model (BERT) to generate **learned sparse representations**
- Two key capabilities:
  1. **Term weighting**: Assigns importance scores to word-pieces (unlike BM25's frequency-based weighting)
  2. **Term expansion**: Adds related terms not in the original text (addresses vocabulary mismatch)
- Produces sparse vectors over the BERT vocabulary (~30k dimensions)
- Compatible with inverted index structures (efficient retrieval)

### SPLADE Versions

**SPLADEv2**: Added regularization improvements and better training

**SPLADEv3** (Lassance et al., 2024):
- Further efficiency improvements
- Better calibration of term weights
- Used in production systems (e.g., SemEval-2026 Task 8)

**SPLADE++**: Ensemble and distillation variants

### Sparse-Dense Hybrid

The consensus in 2024-2026 is that **hybrid search** (sparse + dense) consistently outperforms either alone:

- **Reciprocal Rank Fusion (RRF)**: Standard method to combine sparse and dense rankings
  - `RRF_score(d) = Σ 1/(k + rank_i(d))` across retrieval methods
  - k typically 60
- **Implementation pattern**:
  1. Dense retrieval (embedding model) → top-K₁ results
  2. Sparse retrieval (BM25 or SPLADE) → top-K₂ results
  3. Merge via RRF or learned combiner
  4. Optional reranking stage

**Prompting LLMs for Both Dense and Sparse** (EMNLP 2024):
- Paper: https://aclanthology.org/2024.emnlp-main.250.pdf
- Uses LLMs to generate both dense and sparse representations
- Hybrid system surpasses BM25 and pure dense baselines

**Efficient Dense-Sparse Hybrid Vector Retrieval** (2024):
- Paper: https://arxiv.org/abs/2410.20381
- Addresses the lack of established methods for efficiently storing/retrieving hybrid vectors
- Unified index structure for both sparse and dense components

### Implementability

| Approach | Difficulty | Notes |
|----------|-----------|-------|
| BM25 + Dense hybrid | Easy | Standard in most vector DBs (Weaviate, Qdrant, Milvus) |
| SPLADE standalone | Moderate | Need to run BERT inference for indexing |
| SPLADE + Dense hybrid | Moderate | Best of both worlds |
| LLM-generated sparse+dense | High | Requires LLM calls per document |

---

## 5. HyDE (Hypothetical Document Embeddings)

### Original Paper

- **"Precise Zero-Shot Dense Retrieval without Relevance Labels"**
- Authors: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (CMU)
- Paper: https://arxiv.org/abs/2212.10496 (Dec 2022)

### How It Works

1. Given a query, prompt an LLM to generate a **hypothetical document** that would answer the query
2. Embed this hypothetical document using a standard embedding model
3. Use the embedding to retrieve real documents from the corpus
4. The embedding model's "dense bottleneck" filters out hallucinated details from the hypothetical doc

**Key insight**: The hypothetical document doesn't need to be factually correct — it just needs to be in the right **semantic neighborhood**. The embedding model maps it close to real documents on the same topic.

### Performance

- Significantly outperforms unsupervised dense retrievers (Contriever)
- Comparable to fine-tuned retrievers on web search, QA, fact verification
- Works across languages (sw, ko, ja tested)
- **Zero-shot**: No relevance labels needed

### Recent Improvements

**Adaptive HyDE** (2025):
- Paper: https://arxiv.org/abs/2507.16754
- Combines HyDE with full-answer context
- Adapts the hypothetical document generation based on query type
- Best results when hypothetical doc includes both the question context and a full answer draft

**Task-Adaptive Embedding Refinement** (2026):
- Paper: https://arxiv.org/abs/2605.12487
- Uses test-time LLM guidance to refine embeddings
- HyDE-style generation as one component of a broader refinement framework

### Implementability

**Highly implementable** — one of the easiest advanced retrieval techniques to add:
- Only change: add an LLM call before embedding the query
- No training required
- Works with any embedding model
- Cost: one extra LLM inference per query (can use small/fast models)
- **Best for**: queries that are abstract, conceptual, or don't match document vocabulary well
- **Not ideal for**: factual lookups where exact match matters (BM25 better)

---

## 6. GraphRAG

### Microsoft's Approach

- Paper: https://arxiv.org/abs/2404.16130 (Apr 2024)
- Blog: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- Code: https://github.com/microsoft/graphrag

### Core Architecture

GraphRAG builds a **knowledge graph + community summaries** in two phases:

**Phase 1: Graph Index Construction**
1. LLM extracts entities and relationships from source documents
2. Build a knowledge graph (entities as nodes, relationships as edges)
3. Detect communities using Leiden algorithm (hierarchical graph clustering)
4. LLM generates summaries for each community at each hierarchy level

**Phase 2: Query Processing**
- **Local search**: Given a query, find relevant entities → traverse graph → gather local context
- **Global search**: Use community summaries to answer broad/thematic questions
  - Each community summary generates a partial response
  - Partial responses are aggregated into final answer

### Key Innovation

Traditional RAG fails on **global questions** like "What are the main themes in this dataset?" — these require summarization over the entire corpus, not just retrieval of relevant chunks. GraphRAG solves this by pre-computing hierarchical summaries.

### Implementability Concerns

| Aspect | Assessment |
|--------|-----------|
| Graph construction | ❌ Expensive — requires many LLM calls per document |
| Community detection | ✅ Leiden algorithm is fast |
| Local search | ✅ Standard graph traversal |
| Global search | ⚠️ Needs pre-computed community summaries |
| Cost at scale | ❌ Very high for initial indexing; ~$1-10 per million tokens |

### Alternatives and Lightweight Variants

**LightRAG** (HKU, 2024):
- Code: https://github.com/hkuds/lightrag
- Lightweight alternative to Microsoft GraphRAG
- Dual-layer architecture for managing entities and relationships
- Significantly cheaper to build and query

**nano-graphrag**:
- Minimal implementation of GraphRAG concepts
- Good for prototyping and small-scale experiments

### When GraphRAG Makes Sense

- **Corpus has rich entity relationships** (legal docs, medical records, research papers)
- **Need to answer thematic/global questions** ("What are the trends in...")
- **Need explainability** (graph traversal shows reasoning path)
- **NOT ideal for**: simple factual QA over well-structured data (standard RAG is cheaper and faster)

---

## 7. Agentic RAG

### Survey Paper

- **"Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"**
- Authors: Singh, Ehtesham, Kumar, Khoei, Vasilakos
- Paper: https://arxiv.org/abs/2501.09136 (Jan 2025, updated Apr 2026)

### Core Concept

Agentic RAG integrates **autonomous AI agents** into the RAG pipeline, moving beyond static retrieve-then-read workflows. Key agentic patterns:

1. **Reflection**: Agent evaluates retrieved context, decides if more retrieval is needed
2. **Planning**: Agent decomposes complex queries into sub-queries
3. **Tool use**: Agent selects which retrieval tools/sources to use
4. **Multi-agent collaboration**: Different agents handle different aspects

### Taxonomy

The survey identifies architectures along four axes:
- **Agent cardinality**: Single agent vs. multi-agent
- **Control structure**: Sequential, adaptive, collaborative
- **Autonomy level**: Fixed pipeline vs. fully autonomous
- **Knowledge representation**: Vector DB, graph, hybrid

### Key Systems

**Adaptive-RAG** (2024):
- Tunes retrieval strategy based on query complexity
- Simple queries → direct retrieval; complex queries → multi-step reasoning

**Self-RAG** (2024):
- LLM decides when to retrieve and how to use retrieved content
- Special reflection tokens guide retrieval decisions

**CRAG (Corrective RAG)**:
- Evaluates retrieval quality and triggers re-retrieval if insufficient
- Applies knowledge refinement before generation

### Implementability

| Pattern | Difficulty | Practical? |
|---------|-----------|-----------|
| Simple reflection loop | Easy | ✅ Add "is this sufficient?" check |
| Query decomposition | Moderate | ✅ LLM decomposes, retrieve per sub-query |
| Tool selection | Moderate | ⚠️ Needs routing logic |
| Multi-agent | High | ⚠️ Complex orchestration, latency |

**Most practical for raven-retrieval**: Simple reflection + query decomposition patterns. The key is having the LLM decide if retrieved context is sufficient, and if not, reformulate the query.

---

## 8. Contextual Retrieval (Anthropic)

### Announcement

- Published: Sep 19, 2024
- URL: https://www.anthropic.com/engineering/contextual-retrieval
- Cookbook: https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide

### The Problem

When documents are chunked for RAG, individual chunks lose context. Example:
- Original chunk: "The company's revenue grew by 3% over the previous quarter."
- Problem: Which company? What time period? Impossible to retrieve correctly for "What was ACME Corp's Q2 2023 revenue growth?"

### The Solution

**Contextual Retrieval** prepends chunk-specific context to each chunk before embedding and BM25 indexing:

```
contextualized_chunk = "This chunk is from an SEC filing on ACME corp's 
performance in Q2 2023; the previous quarter's revenue was $314 million. 
The company's revenue grew by 3% over the previous quarter."
```

Two sub-techniques:
1. **Contextual Embeddings**: Prepend context before embedding
2. **Contextual BM25**: Prepend context before BM25 indexing

### Results

- Contextual Embeddings alone: **35% reduction** in failed retrievals (top-20)
- Contextual Embeddings + Contextual BM25: **49% reduction**
- With reranking: **67% reduction**

### Implementation Details

Use an LLM (Claude 3 Haiku in their example) with this prompt per chunk:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall 
document for the purposes of improving search retrieval of the chunk. 
Answer only with the succinct context and nothing else.
```

**Cost optimization**: Use prompt caching — load document once, generate context for all chunks. Cost: ~$1.02 per million document tokens.

### Implementability

**Highly implementable** — this is one of the most practical improvements:
- Works with any embedding model
- Works with any BM25 implementation
- One-time preprocessing cost
- Can use any LLM for context generation (not just Claude)
- **Key**: The context is short (50-100 tokens) so minimal impact on embedding quality
- **Synergy with late chunking**: Contextual Retrieval and Late Chunking address the same problem differently and could be combined

---

## 9. Late Chunking (Jina)

### Paper

- **"Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"**
- Authors: Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, Han Xiao (Jina AI)
- Paper: https://arxiv.org/abs/2409.04701 (Sep 2024, updated Jul 2025)
- Code: https://github.com/jina-ai/late-chunking

### The Problem

Standard chunking: split document → embed each chunk independently. This loses cross-chunk context.

### The Solution: "Late" Chunking

Instead of chunking **before** the transformer, chunk **after**:

1. Pass the **entire document** through the transformer model
2. Get contextualized token embeddings for ALL tokens (with full document context)
3. **Then** split into chunks
4. Apply mean pooling per chunk to get chunk embeddings

The "late" refers to doing the chunking after the transformer layers, not before.

### Key Technical Details

- Requires a **long-context embedding model** (Jina-embeddings-v2 supports 8K tokens)
- No additional training needed (though dedicated fine-tuning improves results)
- Each chunk embedding benefits from cross-chunk attention
- Trade-off: must process the full document even if you only need one chunk

### Performance

- Superior results across various retrieval tasks vs. standard chunking
- Works with any long-context embedding model (not just Jina's)
- Most beneficial when chunks are semantically dependent on surrounding context
- For some datasets, no chunking at all performs best (context window permitting)

### Implementability

**Moderately implementable**:
- ✅ No training needed if you have a long-context embedding model
- ✅ Simple implementation: just change the chunking order
- ⚠️ Requires long-context embedding model (8K+ tokens)
- ⚠️ Higher compute per chunk (processes full document)
- ⚠️ Maximum document length limited by model's context window

**Synergy**: Combine with Contextual Retrieval for maximum context preservation.

---

## 10. ColBERT Token Pooling Strategies

### Standard Approach: Mean Pooling

ColBERT uses **mean pooling** over token embeddings to create document-level representations, but the core innovation is keeping **per-token embeddings** for late interaction (MaxSim scoring).

### Research on Pooling Strategies

**Anchor Token Aware (ATA) Pooling**:
- Uses attention-based weighting to determine which tokens are most important
- Learns an attention mechanism that weights tokens differently for pooling
- More expressive than simple mean pooling

**Learnable Pooling Queries** (2024-2025):
- Introduces learnable "pooling query" vectors
- These queries attend to token embeddings to produce pooled representations
- Similar to attention pooling but with learned query vectors

**Enhancing ColBERT via Token Pruning** (PACLIC 2024):
- Paper: https://aclanthology.org/2024.paclic-1.79.pdf
- Small network assigns importance weights to tokens
- Prunes low-weight tokens to reduce storage and computation
- Key insight: not all tokens contribute equally to relevance

**Multi-Vector Index Compression** (2026):
- Paper: https://arxiv.org/abs/2602.21202
- **AGC (Adaptive Group Compression)**: Learnable universal query tokens select centroids and weight cluster pooling
- Combines pooling with compression in a unified framework

**Variance-Weighted Pooling**:
- Weights tokens by the variance of their embedding dimensions
- High-variance tokens carry more discriminative information

**Top-K Pooling**:
- Only pool over the K most important tokens
- Reduces noise from common/stopword tokens

### Practical Recommendations

For ColBERT-style models:
1. **Default**: Mean pooling works well as baseline
2. **Attention pooling**: Better when token importance varies significantly
3. **Learned pooling**: Best if you can afford training
4. **Token pruning**: Best for storage-constrained environments

---

## 11. Concrete Ideas for raven-retrieval

Based on the research above, here are prioritized improvement ideas:

### Tier 1: High Impact, Easy to Implement

1. **Contextual Retrieval** (Anthropic)
   - Add LLM-generated context prefix to each chunk before embedding
   - ~35-49% reduction in failed retrievals
   - Works with existing infrastructure
   - Implementation: preprocessing step + standard retrieval

2. **HyDE (Hypothetical Document Embeddings)**
   - Generate hypothetical answer before retrieval
   - Zero-shot improvement for abstract/conceptual queries
   - Implementation: one LLM call per query before embedding
   - Use small/fast model (e.g., 7B) for hypothetical doc generation

3. **Hybrid Search (BM25 + Dense + RRF)**
   - If not already implemented, add BM25 alongside dense retrieval
   - Reciprocal Rank Fusion to combine results
   - Consistent improvement across virtually all benchmarks
   - Most vector databases support this natively

4. **Late Chunking**
   - If using long-context embedding model, process full document first
   - Chunk after transformer, not before
   - No training needed
   - Implementation change: reorder chunking pipeline

### Tier 2: High Impact, Moderate Effort

5. **SPLADE-style Sparse Retrieval**
   - Replace or augment BM25 with learned sparse representations
   - Better term weighting and expansion
   - Use SPLADEv3 or SPLATE for candidate generation

6. **RAPTOR Hierarchical Retrieval**
   - Build tree of summaries for multi-level retrieval
   - Especially good for long documents with hierarchical structure
   - Open source implementation available
   - One-time tree-building cost

7. **Query Decomposition (Agentic RAG pattern)**
   - For complex queries, decompose into sub-queries
   - Retrieve per sub-query, aggregate results
   - Simple implementation: LLM-based decomposition

### Tier 3: High Impact, Significant Effort

8. **ColBERT / Late Interaction Retrieval**
   - Replace single-vector retrieval with multi-vector for better relevance
   - Use Jina-ColBERT-v2 for multilingual support
   - Consider SPLATE for CPU-friendly candidate generation
   - MUVERA for single-vector approximation if storage is constrained

9. **GraphRAG for Entity-Rich Domains**
   - Build knowledge graph from documents
   - Use for queries requiring relationship understanding
   - Start with LightRAG (lighter weight)
   - Only if corpus has rich entity relationships

10. **Reflection Loop (Agentic RAG)**
    - After retrieval, LLM evaluates if context is sufficient
    - If not, reformulate query and re-retrieve
    - Adds latency but improves quality on hard queries

### Architecture Recommendation for raven-retrieval

```
Query → [HyDE: generate hypothetical doc] 
      → [Hybrid Search: Dense + BM25/SPLADE] 
      → [RRF Fusion] 
      → [Reranker (cross-encoder)] 
      → [Reflection: sufficient? If not, re-retrieve]
      → Context assembly → LLM generation
```

With preprocessing:
```
Documents → [Chunking] → [Contextual Enrichment] → [Embed + BM25 index]
         → [Optional: RAPTOR tree construction]
         → [Optional: Graph construction]
```

---

## Key Papers Summary

| Paper | Year | URL | Key Contribution |
|-------|------|-----|-----------------|
| ColBERTv2 | 2022 | https://arxiv.org/abs/2112.01488 | Residual compression for late interaction |
| PLAID | 2023 | https://dl.acm.org/doi/10.1145/3511808.3557325 | Centroid interaction pruning engine |
| SPLATE | 2024 | https://arxiv.org/abs/2404.13950 | Sparse candidate generation for ColBERT |
| EMVB | 2024 | https://arxiv.org/abs/2404.02805 | Bit vector filtering for multi-vector retrieval |
| MUVERA | 2024 | https://arxiv.org/abs/2405.19504 | Multi-vector → single-vector via FDEs |
| Jina-ColBERT-v2 | 2024 | https://aclanthology.org/2024.mrl-1.11.pdf | Multilingual late interaction |
| ColBERT-PRF | 2023 | https://dl.acm.org/doi/10.1145/3572405 | Pseudo-relevance feedback for ColBERT |
| RAPTOR | 2024 | https://arxiv.org/abs/2401.18059 | Hierarchical tree retrieval |
| SPLADE | 2022 | https://arxiv.org/abs/2107.05720 | Learned sparse retrieval |
| HyDE | 2022 | https://arxiv.org/abs/2212.10496 | Hypothetical document embeddings |
| GraphRAG | 2024 | https://arxiv.org/abs/2404.16130 | Graph-based retrieval for global queries |
| Agentic RAG Survey | 2025 | https://arxiv.org/abs/2501.09136 | Survey of agent-integrated RAG |
| Contextual Retrieval | 2024 | https://www.anthropic.com/engineering/contextual-retrieval | Chunk context enrichment |
| Late Chunking | 2024 | https://arxiv.org/abs/2409.04701 | Post-transformer chunking |
| PyLate | 2025 | https://arxiv.org/abs/2508.03555 | Late interaction training framework |
| Adaptive HyDE | 2025 | https://arxiv.org/abs/2507.16754 | Improved HyDE with full-answer context |

---

## Glossary

- **MaxSim**: ColBERT's similarity function — for each query token, find max similarity with document tokens, then sum
- **Late Interaction**: Architecture where query and document are encoded independently, but interaction happens at a later stage (token-level similarity)
- **Residual Compression**: Storing only the difference between an embedding and its nearest centroid
- **PRF**: Pseudo-Relevance Feedback — using top retrieved results to improve the query
- **RRF**: Reciprocal Rank Fusion — method to combine rankings from multiple retrieval systems
- **FDE**: Fixed Dimensional Encoding — MUVERA's method to convert multi-vector to single-vector
- **Leiden Algorithm**: Community detection algorithm used in GraphRAG for graph clustering
- **MaxSim**: The core scoring mechanism in ColBERT — max cosine similarity per query token, summed across all query tokens
