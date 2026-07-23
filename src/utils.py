"""Shared utilities for raven-retrieval.

Single source of truth for operations that were previously copy-pasted
across 8+ modules (chunking, corpus iteration, doc-score aggregation).

Also provides:
- Timing helpers for honest index/query latency reporting
- Masked pooling helpers for token-level embeddings
- Dependency-injection-friendly corpus helpers
"""

import time
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── Corpus helpers ───────────────────────────────────────────────────

def iter_corpus(corpus):
    """Yield (doc_id, doc_dict) pairs from a BEIR corpus.

    Handles both dict ({doc_id: {"title":..., "text":...}}) and
    list ([{"_id":..., "title":..., "text":...}]) forms.
    """
    if isinstance(corpus, dict):
        yield from corpus.items()
    else:
        for doc in corpus:
            yield doc["_id"], doc


def full_doc_text(doc):
    """Concatenate title + text the way every pipeline does."""
    return (doc.get("title", "") + " " + doc.get("text", "")).strip()


# ── Tokenization helpers (for BM25) ────────────────────────────────

# Common English stop words — used by BM25 tokenization across all
# pipelines. Keeping this in utils ensures consistency (the old code
# used .lower().split() everywhere, which overstated BM25's weaknesses
# by including stop words and unstemmed variants in scoring).
ENGLISH_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "also", "this", "that", "these", "those", "what",
    "which", "who", "whom", "it", "its", "he", "she", "they", "we",
    "me", "my", "your", "his", "her", "our", "their", "you", "i",
})


def tokenize_for_bm25(text):
    """Tokenize text for BM25 with stemming and stop-word removal.

    Uses simple Porter-style suffix stripping (no NLTK dependency).
    This is more correct than .lower().split() — BM25 performs better
    with stemmed tokens because "running" and "runs" map to the same
    term, and stop words add noise to TF*IDF scoring.
    """
    # Lowercase and split on whitespace + punctuation
    import re
    tokens = re.findall(r'\b[a-z]+\b', text.lower())

    # Remove stop words
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) >= 2]

    # Simple suffix stripping (approximates Porter stemmer without NLTK)
    stemmed = []
    for token in tokens:
        # Strip common suffixes
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("tion") and len(token) > 5:
            token = token[:-4] + "e"
        elif token.endswith("sion") and len(token) > 5:
            token = token[:-4] + "s"
        elif token.endswith("ness") and len(token) > 5:
            token = token[:-4]
        elif token.endswith("ment") and len(token) > 5:
            token = token[:-4]
        elif token.endswith("able") and len(token) > 5:
            token = token[:-4] + "e"
        elif token.endswith("ible") and len(token) > 5:
            token = token[:-4] + "e"
        elif token.endswith("ly") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("ity") and len(token) > 4:
            token = token[:-3] + "e"
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 3:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3 and token[-2] not in "su":
            token = token[:-1]
        stemmed.append(token)

    return stemmed


# ── Chunking ─────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=200, chunk_overlap=50):
    """Split text into word-level chunks with overlap.

    This is THE chunker used by all baseline retrievers. RAPTOR uses
    src.raptor.chunker.TextChunker (returns dicts with metadata).
    """
    tokens = text.split()
    if not tokens:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - chunk_overlap
    return chunks


def chunk_corpus(corpus, chunk_size=200, chunk_overlap=50, id_template="{doc}::chunk::{i}"):
    """Chunk an entire corpus. Returns (chunk_ids, chunk_texts)."""
    all_ids, all_texts = [], []
    for doc_id, doc in iter_corpus(corpus):
        for i, chunk in enumerate(chunk_text(full_doc_text(doc), chunk_size, chunk_overlap)):
            all_ids.append(id_template.format(doc=doc_id, i=i))
            all_texts.append(chunk)
    return all_ids, all_texts


# ── Score aggregation ────────────────────────────────────────────────

def aggregate_doc_scores(chunk_ids, chunk_scores, top_k=10):
    """Aggregate chunk-level scores to document level (max-pooling).

    Chunk IDs follow the "{doc_id}::..." convention; the doc id is the
    part before the first "::". Returns [(doc_id, score), ...] sorted
    by descending score, truncated to top_k.
    """
    doc_scores = {}
    for cid, score in zip(chunk_ids, chunk_scores):
        doc_id = cid.split("::")[0]
        s = float(score)
        if doc_id not in doc_scores or s > doc_scores[doc_id]:
            doc_scores[doc_id] = s
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(rankings, k=60):
    """Reciprocal Rank Fusion over multiple rankings.

    rankings: list of [(doc_id, score), ...] sorted lists.
    Returns fused [(doc_id, rrf_score), ...] sorted descending.
    """
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Embedding helpers ────────────────────────────────────────────────

def l2_normalize(matrix, axis=-1):
    """L2-normalize a numpy array along an axis (safe for zero vectors)."""
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def assign_centroids(tokens, centroids):
    """Assign each token to its nearest centroid (vectorized via ||a-b||² expansion).

    Uses ||a-b||² = |a|² + |b|² - 2a·b to avoid materializing
    (n_tokens, n_centroids, dim) intermediate arrays.

    This is the canonical version — used by compression.py and approximate.py
    instead of their own local copies.
    """
    tokens = to_numpy(tokens)
    centroid_sq = (centroids ** 2).sum(axis=1)
    token_sq = (tokens ** 2).sum(axis=1, keepdims=True)
    dists_sq = token_sq + centroid_sq[None, :] - 2.0 * (tokens @ centroids.T)
    return np.argmin(dists_sq, axis=1)


def to_numpy(x):
    """Convert torch tensors or array-like objects to numpy float32 array.

    Handles: torch tensors (detach + cpu), numpy arrays, lists.
    Squeezes leading batch dimension if present (3D → 2D).
    This is the canonical version — used by maxsim modules instead of
    their own local copies.
    """
    try:
        import torch
        if hasattr(x, 'detach') and hasattr(x, 'cpu'):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x.squeeze(0)
    return x


def masked_mean_pool(token_embeddings, mask):
    """Mean-pool token embeddings respecting the attention mask.

    token_embeddings: (seq_len, dim) numpy array
    mask: (seq_len,) array of 0/1
    Returns (dim,) vector. Padding positions contribute nothing.
    """
    mask = np.asarray(mask, dtype=np.float32)
    total = mask.sum()
    if total < 1:
        total = 1.0
    return (token_embeddings * mask[:, None]).sum(axis=0) / total


# ── Timing ───────────────────────────────────────────────────────────

class PipelineTimer:
    """Honest timing: index time and query time tracked separately.

    Previous benchmarks compared "index+query" for heavy pipelines against
    "query only" for light ones — this makes the comparison explicit.

    Usage:
        timer = PipelineTimer("naive_dense")
        with timer.index_phase():
            retriever.index(corpus)
        with timer.query_phase():
            ...run all queries...
        record = timer.record(n_queries=100)
    """

    def __init__(self, name):
        self.name = name
        self.index_s = 0.0
        self.query_s = 0.0
        self._t0 = None
        self._phase = None

    class _Phase:
        def __init__(self, timer, phase):
            self.timer = timer
            self.phase = phase

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self._t0
            if self.phase == "index":
                self.timer.index_s += elapsed
            else:
                self.timer.query_s += elapsed
            return False

    def index_phase(self):
        return PipelineTimer._Phase(self, "index")

    def query_phase(self):
        return PipelineTimer._Phase(self, "query")

    def record(self, n_queries=0):
        return {
            "index_s": round(self.index_s, 3),
            "query_s": round(self.query_s, 3),
            "total_s": round(self.index_s + self.query_s, 3),
            "per_query_ms": round(1000.0 * self.query_s / n_queries, 2) if n_queries else None,
        }
