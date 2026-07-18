"""Baseline retrieval pipelines.

Lazy imports to avoid pulling in heavy dependencies (torch, transformers)
when only specific modules are needed.
"""


def __getattr__(name):
    """Lazy import for baseline modules."""
    _import_map = {
        "DenseRetriever": ".dense",
        "HybridRetriever": ".hybrid",
        "HyDERetriever": ".hyde",
        "SPLADERetriever": ".splade",
        "HybridSPLADERetriever": ".splade",
        "ContextualDenseRetriever": ".contextual",
        "ContextualBM25Retriever": ".contextual",
        "ContextualHybridRetriever": ".contextual",
        "LateChunkingRetriever": ".late_chunking",
        "QueryDecomposer": ".agentic",
        "ReflectionRetriever": ".agentic",
        "MultiHopRetriever": ".agentic",
        "CrossEncoderReranker": ".reranker",
        "TwoStageRetriever": ".reranker",
        "BM25PRFRetriever": ".bm25_prf",
        "GraphRetriever": ".graph_retrieval",
        "DocumentGraph": ".graph_retrieval",
    }

    if name in _import_map:
        import importlib
        module = importlib.import_module(_import_map[name], __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
