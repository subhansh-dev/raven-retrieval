"""Configuration management for raven-retrieval.

Centralizes all hyperparameters, model settings, and evaluation config.
Supports YAML config files, CLI overrides, and environment variables.

Usage:
    from src.config import ExperimentConfig
    config = ExperimentConfig.from_yaml("configs/scifact.yaml")
    config = ExperimentConfig.defaults(dataset="scifact")
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """ColBERT encoder configuration."""
    model_name: str = "bert-base-uncased"
    projection_dim: int = 128
    projection_init: str = "xavier"  # "xavier" or "random"
    use_flash_attention: bool = False
    max_query_length: int = 64
    max_doc_length: int = 256


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = 200
    chunk_overlap: int = 50


@dataclass
class ClusteringConfig:
    """RAPTOR clustering configuration."""
    soft_threshold: float = 0.1
    min_cluster_size: int = 3
    max_k: int = 10
    umap_n_components: int = 10


@dataclass
class SummarizerConfig:
    """Summarizer configuration."""
    model_name: str = "facebook/bart-large-cnn"
    max_chunk_tokens: int = 1024
    max_summary_tokens: int = 200
    fallback_to_extractive: bool = True


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    top_k: int = 10
    first_stage_k: int = 100  # For two-stage retrieval
    rrf_k: int = 60           # Reciprocal Rank Fusion k
    bm25_prf_k: int = 5       # PRF top-k documents
    expansion_terms: int = 10  # PRF expansion terms
    splade_idf_reweighting: bool = True
    hyde_use_llm: bool = False  # Use LLM for HyDE (vs template)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 100])
    n_resamples: int = 10000   # Bootstrap resamples
    alpha: float = 0.05        # Significance level
    correction: str = "bonferroni"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment
    dataset: str = "scifact"
    max_queries: Optional[int] = 100
    seed: int = 42
    output_dir: str = "./experiments/runs"

    # Components
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Pipelines to run
    pipelines: List[str] = field(default_factory=lambda: [
        "naive_dense", "hybrid_rag", "hyde", "splade", "bm25_prf",
        "contextual_hybrid", "late_interaction", "raptor_late_collapsed",
    ])

    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 1

    @classmethod
    def defaults(cls, dataset="scifact", **overrides):
        """Create default config with optional overrides."""
        config = cls(dataset=dataset)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif "." in key:
                # Handle nested config like "encoder.projection_dim"
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        return config

    @classmethod
    def from_yaml(cls, path):
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path):
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data):
        """Create config from dict."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if isinstance(attr, (EncoderConfig, ChunkingConfig, ClusteringConfig,
                                     SummarizerConfig, RetrievalConfig, EvaluationConfig)):
                    # Nested config
                    for k, v in value.items():
                        if hasattr(attr, k):
                            setattr(attr, k, v)
                else:
                    setattr(config, key, value)
        return config

    def to_dict(self):
        """Convert to dict."""
        return asdict(self)

    def to_json(self, path):
        """Save as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path):
        """Save as YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def validate(self):
        """Validate configuration."""
        valid_pipelines = {
            "naive_dense", "hybrid_rag", "hyde", "splade", "splade_hybrid",
            "bm25_prf", "contextual_hybrid", "late_chunking",
            "late_interaction", "raptor_single_vector",
            "raptor_late_collapsed", "raptor_late_traversal",
        }
        invalid = set(self.pipelines) - valid_pipelines
        if invalid:
            raise ValueError(f"Unknown pipelines: {invalid}")

        if self.max_queries is not None and self.max_queries < 1:
            raise ValueError("max_queries must be >= 1")

        if self.retrieval.top_k < 1:
            raise ValueError("top_k must be >= 1")

        return True


# Pre-built configs for common experiments
SCIFACT_DEFAULTS = ExperimentConfig.defaults(
    dataset="scifact",
    max_queries=100,
    pipelines=["naive_dense", "hybrid_rag", "hyde", "bm25_prf", "contextual_hybrid"],
)

HOTPOTQA_DEFAULTS = ExperimentConfig.defaults(
    dataset="hotpotqa",
    max_queries=50,
    pipelines=["naive_dense", "hybrid_rag", "hyde"],
)

FULL_ABLATION = ExperimentConfig.defaults(
    dataset="scifact",
    max_queries=100,
    pipelines=[
        "naive_dense", "hybrid_rag", "hyde", "splade", "splade_hybrid",
        "bm25_prf", "contextual_hybrid", "late_chunking",
        "late_interaction", "raptor_single_vector",
        "raptor_late_collapsed", "raptor_late_traversal",
    ],
)
