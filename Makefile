.PHONY: install test lint benchmark clean help

# Python interpreter
PYTHON ?= python3
PIP ?= pip

# Default dataset and query count
DATASET ?= scifact
MAX_QUERIES ?= 100
MAX_DOCS ?=
TOP_K ?= 10

# Optional corpus subsampling flag (set MAX_DOCS=2000 for low-RAM machines)
ifdef MAX_DOCS
MAX_DOCS_FLAG = --max-docs $(MAX_DOCS)
else
MAX_DOCS_FLAG =
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install dev dependencies (linting, testing)
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-core: ## Run core tests (no torch required)
	$(PYTHON) tests/run_core_tests.py

lint: ## Run linting
	$(PYTHON) -m flake8 src/ --max-line-length=120 --ignore=E501,W503
	$(PYTHON) -m mypy src/ --ignore-missing-imports --no-strict-optional

benchmark: ## Run unified benchmark (all default pipelines)
	$(PYTHON) run_enhanced_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES) --top-k $(TOP_K) $(MAX_DOCS_FLAG)

benchmark-fast: ## Run fast benchmark (skip heavy pipelines)
	$(PYTHON) run_enhanced_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES) --top-k $(TOP_K) --skip-heavy $(MAX_DOCS_FLAG)

benchmark-baselines: ## Run baselines only
	$(PYTHON) run_enhanced_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES) --top-k $(TOP_K) \
		--pipelines naive_dense hybrid_rag bm25_prf contextual_hybrid hyde $(MAX_DOCS_FLAG)

benchmark-trained: ## Run ColBERT pipelines with a trained checkpoint
	$(PYTHON) run_enhanced_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES) \
		--pipelines late_interaction late_interaction_approx raptor_late_collapsed raptor_late_traversal \
		--colbert-checkpoint checkpoints/final_model.pt $(MAX_DOCS_FLAG)

train-colbert: ## Train ColBERT encoder on BEIR triples
	$(PYTHON) -m src.training.train_colbert --beir-dataset $(DATASET) --epochs 3

report: ## Generate report from latest run
	$(PYTHON) -m src.eval.report $(shell ls -td experiments/runs/*/ | head -1)

clean: ## Remove generated files
	rm -rf experiments/runs/*
	rm -rf data/
	rm -rf checkpoints/
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__

all: install test benchmark report ## Full pipeline: install, test, benchmark, report
