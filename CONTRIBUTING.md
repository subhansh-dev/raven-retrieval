# Contributing

## Setup

```bash
git clone https://github.com/subhansh-dev/raven-retrieval.git
cd raven-retrieval
pip install -r requirements.txt -r requirements-dev.txt
```

## Running Tests

```bash
# Core tests (no torch)
python tests/run_core_tests.py

# Full suite
python -m pytest tests/ -v
```

## Adding a Pipeline

1. Create `src/baselines/your_pipeline.py` with `index(corpus)` and `retrieve(query, top_k)` methods
2. Add lazy import to `src/baselines/__init__.py`
3. Register in `run_enhanced_benchmark.py`
4. Add at least a smoke test in `tests/`
5. Update README

## Code Style

- PEP 8, 120 char lines
- Docstrings on public methods
- Type hints where they help
- `logging` not `print()`

## Commits

Use `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:` prefixes. Keep it short.
