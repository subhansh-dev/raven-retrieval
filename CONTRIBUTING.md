# Contributing to Raven-Retrieval

## Development Setup

```bash
git clone https://github.com/subhansh-dev/raven-retrieval.git
cd raven-retrieval
pip install -r requirements.txt -r requirements-dev.txt
```

## Running Tests

```bash
# Core tests (no torch required)
python tests/run_core_tests.py

# Full test suite
python -m pytest tests/ -v
```

## Adding a New Pipeline

1. Create `src/baselines/your_pipeline.py`
2. Implement a class with `index(corpus)` and `retrieve(query, top_k)` methods
3. Add lazy import to `src/baselines/__init__.py`
4. Add to `run_enhanced_benchmark.py`
5. Add tests to `tests/test_integration.py`
6. Update README with description and expected results

## Code Style

- Follow PEP 8 (120 char line limit)
- Add docstrings to all public methods
- Use type hints where practical
- Log with `logging` module, not `print()`

## Commit Messages

```
type: short description

Longer description if needed.

- What changed
- Why it changed
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update README if adding a pipeline
6. Submit PR with clear description
