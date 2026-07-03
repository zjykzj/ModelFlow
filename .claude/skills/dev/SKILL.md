---
name: dev
description: Run development commands — test, lint, typecheck. Use when the user asks to run tests, lint, type check, or check test coverage.
allowed-tools: Bash
---

# Development Commands

## Testing

```bash
pytest                              # All tests (from project root)
pytest -x -q                        # Stop on first failure, quiet
pytest modelflow/tests/test_processors.py  # Single file
pytest modelflow/tests/test_processors.py::test_detect_postprocess_nms  # Single test
pytest --cov=modelflow --cov-report=html  # Coverage report
pytest -n auto                      # Parallel (requires pytest-xdist)
```

Project-specific test paths, Docker environments, and dataset download scripts are documented in CLAUDE.md.

## Linting

```bash
black modelflow eval data export vlms  # Format
isort modelflow eval data export vlms  # Imports
flake8 modelflow eval data export vlms # Lint
mypy modelflow                         # Type check
```
