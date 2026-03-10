# Contributing to Lightweight Gravitational Transformer

Thank you for considering contributing to LGT! This document outlines the development setup, coding standards, and pull-request process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Coding Standards](#coding-standards)
- [Pull-Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

---

## Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/<your-username>/Lightweight-Gravitational-Transformer.git
cd Lightweight-Gravitational-Transformer

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# Install in editable mode with dev extras
pip install -e ".[dev]"
```

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a single test class
pytest tests/test_lgt.py::TestGravitationalAttentionHead -v

# Run a single test method
pytest tests/test_lgt.py::TestGravitationalAttentionHead::test_output_shape -v
```

All tests must pass before submitting a pull request. New features must include corresponding tests in `tests/test_lgt.py`.

---

## Coding Standards

- **Python version**: Target Python 3.9+.
- **Type hints**: All public functions and class `__init__` signatures must include type hints.
- **Docstrings**: Use NumPy-style docstrings for all public classes and functions.
- **Line length**: 100 characters maximum.
- **Formatting**: Code should be consistently formatted; match the style of existing modules.
- **Imports**: Standard library first, then third-party (`torch`, `numpy`), then local imports. One blank line between groups.
- **Physics parameters**: Any new physics-inspired parameter (G, curvature, masses, etc.) must be documented with the physical intuition in its docstring.
- **No silent failures**: Raise informative `ValueError` or `RuntimeError` with a descriptive message rather than silently returning incorrect results.

---

## Pull-Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make your changes** with clear, focused commits.
3. **Add or update tests** in `tests/test_lgt.py`.
4. **Ensure all tests pass**: `pytest tests/ -v`
5. **Update documentation**:
   - Add your change to `CHANGELOG.md` under `[Unreleased]`.
   - Update the relevant section(s) in `docs/` and/or `README.md`.
6. **Open a pull request** against `main` with a clear title and description.

### PR Title Format

```
<type>: <short description>

Types: feat | fix | docs | refactor | test | chore
```

Examples:
- `feat: add learnable event horizon per attention head`
- `fix: prevent NaN in GravitationalAttentionHead when positions are zero`
- `docs: add fractal position embedding tutorial to user guide`

---

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/issues) and include:

1. **Python and PyTorch versions** (`python --version`, `python -c "import torch; print(torch.__version__)"`)
2. **Minimal reproducible example** — the smallest code snippet that triggers the bug.
3. **Expected behaviour** vs **actual behaviour**.
4. **Full traceback** (if applicable).

---

## Feature Requests

Open a [GitHub Issue](https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/issues) labelled `enhancement` describing:

1. **Motivation** — what problem does the feature solve?
2. **Proposed API** — what would the interface look like?
3. **Alternatives considered** — what other approaches did you evaluate?
