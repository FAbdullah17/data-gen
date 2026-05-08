# Contributing to data-gen

First off, thank you for considering contributing to `data-gen`! It's people like you that make this tool great. 

## 🛠️ Development Setup

To set up a local development environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/data-gen.git
   cd data-gen
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[all]"
   pip install pytest pytest-cov pre-commit build twine
   ```

4. **Install Pre-commit hooks:**
   We enforce strict formatting and linting rules using `pre-commit` (which wraps `ruff` and `mypy`):
   ```bash
   pre-commit install
   ```

## 🧪 Testing

We use `pytest` for running our test suite. Before submitting any changes, ensure all tests pass and coverage is maintained.

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/data_gen --cov-report=term-missing
```

## 📝 Code Style & Linting

All PRs must pass the CI checks before they can be merged. You can run the CI checks locally to ensure your code complies:

```bash
# Run all formatters and linters manually
pre-commit run --all-files
```
Our main tools are:
- `ruff` for fast python linting and formatting.
- `mypy` for static type checking.

## 🚀 Submitting a Pull Request

1. Fork the repository and create your branch from `main`.
2. Ensure you have added corresponding tests for your feature or bugfix.
3. Verify that your tests pass (`pytest`) and code lints cleanly (`pre-commit`).
4. Update the `CHANGELOG.md` with a summary of your changes.
5. Create the PR describing the problem you fixed or the feature you added. 

Thank you for your contribution!
