# AGENTS.md

Dev contract for coding agents when working in this repository.

## Environment

Use the `mlops` Conda environment from `environment.yml` for development and verification.

## Commands

```text
ruff check src/ tests/                 # lint
mypy src/yield_risk                    # type-check

pytest                                  # all tests
pytest tests/test_file.py               # single file
pytest -k "test_name"                   # single test by name
```

## Coding style
- **Docstrings.** Every public module, class, and function gets a
  Google-style docstring with `Args:`, `Returns:`, and `Raises:` sections
  where applicable.
- **Type annotations.** All function signatures must be fully annotated;
  `mypy --strict` must pass.
- **Ruff** enforces `E`, `F`, `W`, `I`, `UP`, and `ANN` rules at line
  length 88. Fix lint errors before committing.
