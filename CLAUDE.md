# CLAUDE.md

Dev contract for Claude Code when working in this repository.

## Environment

Conda lives at `$env:USERPROFILE\miniconda3`. Claude Code does not inherit a login shell, so the `mlops` env must be activated explicitly before running Python commands.

**PowerShell tool:**
```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
```

**Bash tool:**
```bash
source ~/miniconda3/Scripts/activate
conda activate mlops
```

## Commands

```bash
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
