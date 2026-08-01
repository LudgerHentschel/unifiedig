# Contributing

Unified IG is in an early design phase. Changes should preserve its small,
model-agnostic public API and keep model-specific behavior inside backends.

## Development setup

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Checks

Before opening a pull request, run:

```console
pytest
python -m build
python -m twine check dist/*
```

New backends should include completeness tests showing that attribution sums
recover the explained model output relative to the baseline output.

