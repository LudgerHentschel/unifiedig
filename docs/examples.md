---
myst:
  html_meta:
    description: "Run complete UnifiedIG examples for regression, classification, feature spaces, and explicit numerical fallback."
---

# Worked examples

These complete scripts run from the repository with `python examples/<name>.py`.
The core examples need only a normal UnifiedIG installation. The documentation
checker executes them in separate processes, and the site includes the same
source files rather than maintaining copied snippets.

## Regression with CBaseline

```{literalinclude} ../examples/quickstart.py
:language: python
:lines: 3-
```

## Multiclass classification

The background is calibrated on centered scores. Contributions reconstruct
those scores; pairwise contrasts require no second model pass.

```{literalinclude} ../examples/multiclass_classification.py
:language: python
:lines: 3-
```

## Pipeline feature spaces

Compare original features with outputs of named preprocessing steps. Both the
evaluation data and baseline rows are transformed together.

```{literalinclude} ../examples/feature_spaces.py
:language: python
:lines: 3-
```

## Smooth numerical fallback

A Gaussian-process regressor exercises the explicit finite-difference route.
Specialized model backends always take precedence over a requested fallback.

```{literalinclude} ../examples/numerical_fallback.py
:language: python
:lines: 3-
```

## More examples

- [Binary scores and CBaseline](baselines.md).
- [Squared-error and classification loss](loss.md).
- [PyTorch, JAX, and TensorFlow](frameworks.md).
- [All example scripts on GitHub](https://github.com/LudgerHentschel/unifiedig/tree/main/examples).
