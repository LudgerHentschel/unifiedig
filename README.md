# Unified IG

Unified IG provides one small, SHAP-like API for Integrated Gradients across
model families. The first implementation supports closed-form attributions for
scikit-learn `LinearRegression` and binary `LogisticRegression` models.

```python
import unifiedig as uig

explainer = uig.Explainer(model, baseline)
explanation = explainer(X)
```

`Explanation` is lightweight and has SHAP-compatible fields. SHAP remains an
optional dependency; call `explanation.to_shap()` to use its plotting tools.

## Installation

During development, install the project and its test dependencies with:

```console
python -m pip install -e ".[test]"
```

## Output semantics

For regression, attributions sum to the difference between the prediction and
the baseline prediction. Binary logistic regression is explained on its
decision-score (logit) scale; probability attributions are not part of V1.

## Development

Run the tests and validate distribution artifacts with:

```console
pytest
python -m build
python -m twine check dist/*
```

See `CONTRIBUTING.md` for the development workflow.

