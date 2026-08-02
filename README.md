# Unified IG

Unified IG provides one small, SHAP-like API for Integrated Gradients across
model families. The first implementation supports closed-form attributions for
scikit-learn linear models and multilayer perceptrons.

```python
import unifiedig as uig

explainer = uig.Explainer(model, baseline)
explanation = explainer(X)
```

For numerical backends, the quadrature resolution is configurable:

```python
explainer = uig.Explainer(model, baseline, n_steps=128)
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
the baseline prediction. Binary classifiers are explained on their
decision-score (logit) scale; probability attributions are not part of V1.
See `docs/semantics.md` for the complete array-shape and output contract.

Numerical explanations expose their observed completeness residual:

```python
explanation.completeness_error
explanation.max_abs_completeness_error
```

Unified IG warns when this error exceeds the configured tolerance. Increasing
`n_steps` usually improves it. See the `examples/` directory for complete
linear, logistic, MLP regression, and MLP classification programs.

## Supported models

- `sklearn.linear_model.LinearRegression` (closed form)
- Binary `sklearn.linear_model.LogisticRegression` (closed form)
- `sklearn.neural_network.MLPRegressor` (analytic gradients and quadrature)
- Binary `sklearn.neural_network.MLPClassifier` (analytic logit gradients and quadrature)

## Development

Run the tests and validate distribution artifacts with:

```console
pytest
python -m build
python -m twine check dist/*
```

See `CONTRIBUTING.md` for the development workflow.
