# Unified IG

Unified IG provides one small, SHAP-like API for Integrated Gradients across
linear models, neural networks, and tree ensembles.

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

Install the core sklearn support from PyPI:

```console
pip install unifiedig
```

Install every optional backend and the SHAP adapter with:

```console
pip install "unifiedig[all]"
```

During development, install the project and its test dependencies with:

```console
python -m pip install -e ".[test]"
```

## Output semantics

For regression, attributions sum to the difference between the prediction and
the baseline prediction. Binary classifiers are explained on their
decision-score (logit) scale; probability attributions are not part of V1.
See `docs/semantics.md` for the complete array-shape and output contract.

A baseline matrix is an equally weighted distribution shared by every input:

```python
background = X_train[:100]
explanation = uig.Explainer(model, background)(X_test)
```

Unified IG averages the attribution paths and model output over all background
rows. It does not pair background row `i` with input row `i`.

Numerical explanations expose their observed completeness residual:

```python
explanation.completeness_error
explanation.max_abs_completeness_error
```

Unified IG warns when this error exceeds the configured tolerance. Increasing
`n_steps` usually improves it. See the `examples/` directory for complete
linear, logistic, MLP regression, and MLP classification programs.

## Numerical fallback

An otherwise unsupported smooth sklearn estimator can be explained with
batched central finite differences:

```python
explainer = uig.Explainer(
    model,
    baseline,
    fallback="finite_difference",
)
explanation = explainer(X)
```

Specialized backends always take precedence. The fallback supports fitted
sklearn regressors with `predict` and binary classifiers with
`decision_function`; it never substitutes probability outputs. It can be much
slower because its work grows with the number of features, quadrature nodes,
inputs, and baselines. `finite_difference_step` controls the relative central
difference step, and `finite_difference_batch_size` bounds the number of
perturbed rows evaluated together.

Finite differences are inappropriate for piecewise-constant models because
local gradients generally miss their discontinuities. Known tree and
nearest-neighbor families are therefore rejected rather than given misleading
attributions. Completeness diagnostics should be inspected carefully for every
fallback result.

## Supported models

Unified IG currently recognizes the following fitted estimators:

| Model family | Supported estimators | Explained output | Method |
|---|---|---|---|
| sklearn affine regression | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Prediction | Exact closed form via skgrad |
| sklearn affine classification | Binary `LogisticRegression`, `RidgeClassifier` | Decision score | Exact closed form via skgrad |
| sklearn neural networks | Identity-output `MLPRegressor` | Prediction | Analytic skgrad Jacobians with Gauss–Legendre quadrature |
| sklearn neural networks | Binary `MLPClassifier` | Pre-probability logit | Analytic skgrad Jacobians with Gauss–Legendre quadrature |
| sklearn trees | `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor` | Prediction | Exact via TreeIG |
| sklearn boosted trees | Binary `GradientBoostingClassifier` | Decision score | Exact via TreeIG |
| XGBoost | `XGBRegressor`, binary `XGBClassifier`, and compatible native `Booster` models | Prediction or raw margin | Exact via TreeIG |
| LightGBM | `LGBMRegressor`, binary `LGBMClassifier`, and compatible native `Booster` models | Prediction or raw score | Exact via TreeIG |
| PyTorch | `torch.nn.Module` with one raw scalar output per sample | Model output | Captum Gauss–Legendre IG |
| Other smooth sklearn estimators | Regressors with `predict`; binary classifiers with `decision_function` | Prediction or decision score | Opt-in finite differences and Gauss–Legendre quadrature |

Specialized smooth sklearn models are recognized through the single
`skgrad.supports()` predicate, while tree models are recognized through
`treeig.supports()`. This keeps estimator registries in their owning packages
so additions can flow into Unified IG without duplicating model lists in its
dispatch layer. skgrad's constant-Jacobian metadata preserves the exact affine
fast path without exposing separate model-family support predicates.

V1 intentionally rejects multiclass classifiers because `Explainer` does not
yet expose an output target. Multi-output regressors are supported when their
output follows the documented array contract. MLP hidden activations may be
identity, logistic, tanh, or ReLU;
`MLPRegressor` models with a non-identity output activation are rejected.
TreeIG currently requires finite numeric inputs and numeric splits; its other
documented exclusions also apply through Unified IG.

Install PyTorch support separately so the core package remains lightweight:

```console
pip install "unifiedig[torch]"
```

Install exact tree-model support separately:

```console
pip install "unifiedig[trees]"
```

XGBoost and LightGBM models additionally require their respective packages.

Tree attributions are computed by TreeIG; Unified IG normalizes the input and
baseline and adapts TreeIG's exact result to `Explanation`.

PyTorch models may accept tabular or structured single-tensor inputs. Unified IG
preserves the model's device and floating-point dtype, temporarily evaluates the
model in inference mode, and restores every module's prior training state. V1
expects one raw scalar output per sample. For binary classification that output
must be the logit, not a sigmoid probability.

## Development

Run the tests and validate distribution artifacts with:

```console
pytest
python -m build
python -m twine check dist/*
```

See `CONTRIBUTING.md` for the development workflow.

## V1 public API

The stable V1 surface is deliberately small: `Explainer`, `Explanation`, and
`Explanation.to_shap()`. Unified IG has no plotting API and SHAP is not a core
dependency.
