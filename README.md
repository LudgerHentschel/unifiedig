# Unified IG

[![Tests](https://github.com/LudgerHentschel/unifiedig/actions/workflows/tests.yml/badge.svg)](https://github.com/LudgerHentschel/unifiedig/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/unifiedig.svg)](https://pypi.org/project/unifiedig/)
[![Python versions](https://img.shields.io/pypi/pyversions/unifiedig.svg)](https://pypi.org/project/unifiedig/)
[![License](https://img.shields.io/pypi/l/unifiedig.svg)](LICENSE)

**Integrated Gradients prediction attribution for a broad range of Python
machine-learning models, through one interface.**

> **UnifiedIG customizes the solver, not the estimand.**

Give Unified IG a fitted model and a reference background. Unified IG selects
the appropriate implementation and returns one consistent explanation object.

```python
import unifiedig as uig

explainer = uig.Explainer(model, background)
explanation = explainer(X)
```

That is the standard user experience. There are no model-specific explainer
classes to choose and no gradient backend to configure.

Across linear models, polynomial pipelines, neural networks, and trees, the
attribution definition and baseline semantics stay fixed. UnifiedIG changes
only the internal calculation used to make that definition fast and accurate.

Use UnifiedIG when you want:

- one attribution methodology across different fitted model classes;
- an explicit single baseline or shared weighted baseline distribution;
- exact structural shortcuts and native gradients where they are available;
- automatic numerical accuracy checks without model-specific configuration;
  and
- a small API whose returned values can still use SHAP's plotting ecosystem.

The result contains feature attributions, baseline values, input data, labels,
and a completeness diagnostic:

```python
explanation.values
explanation.base_values
explanation.data
explanation.feature_names
explanation.output_names
explanation.max_abs_completeness_error
```

If you want plots, convert the result and use SHAP's graphing routines:

```python
import shap

shap_values = explanation.to_shap()
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.bar(shap_values)
```

Unified IG deliberately has no plotting subsystem of its own.

## Broad model coverage

The same public API currently covers:

- linear and regularized linear sklearn models;
- binary and multiclass linear classifiers on their decision-score scale;
- sklearn multilayer perceptrons;
- supported sklearn decision trees and ensembles;
- probability-only decision-tree, random-forest, and extra-trees classifiers
  through explicit probability-derived score attribution;
- XGBoost and LightGBM models supported by TreeIG;
- numeric-input CatBoost and other recognized piecewise-constant tree models
  through explicit numerical path-event detection;
- scalar-output, multi-output regression, and class-score PyTorch modules;
- differentiable scalar, multi-output regression, and class-score JAX
  prediction functions, including functions backed by Flax, NNX, Equinox, or
  Haiku models;
- TensorFlow models and Keras 3 models running on TensorFlow, JAX, or PyTorch;
  and
- other smooth sklearn regressors and decision-score classifiers through an
  explicit numerical fallback.

## Documentation

- [How UnifiedIG works](docs/how-it-works.md): the fixed attribution
  functional, specialized solvers, numerical accuracy, performance design,
  and relationship to SHAP.
- [Explanation semantics](docs/semantics.md): baseline distributions, output
  scales, multiclass conventions, array shapes, and completeness.
- [Roadmap](docs/roadmap.md): implemented capabilities and deferred scope.

The README introduces normal use. The linked documents explain the machinery
without expanding the public API.

## Quick start

```python
import numpy as np
from sklearn.linear_model import Ridge

from cbaseline import background
import unifiedig as uig

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 4))
y_train = 2.0 * X_train[:, 0] - X_train[:, 1] + 0.5 * X_train[:, 2]
model = Ridge(alpha=0.5).fit(X_train, y_train)

# Choose a reference prediction and construct observed baseline inputs whose
# weighted mean model prediction equals that reference.
f_train = model.predict(X_train)
f0 = float(f_train.mean())
bg = background(
    predictions=f_train,
    f0=f0,
    features=X_train,
    weighting="calibrated",
)
X_eval = X_train[100:105]

explanation = uig.Explainer(model, bg)(X_eval)

np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    model.predict(X_eval),
)
```

For a pandas `DataFrame`, Unified IG carries column labels into
`explanation.feature_names`.

If one particular input is the intended starting point, pass it directly
instead: `uig.Explainer(model, x0)`.

## Installation

The recommended workflow installs UnifiedIG together with CBaseline:

```console
pip install unifiedig cbaseline
```

The minimal core installation includes NumPy, scikit-learn, skgrad, and the
numerical fallback:

```console
pip install unifiedig
```

Optional capabilities are installed separately:

| Capability | Installation |
|---|---|
| Exact sklearn, XGBoost, and LightGBM tree attribution | `pip install "unifiedig[trees]"` |
| Numerical CatBoost attribution | `pip install "unifiedig[catboost]"` |
| PyTorch attribution through native autograd | `pip install "unifiedig[torch]"` |
| JAX automatic-gradient attribution | `pip install "unifiedig[jax]"` |
| TensorFlow and TensorFlow-backed Keras attribution | `pip install "unifiedig[tensorflow]"` |
| Conversion to `shap.Explanation` | `pip install "unifiedig[shap]"` |
| Prediction-neutral weighted backgrounds | `pip install cbaseline` |
| All Unified IG backend and adapter extras | `pip install "unifiedig[all]"` |

XGBoost and LightGBM models also require their respective model packages.

## One attribution mechanism

Unified IG computes the same quantity for every model family: Integrated
Gradients along the straight-line path from a baseline input `x0` to an
evaluation input `x`.

For feature `j`,

```text
IG_j(x; x0) = (x_j - x0_j)
               * integral from 0 to 1 of
                 d f(x0 + t(x - x0)) / d x_j dt
```

The attribution mechanism does not change across models. What changes is how
Unified IG obtains the path information efficiently and accurately.

## Specialized computation for speed and accuracy

Unified IG automatically selects the strongest available route:

| Route | Gradient or path calculation | Integration | Typical models |
|---|---|---|---|
| Exact affine | Constant analytic Jacobian from skgrad | Closed form | Linear and regularized linear models |
| Exact trees | TreeIG split-boundary traces | Exact | Supported sklearn, XGBoost, and LightGBM trees |
| Differentiable sklearn | Analytic Jacobians from skgrad | Gauss–Legendre quadrature | sklearn MLPs |
| PyTorch | Native automatic gradients | Gauss–Legendre quadrature | Scalar, multi-output regression, and class-score `torch.nn.Module` models |
| JAX | Native automatic gradients | Gauss–Legendre quadrature | Differentiable scalar, multi-output regression, and class-score functions |
| TensorFlow | Native automatic gradients | Gauss–Legendre quadrature | TensorFlow functions and TensorFlow-backed Keras models |
| Numerical trees | TreeIG path-event detection | Approximate crossing search | CatBoost and recognized unsupported piecewise-constant trees |
| Numerical fallback | Batched central finite differences | Gauss–Legendre quadrature | Other smooth sklearn estimators |

Specialized routes always take precedence over the fallback. Users get exact
or high-quality gradients when the model makes them available, without having
to identify the backend themselves.

This is the central design boundary: model-specific code may accelerate the
calculation, but it does not select a different path, background construction,
or attribution game. See [How UnifiedIG works](docs/how-it-works.md) for the
full methodology and performance discussion.

## Baselines and CBaseline

The baseline defines the reference prediction from which the explanation
starts. In most tabular applications, prefer a shared distribution of observed
reference cases over an arbitrary all-zero input. A zero baseline is
appropriate only when zero itself represents the intended comparison.

Unified IG accepts one baseline observation or a shared baseline distribution.
Matrix rows receive equal weight by default:

```python
background = X_train[:100]
explanation = uig.Explainer(model, background)(X_eval)
```

Every evaluation observation is compared with every background row. Equal
input and background row counts never imply row-by-row pairing. Unified IG
averages the paths and baseline outputs over the complete shared distribution.

Supply explicit weights when the reference distribution is not uniform:

```python
explanation = uig.Explainer(
    model,
    background,
    baseline_weights=weights,
)(X_eval)
```

Weights must be finite, nonnegative, and aligned with the background rows.
Unified IG normalizes them to sum to one.

### Constructing baselines from a reference prediction

Often the natural starting point is a prediction `f0`, not a particular input
row. For example, `f0` may be the model's unconditional mean prediction or a
classification decision threshold.

A prediction does not identify a unique input baseline. UnifiedIG therefore
does not silently invert `f0` into a synthetic feature vector. Use
[CBaseline](https://pypi.org/project/cbaseline/) to construct an empirical,
prediction-neutral baseline distribution:

```python
from cbaseline import background

f_train = model.predict(X_train)
bg = background(
    predictions=f_train,
    f0=float(f_train.mean()),
    features=X_train,
    weighting="calibrated",
)

explanation = uig.Explainer(model, bg)(X_eval)
```

Unified IG recognizes CBaseline's `rows` and `weights` properties directly,
without requiring CBaseline as a core dependency. Equal, kernel-weighted, and
calibrated backgrounds therefore use the same explainer call. This produces
attributions relative to the constructed reference distribution while keeping
the background supported by observed data.

Together, CBaseline and UnifiedIG separate two questions cleanly:

1. CBaseline constructs the observed reference distribution that represents
   the desired prediction `f0`.
2. UnifiedIG attributes `f(x) - f0` using the same path functional for every
   supported model class.

For multiclass classification, construct one background for the complete
centered score vector:

```python
scores = model.decision_function(X_train)
centered_scores = scores - scores.mean(axis=1, keepdims=True)

bg = background(
    predictions=centered_scores,
    f0=centered_scores.mean(axis=0),
    features=X_train,
    weighting="calibrated",
)

explanation = uig.Explainer(model, bg)(X_eval)
```

CBaseline detects the redundant common-score direction and constructs the
background in the effective `K - 1` dimensional score space.

## Why Integrated Gradients rather than SHAP attribution?

Integrated Gradients and SHAP answer different attribution questions.

- **IG is path based.** It decomposes the change in model output along a path
  from an explicit reference input or distribution to the observation.
- **SHAP is coalition based.** It attributes output using Shapley-value
  averaging over feature-presence coalitions defined by a background and
  masking rule.
- **IG can exploit gradients.** For differentiable models, one gradient pass
  returns information for every feature. Analytic and automatic gradients can
  therefore be substantially faster than feature-by-feature perturbation.
- **Both are additive explanations.** Unified IG records the observed
  completeness residual and can pass its result to SHAP for plotting.

IG is attractive when the path from a meaningful reference is the scientific
or practical comparison of interest, and when gradients or exact path methods
are available. SHAP remains appropriate when Shapley coalition semantics are
the desired object. Unified IG is not an approximation to SHAP.

## Defaults and available controls

The default choices are designed to make the common case short:

| Choice | Default behavior |
|---|---|
| Backend | Automatically selected from the model |
| Regression output | Model prediction |
| Binary classification output | Positive-class decision score, logit, or raw margin |
| Multiclass classification output | Complete centered decision-score vector |
| Probability attribution | Not offered |
| Baseline matrix | Shared distribution; equally weighted unless weights are supplied |
| Path | Straight line from each baseline to each input |
| Numerical integration | 16-point Gauss–Legendre, automatically refined to 32 or 64 when needed |
| Numerical tree search | 1,024 grid intervals plus four adaptive levels |
| Completeness checking | Enabled |
| Black-box numerical fallback | Disabled unless explicitly requested |
| Vector automatic-gradient output | Class scores; declare `output_kind="regression"` for multi-output regression |

Numerical resolution and diagnostics can be adjusted when necessary:

```python
explainer = uig.Explainer(
    model,
    background,
    n_steps=128,
    gradient_batch_size=8192,
    completeness_atol=1e-6,
    completeness_rtol=1e-4,
    check_completeness=True,
)
```

Exact affine and tree routes ignore `n_steps`. Polynomial pipelines ending in
an affine estimator automatically cap excessive quadrature: a degree-`d`
pipeline needs only `ceil(d / 2)` Gauss–Legendre nodes. A smaller explicitly
requested `n_steps` remains unchanged.

When `n_steps` is omitted, numerical gradient backends start with 16 nodes and
retry with 32, then 64, only when the completeness tolerance is not met. The
successful resolution is retained by the explainer for later calls. Supplying
any integer disables this automatic refinement and uses that requested limit;
disabling completeness checking also disables refinement.

For scalar-output skgrad models, `gradient_batch_size` bounds the number of
baseline-observation path rows processed in one analytic-gradient call. Larger
batches can improve dense MLP throughput at the cost of temporary memory.

## Detailed current coverage

### Exact attribution

| Ecosystem | Supported models | Explained output |
|---|---|---|
| sklearn affine regression | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Prediction |
| sklearn affine classification | Binary and multiclass `LogisticRegression`, `RidgeClassifier` | Margin or centered score vector |
| sklearn trees | `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor` | Prediction |
| sklearn boosted classification | Binary and multiclass `GradientBoostingClassifier` | Margin or centered score vector |
| XGBoost | `XGBRegressor`, binary and multiclass `XGBClassifier`, compatible native `Booster` models | Prediction, margin, or centered score vector |
| LightGBM | `LGBMRegressor`, binary and multiclass `LGBMClassifier`, compatible native `Booster` models | Prediction, margin, or centered score vector |

Tree support is delegated to TreeIG. TreeIG's requirements and exclusions—such
as finite numeric inputs and numeric splits—also apply through Unified IG.

### Numerical tree path detection

For a recognized piecewise-constant tree model without an exact TreeIG parser,
request TreeIG's numerical event detector explicitly:

```python
explainer = uig.Explainer(
    model,
    background,
    fallback="tree_numeric",
    tree_grid_size=1024,
    tree_max_refine=4,
    probability_floor=1e-6,  # needed only if class probabilities can reach zero
)
```

This route scans each baseline-to-input path for output jumps, adaptively
bisects changed intervals, and then probes detected events to identify
responsible features. It supports weighted background distributions and
preserves Unified IG's binary-margin and centered-multiclass score semantics.
`tree_max_refine` defaults to four bisection levels; zero recovers fixed-grid
behavior. CatBoost uses `RawFormulaVal`, not probabilities. Native categorical
inputs are excluded because straight-line interpolation between category codes
is not meaningful; numeric and externally encoded inputs are supported.

Unlike structural TreeIG, event detection is approximate. `tree_grid_size`
sets the number of scanned path intervals and defaults to 1,024. Unified IG's
completeness diagnostic reveals missed endpoint changes, although merged
crossings can still affect feature allocation without producing a residual.
Exact structural backends always take precedence even when this fallback is
requested.

For allocation-sensitive numerical-tree work, rerun a representative subset
with a larger `tree_grid_size` and compare the feature attributions, not only
their completeness residuals. Resolution stability is the relevant check for
nearby or offsetting crossings; adaptive refinement cannot subdivide a coarse
interval whose endpoints have identical outputs.

For `DecisionTreeClassifier`, `RandomForestClassifier`, and
`ExtraTreesClassifier`, which expose probabilities but no native score,
Unified IG transforms the complete model probability vector. Binary models use
`log(p1) - log(p0)`. Multiclass models use
`log(p_k) - mean(log(p))`, retaining all `K` centered coordinates. This is a
transformation of the forest probability after aggregation, not a sum of
separately transformed tree outputs.

This transformation uses the canonical score vector implied by the complete
probability vector: applying softmax to the centered log scores recovers the
original probabilities, and every pairwise score difference is the associated
log odds. It does not claim to recover an unavailable training-time margin;
the derived log-score vector is the explicitly defined model output being
explained.

Tree probabilities can be exactly zero. Unified IG never clips them silently:
if any evaluated path point has zero probability, attribution raises unless
the user supplies `probability_floor`. The floor is applied to every class and
the vector is renormalized before taking logarithms. Consequently, choosing a
floor explicitly defines the finite score object being explained.

### Fast gradients with numerical integration

| Ecosystem | Supported models | Explained output |
|---|---|---|
| sklearn neural networks | Identity-output `MLPRegressor` | Prediction |
| sklearn neural networks | Binary and multiclass `MLPClassifier` | Logit or centered logit vector |
| PyTorch | `torch.nn.Module` with one scalar, multiple regression outputs, or one raw score per class | Model output or centered score vector |
| JAX | Batched differentiable prediction function with one scalar, multiple regression outputs, or one raw score per class | Model output or centered score vector |
| TensorFlow/Keras | Batched model with one scalar, multiple regression outputs, or one raw score per class | Model output or centered score vector |

sklearn MLP hidden activations may be identity, logistic, tanh, or ReLU.
Multi-output MLP regression is supported. A two-score PyTorch output becomes
the single margin `score[1] - score[0]`; three or more scores are centered.
PyTorch inputs may have any single-tensor sample shape; Unified IG preserves
the module's device, floating-point dtype, and prior training/evaluation state.

Vector-valued PyTorch, JAX, and TensorFlow outputs are assumed to be class
scores by default because the frameworks do not identify whether a generic
tensor is a classification score vector or a multi-output regression
prediction. Declare the latter explicitly:

```python
explanation = uig.Explainer(
    model,
    background,
    output_kind="regression",
)(X_eval)
```

Known sklearn and tree estimators expose their task type, so `output_kind` is
unnecessary and rejected for those models.

JAX functions are wrapped explicitly so Unified IG never guesses whether an
arbitrary Python callable is JAX-compatible:

```python
import unifiedig as uig

model = uig.JaxModel(predict_fn, params=params)
explanation = uig.Explainer(model, background)(X_eval)
```

If parameters are captured in a closure, omit `params`. Functions that accept
one sample instead of a batch use `vectorize=True`. Optional `call_kwargs`,
`output_names`, and `dtype` make inference behavior explicit. A Flax
Linen-style model, for example, can use
`uig.JaxModel(model.apply, params=variables)`. Callable NNX and Equinox models
can be wrapped directly. Haiku and stateful framework APIs can be exposed
through a small inference closure.

TensorFlow-backed Keras models work directly:

```python
from tensorflow import keras
import unifiedig as uig

model = keras.Sequential(
    [keras.Input((n_features,)), keras.layers.Dense(1)]
)
explanation = uig.Explainer(model, background)(X_eval)
```

Keras 3 models use their configured native backend. A Keras model running on
the JAX backend uses Unified IG's JAX gradients; one running on the PyTorch
backend uses the native PyTorch route. Arbitrary batched TensorFlow functions
can be selected explicitly with `uig.TensorFlowModel(predict_fn)`, analogous
to `JaxModel`.

Classification models must expose logits or raw scores. Unified IG rejects a
visible final Keras `sigmoid` or `softmax` activation instead of silently
attributing probabilities. Construct an inference model ending at the
pre-activation logits. If a sigmoid or softmax output genuinely represents a
bounded multi-output regression, declare `output_kind="regression"`.

### Opt-in numerical fallback

Otherwise unsupported smooth sklearn estimators can use batched central finite
differences:

```python
explainer = uig.Explainer(
    model,
    background,
    fallback="finite_difference",
    n_steps=32,
)
```

The fallback accepts fitted regressors with `predict` and classifiers whose
`decision_function` returns one binary margin or one score per multiclass
label. It does not infer probability outputs. Pairwise-derived multiclass SVC
scores are rejected. Work grows with the number of inputs, baselines,
quadrature nodes, and features, so this route may be substantially slower than
analytic or automatic gradients.

Advanced numerical controls are available:

```python
explainer = uig.Explainer(
    model,
    background,
    fallback="finite_difference",
    finite_difference_step=1e-5,
    finite_difference_batch_size=8192,
)
```

Finite differences are not a valid substitute for TreeIG on piecewise-constant
models. Local gradients generally miss discontinuous boundary crossings, so
known tree and nearest-neighbor families are rejected rather than assigned
misleading attributions.

## Explanation semantics

For every scalar-output explanation, Unified IG targets

```text
explanation.base_values + explanation.values.sum(over features)
    = explained model output
```

For regression, the explained output is the prediction. For binary
classification, it is the positive-class decision score, logit, or raw margin.

For a `K`-class model with raw scores `z`, Unified IG explains

```text
centered_scores = z - mean(z over classes)
```

The result stores `K` labeled outputs with only `K - 1` effective dimensions:

```text
explanation.values.shape = (samples, features, classes)
sum(explanation.values over classes) = 0
sum(explanation.base_values over classes) = 0
```

Completeness holds independently for every centered score. Pairwise margins
are derived without recomputation:

```python
class_a_vs_b = explanation.contrast("class_a", "class_b")
```

The resulting scalar margin explanation can be plotted directly:

```python
margin_values = class_a_vs_b.to_shap()
shap.plots.beeswarm(margin_values)
shap.plots.waterfall(margin_values[0])
```

| Field | Meaning |
|---|---|
| `values` | Feature attributions; shaped like `data`, with a trailing output axis for multi-output models |
| `base_values` | Mean explained output over the baseline distribution, repeated for each input |
| `data` | Normalized evaluation data |
| `feature_names` | DataFrame column names when available |
| `output_names` | Output labels when available |
| `completeness_error` | Signed output-reconstruction residual |
| `max_abs_completeness_error` | Largest absolute residual |

See [docs/semantics.md](docs/semantics.md) for the complete shape contract.

## Completeness and numerical accuracy

Exact backends generally reach floating-point precision. Quadrature and finite
differences are approximate. Unified IG emits a `RuntimeWarning` when the
completeness residual exceeds the configured tolerance.

The check uses the mixed absolute-relative bound
`completeness_atol + completeness_rtol * abs(model_output)`. The displayed
`max_abs_completeness_error` is the largest absolute residual; it is not itself
a normalized error measure.

Increasing `n_steps` usually reduces quadrature error. For the numerical
fallback, `finite_difference_step` may also matter. A small completeness
residual is an important diagnostic, but it does not prove that an arbitrary
model is smooth or that every individual attribution is accurate.

For affine models, a weighted baseline distribution is collapsed exactly to
its weighted mean. Polynomial and other nonlinear models retain the complete
baseline distribution.

## Current gaps and deferred scope

Notable remaining gaps include exact structural support for CatBoost and
probability-averaging sklearn forests. These are already covered by explicit
numerical path-event detection, but their feature allocation remains
approximate.

Also deferred:

- exact piecewise-linear integration at ReLU activation boundaries;
- multiple-input PyTorch models;
- multiple-input or structured-output TensorFlow/Keras models; and
- a public third-party backend registry.

## Examples

Complete examples are available for:

- [a CBaseline reference prediction](examples/cbaseline_reference_prediction.py)
- [linear regression](examples/linear_regression.py)
- [binary logistic regression](examples/logistic_regression.py)
- [multiclass classification](examples/multiclass_classification.py)
- [sklearn MLP regression](examples/mlp_regression.py)
- [sklearn MLP classification](examples/mlp_classification.py)
- [PyTorch](examples/pytorch.py)
- [JAX](examples/jax_model.py)
- [TensorFlow/Keras](examples/tensorflow_keras.py)
- [the numerical fallback](examples/numerical_fallback.py)
- [numerical tree path detection](examples/numerical_tree.py)
- [probability-only random forest classification](examples/probability_forest.py)

## Development

Install the project with its test dependencies:

```console
python -m pip install -e ".[test]"
```

Run the tests and validate distribution artifacts:

```console
pytest
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[CHANGELOG.md](CHANGELOG.md) for release history.

## Public API

The stable public surface remains deliberately small:

```python
uig.Explainer
uig.Explanation
uig.JaxModel
uig.TensorFlowModel
uig.Explanation.contrast
uig.Explanation.to_shap
```

Model-family backends are private implementation details.
