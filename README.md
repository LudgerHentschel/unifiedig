# Unified IG

[![Documentation](https://img.shields.io/badge/docs-user%20guide-blue.svg)](https://ludgerhentschel.github.io/unifiedig/)

[![Tests](https://github.com/LudgerHentschel/unifiedig/actions/workflows/tests.yml/badge.svg)](https://github.com/LudgerHentschel/unifiedig/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/unifiedig.svg)](https://pypi.org/project/unifiedig/)
[![Python versions](https://img.shields.io/pypi/pyversions/unifiedig.svg)](https://pypi.org/project/unifiedig/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Integrated Gradients is the Aumann–Shapley value. UnifiedIG computes it across
the model families used in practice.**

UnifiedIG computes one estimand across linear models, pipelines, SVMs, MLPs,
gradient-boosted trees, random forests, PyTorch, JAX, and TensorFlow. It selects
an exact route wherever model structure permits, avoiding both sampling and
quadrature; reports the completeness residual per sample wherever it does not;
and explains every model against an explicit, auditable reference distribution.

Other attribution libraries typically dispatch to a different algorithm per
model family. In SHAP, tree models use TreeSHAP, neural networks use DeepSHAP,
and the remainder use KernelSHAP. These estimate different quantities under
different assumptions, so attributions from different explainers are not
directly comparable: an apparent disagreement between two models may reflect the
explainers rather than the models.

```python
import unifiedig as uig

explanation = uig.Explainer(model, background)(X)
```

## Discrete and continuous value theory

The Shapley value is the unique attribution satisfying efficiency, symmetry,
dummy, and additivity for cooperative games with a finite player set. Continuous
features are not a finite player set. Applying the discrete theory to them
requires a value function $v(S)$ specifying the model output when a subset of
features is absent — a modeling choice the axioms do not determine, with
conditional and interventional conventions yielding different answers — followed
by an approximation over $2^p$ coalitions.

The corresponding theory for non-atomic games yields the Aumann–Shapley value,
which for differentiable $F$ is the integral of $\nabla F$ along the
straight-line path (Aumann and Shapley, 1974; Sundararajan, Taly and Yan, 2017).
Read distributionally, that integral is defined for piecewise-constant $F$ as
well, where $\nabla F$ carries an impulse at each split boundary
([Hentschel, 2026b](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf)).
Tree ensembles therefore belong inside the same theory rather than requiring a
separate method.

The construction requires a reference distribution and a path. Both are stated
explicitly — the reference constructed from observed data to a chosen output
$f_0$
([Hentschel, 2026a](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf)),
the path fixed by symmetry (Friedman, 2004) — and the cost is linear in the
number of path nodes rather than exponential in the number of features.

Read the **[UnifiedIG documentation](https://ludgerhentschel.github.io/unifiedig/)**
for the user guide, worked examples, and API reference, and
[references and citation](https://ludgerhentschel.github.io/unifiedig/references.html)
for the full bibliography.

## Installation

```console
pip install unifiedig
```

Requires Python 3.10 or newer. CBaseline, skgrad, and TreeIG are installed
alongside UnifiedIG; no separate attribution-backend setup is needed. Install
your model's framework separately when using PyTorch, JAX, TensorFlow, CatBoost,
XGBoost, or LightGBM.

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


## Plot with SHAP

Install the optional plotting dependencies:

```console
pip install "unifiedig[shap]"
```

Convert the result and use familiar SHAP plotting tools:

```python
import shap

plot_values = explanation.to_shap()
shap.plots.waterfall(plot_values[0])
shap.plots.beeswarm(plot_values)
shap.plots.bar(plot_values)
```

Conversion does not rerun the model: the plotted values remain Integrated
Gradients contributions. See the **[plotting guide and gallery](https://ludgerhentschel.github.io/unifiedig/plotting.html)**
for multiclass contrasts, scatter plots, labeling, and saving figures.

## Model coverage and interpretation

The same interface covers supported sklearn linear models, MLPs and pipelines;
selected sklearn, XGBoost and LightGBM trees; and native PyTorch, JAX, and
TensorFlow/Keras models. Explicit numerical fallbacks extend coverage to other
smooth estimators and recognized tree families, including numeric CatBoost.
The **[model/backend matrix](https://ludgerhentschel.github.io/unifiedig/supported-models.html)**
distinguishes exact routes, numerical routes, and their restrictions.

**Classification attributions explain scores, not probabilities.** Binary
outputs use margins or logits; multiclass outputs use centered scores and
support pairwise contrasts. We recommend against probability attribution for
explaining classification decisions because probability links compress and
couple score changes. Read the [classification guide](https://ludgerhentschel.github.io/unifiedig/classification.html)
for the rationale and conventions.

For scalar tabular predictions, `explanation.values` has shape
`(samples, features)`. Adding its feature sum to `explanation.base_values`
reconstructs the prediction. Inspect `explanation.max_abs_completeness_error`
and follow the [accuracy guide](https://ludgerhentschel.github.io/unifiedig/numerical.html)
when using numerical routes.

## Explore the documentation

| Topic | Guide |
|---|---|
| Choose a coherent reference population | [Baselines and CBaseline](https://ludgerhentschel.github.io/unifiedig/baselines.html) |
| Understand the returned values and shapes | [Reading an explanation](https://ludgerhentschel.github.io/unifiedig/explanations.html) |
| Explain original or transformed pipeline features | [Feature spaces](https://ludgerhentschel.github.io/unifiedig/feature-spaces.html) |
| Connect framework models | [Framework adapters](https://ludgerhentschel.github.io/unifiedig/frameworks.html) |
| Follow complete runnable examples | [Worked examples](https://ludgerhentschel.github.io/unifiedig/examples.html) |
| Understand tree paths, gradients, and integration | [How UnifiedIG works](https://ludgerhentschel.github.io/unifiedig/how-it-works.html) |
| Look up parameters and public objects | [API reference](https://ludgerhentschel.github.io/unifiedig/api.html) |

UnifiedIG also offers **[loss attribution](https://ludgerhentschel.github.io/unifiedig/loss.html)**
when observed targets are available. `LossExplainer` explains which features
raise or lower loss relative to the reference, using much of the same path
machinery. It is an additional capability beyond prediction attribution.

## Project information

- [Changelog](CHANGELOG.md) and [roadmap](https://ludgerhentschel.github.io/unifiedig/roadmap.html)
- [Contributing](CONTRIBUTING.md) and [building the documentation](https://ludgerhentschel.github.io/unifiedig/building.html)
- [Issue tracker](https://github.com/LudgerHentschel/unifiedig/issues)
- [BSD-3-Clause license](LICENSE) and [software citation](CITATION.cff)
- [The IG stack: CBaseline, skgrad, TreeIG, and UnifiedIG](https://ludgerhentschel.github.io/unifiedig/ig-stack.html)

Release maintainers: see [Publishing releases](docs/publishing.md).
