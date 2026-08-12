# How UnifiedIG works

UnifiedIG has one organizing principle:

> **Customize the solver, not the estimand.**

Every supported model is explained with the same Integrated Gradients
functional, the same interpretation of a baseline distribution, and the same
output conventions. UnifiedIG changes only the machinery used to evaluate
that functional efficiently.

This separation matters. A model-specific algorithm can be much faster than a
generic implementation without changing what an attribution means. It also
makes attributions comparable across model classes: a linear model, a neural
network, and a tree ensemble are evaluated under the same attribution question.

## The attribution question

For a scalar prediction function `f`, observation `x`, and baseline `x0`, the
attribution to feature `j` is

```text
IG_j(x; x0) = (x_j - x0_j)
               * integral from 0 to 1 of
                 d f(x0 + t(x - x0)) / d x_j dt.
```

The path is the straight line from `x0` to `x`. Attributions decompose the
change in the model output along that path:

```text
sum_j IG_j(x; x0) = f(x) - f(x0).
```

For a weighted baseline distribution with rows `x0_b` and normalized weights
`w_b`, UnifiedIG averages complete path attributions:

```text
value_j(x) = sum_b w_b * IG_j(x; x0_b)
base_value = sum_b w_b * f(x0_b).
```

Consequently,

```text
base_value + sum_j value_j(x) = f(x).
```

Every evaluation observation is compared with every baseline row. Equal input
and baseline row counts never imply row-by-row pairing.

The baseline distribution is therefore part of the estimand, not a tuning
sample used only by a particular backend. Passing the same distribution to
different supported model classes preserves the same reference semantics.

## What remains fixed

UnifiedIG keeps the following choices invariant across model families:

- the straight-line path from each baseline to each observation;
- averaging over the complete weighted baseline distribution;
- the explained model-output scale;
- the completeness identity;
- the shape and meaning of the returned `Explanation`; and
- the interpretation of feature attributions as a continuous path
  decomposition.

These choices do not depend on whether a model happens to expose coefficients,
automatic gradients, polynomial structure, or tree splits.

## What changes internally

The efficient way to calculate the path attribution does depend on model
structure. UnifiedIG dispatches automatically to the strongest supported
route:

| Model structure | Internal calculation | Numerical status |
|---|---|---|
| Affine sklearn model | Constant analytical Jacobian; weighted baselines collapse to their mean | Exact up to floating point |
| Polynomial preprocessing followed by an affine estimator | Analytical chain rule with degree-aware Gauss–Legendre quadrature | Exact at the required polynomial order |
| Supported tree ensemble | TreeIG traces split-boundary events along the path | Exact structural path attribution |
| Supported sklearn MLP | Analytical gradients from skgrad | Controlled quadrature |
| PyTorch model | Native automatic differentiation | Controlled quadrature |
| JAX model | Native automatic differentiation | Controlled quadrature |
| TensorFlow/Keras model | Native automatic differentiation | Controlled quadrature |
| Explicit numerical tree fallback | Adaptive numerical path-event detection | Approximate |
| Explicit finite-difference fallback | Batched numerical gradients and quadrature | Approximate |

The public call remains the same:

```python
explainer = uig.Explainer(model, background)
explanation = explainer(X)
```

Backend classes are private implementation details. Users do not choose an
"IG for trees" or an "IG for neural networks." They choose Integrated
Gradients, and UnifiedIG selects an implementation.

## Why the specialized routes are fast

### Affine models

If `f(x) = beta @ x + c`, its gradient is constant and

```text
value_j(x) = beta_j * (x_j - E[x0_j]).
```

The entire weighted baseline distribution can be replaced by its weighted
mean without approximation. UnifiedIG therefore needs neither multiple path
nodes nor work proportional to the number of baseline rows.

### Polynomial pipelines

Along a straight path, the gradient of a degree-`d` polynomial is a polynomial
of degree at most `d - 1` in the path parameter. Gauss–Legendre quadrature is
exact with `ceil(d / 2)` nodes. UnifiedIG obtains the polynomial powers and
scaling transformations from skgrad and applies the chain rule back to the
original input features.

### Neural networks and other differentiable models

A reverse-mode gradient evaluates derivatives for all input features in one
pass. UnifiedIG batches observations, baselines, and path nodes subject to a
bounded working size. The default 16-node Gauss–Legendre rule is automatically
refined only when the completeness diagnostic requires it.

Feature count does not introduce a coalition-enumeration dimension. Runtime is
driven primarily by model evaluation, the baseline distribution, and the
small quadrature rule.

### Trees

Ordinary local gradients vanish almost everywhere for a piecewise-constant
tree. TreeIG instead finds the split boundaries crossed by the same straight
path and allocates the corresponding output jumps to the features responsible
for those crossings. This is a specialized exact calculation of the same path
functional, not a different attribution game.

TreeIG accepts the same weighted baseline distribution as every other
UnifiedIG backend. Its compiled kernels may have a visible one-time startup
cost, but repeated and large attribution jobs amortize that cost.

## Numerical integration and automatic refinement

Gradient backends use Gauss–Legendre quadrature on the unit path interval.
When `n_steps` is omitted, UnifiedIG:

1. starts with 16 nodes;
2. checks completeness;
3. retries with 32 nodes if the tolerance is not met; and
4. retries with 64 nodes if necessary.

The successful resolution is retained by the explainer for later calls.

```python
explainer = uig.Explainer(model, background)
explanation = explainer(X)
print(explainer.n_steps)
```

Supplying an integer deliberately fixes the resolution and disables automatic
refinement:

```python
explainer = uig.Explainer(model, background, n_steps=128)
```

Exact affine and structural-tree backends do not depend on `n_steps`.
Polynomial pipelines automatically cap excessive resolution at the exact
degree-dependent requirement.

## Completeness is a diagnostic, not a proof

UnifiedIG stores the signed residual

```text
error = model_output - (base_value + sum(feature_attributions)).
```

Automatic refinement fails the check if any residual exceeds

```text
completeness_atol + completeness_rtol * abs(model_output).
```

The defaults are `1e-6` and `1e-4`, respectively. This mixed tolerance remains
meaningful for outputs near zero while scaling with larger predictions.

```python
explanation.completeness_error
explanation.max_abs_completeness_error
```

A small residual establishes that the attributions reconstruct the explained
output. It does not prove that every individual attribution is numerically
accurate: offsetting feature errors can still satisfy completeness. For
high-stakes numerical validation, compare a representative subset with a
higher fixed node count. Numerical tree fallbacks should similarly be checked
at a larger path-grid resolution.

## Output semantics

Attributions are meaningful only after fixing the model output being
decomposed. UnifiedIG uses explicit, consistent conventions:

- regression explains the model prediction;
- binary classification explains a decision score, logit, or raw margin;
- multiclass classification explains the complete centered raw-score vector;
- generic vector-valued automatic-gradient models default to class scores;
  use `output_kind="regression"` for multi-output regression; and
- probabilities are not silently substituted for unavailable scores.

Centered multiclass scores preserve all pairwise margins without selecting an
arbitrary reference class. `Explanation.contrast(a, b)` derives any pairwise
margin attribution without another model or gradient evaluation.

See [semantics.md](semantics.md) for the complete output and array-shape
contract.

## Relationship to SHAP

Integrated Gradients and SHAP answer different questions. IG decomposes the
change along a continuous path from an explicit reference. SHAP allocates
value over discrete feature coalitions under a specified missing-feature game.
Neither definition is universally preferable.

The practical distinction is that the word "SHAP" covers several games and
solver families. Depending on the explainer and its configuration, the
background, dependence treatment, or approximation may change. For example,
interventional TreeSHAP uses an explicit background distribution, while
tree-path-dependent TreeSHAP uses training counts stored in the fitted trees
as an implicit reference distribution.

UnifiedIG takes the opposite design approach: model dispatch may change the
solver but does not change the attribution functional. This is particularly
useful when explanations must be compared across model classes.

UnifiedIG is not an approximation to SHAP. Its `to_shap()` method converts the
result only to reuse SHAP's plotting ecosystem.

## Performance scope

The specialized backends are designed to make the consistent methodology
practical, not merely uniform:

- affine baseline distributions collapse exactly;
- polynomial order determines the exact quadrature requirement;
- native automatic gradients return all feature derivatives together;
- observation, baseline, and path-node dimensions are batched; and
- TreeIG uses structural path information rather than numerical gradients.

Performance comparisons are meaningful only when methods explain the same
observations on the same output scale and use the same explicit background
distribution. A method that changes the reference construction may be faster,
but it is not a matched implementation comparison.

The performance claim intentionally excludes `fallback="finite_difference"`
and `fallback="tree_numeric"`. These routes extend coverage and make their
approximate status explicit; they are not intended to match specialized
backend performance.

## Reproducible use

For comparable explanations across models:

1. explain the same observations;
2. use the same baseline rows and weights;
3. keep the model-output scale fixed;
4. retain the default adaptive quadrature or record an explicit `n_steps`;
5. inspect completeness; and
6. record whether an explicit numerical fallback was requested.

Under those conditions, differences in UnifiedIG attributions reflect
differences in fitted prediction functions rather than changes in attribution
methodology.
