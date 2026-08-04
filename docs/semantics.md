# Unified IG semantics

This document fixes the conventions that every backend must follow. The public
API remains independent of model family:

```python
explanation = unifiedig.Explainer(model, baseline)(data)
```

## Inputs and baselines

For sklearn, `data` is one sample with shape `(features,)` or a batch with shape
`(samples, features)`. PyTorch and JAX additionally accept structured
single-array inputs with any shape `(samples, ...)`. A baseline may be a
scalar, one sample, or a baseline distribution with shape `(baselines, ...)`. Optional
`baseline_weights` must align with its rows. A background object exposing
`rows` and `weights`, including a CBaseline `Background`, may be passed
directly. Unified IG—not individual backends—validates and normalizes the
distribution.

Every input is attributed from the same baseline distribution. Unified IG
averages its Integrated Gradients paths over the distribution; it never infers
row pairing from equal input and baseline counts. Matrix rows receive equal
weight by default. Explicit weights must be finite and nonnegative with a
positive sum; Unified IG normalizes them to sum to one.

For normalized weights `w_b`, the explanation averages complete paths:

```text
values = sum_b w_b * IG(data; baseline_b)
base_values = sum_b w_b * model_output(baseline_b)
```

Passing a background object and also supplying `baseline_weights` is rejected
so that there is only one source of weighting semantics.

## Explanation arrays

For a scalar model output:

- `values`: the same shape as `data`
- `base_values`: `(samples,)`
- `data`: `(samples, ...)`

For multiple model outputs:

- `values`: `(samples, features, outputs)`
- `base_values`: `(samples, outputs)`
- `data`: `(samples, features)`

This orientation matches SHAP's current multi-output convention. Output names,
when available, identify the final axis.

## Explained output

Regression backends explain the model prediction. Binary classification
backends explain the decision score (logit), with the positive class as the
named output. If a model returns two raw binary scores, Unified IG explains
their difference, `score[1] - score[0]`.

For multiclass classification, let `z(x)` be the model's vector of `K` raw
class scores. Unified IG explains the centered score vector

```text
s(x) = z(x) - mean(z(x) over classes).
```

The `K` labeled coordinates sum to zero and represent a `K - 1` dimensional
decision-score object. This removes the common-score direction without
choosing an arbitrary reference class. Completeness holds separately for every
centered score:

```text
sum_j values[i, j, k] + base_values[i, k] = s_k(data[i]).
```

The stronger zero-sum identities also hold up to floating-point error:

```text
sum_k values[i, j, k] = 0
sum_k base_values[i, k] = 0.
```

`Explanation.contrast(a, b)` subtracts two stored coordinates to recover IG
for the invariant pairwise margin `z_a - z_b`, without recomputing gradients
or paths. Independent target-class attribution is not a separate Unified IG
estimand. Probability attribution is intentionally excluded.

## Completeness

Every explanation targets:

```text
sum(values over features) + base_values = explained model output
```

Closed-form backends satisfy this up to floating-point arithmetic. Numerical
backends approximate the path integral and document their integration method;
their completeness tests use an explicit numerical tolerance.

Supported tree models are delegated to TreeIG, which computes their path
attributions exactly and applies the same shared-distribution semantics.

`Explanation.completeness_error` stores the signed residual between the model
output and the reconstructed output. `max_abs_completeness_error` summarizes
the worst sample/output. Numerical backends emit a `RuntimeWarning` when this
exceeds the configured absolute and relative tolerances. The check can be
configured with `completeness_atol`, `completeness_rtol`, and
`check_completeness` on `Explainer`.

## Numerical integration

Numerical backends use Gauss–Legendre quadrature on the unit path interval.
`Explainer(..., n_steps=N)` controls the number of quadrature nodes. More nodes
usually improve accuracy but require proportionally more gradient evaluations.
The default is 64.

PyTorch uses Captum's Gauss–Legendre implementation. JAX evaluates native
automatic gradients at the same quadrature nodes. A JAX prediction function
must produce samplewise outputs: one scalar or one raw class-score vector for
each leading input row. Two class scores are reduced to their margin, and
three or more are centered under the multiclass convention above.

When explicitly enabled with `fallback="finite_difference"`, Unified IG uses
central finite differences to approximate gradients for otherwise unsupported
smooth sklearn estimators. The step for coordinate `j` is
`finite_difference_step * max(1, abs(x_j))` at each path point. Perturbed rows
are evaluated in bounded batches.

This fallback requires `predict` for regression or `decision_function` for
classification. Multiclass decision functions must return one score per class;
pairwise-derived SVC scores are rejected. Probability outputs are never
inferred. Known tree and nearest-neighbor estimators are rejected because their
local finite-difference gradients do not represent path discontinuities
reliably. A small completeness residual is an important numerical diagnostic,
but it is not a general proof that a model is smooth or that every individual
attribution is accurate.
