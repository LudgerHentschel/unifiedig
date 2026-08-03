# Unified IG semantics

This document fixes the conventions that every backend must follow. The public
API remains independent of model family:

```python
explanation = unifiedig.Explainer(model, baseline)(data)
```

## Inputs and baselines

For sklearn, `data` is one sample with shape `(features,)` or a batch with shape
`(samples, features)`. PyTorch additionally accepts structured single-tensor
inputs with any shape `(samples, ...)`. A baseline may be a scalar, one sample,
or a baseline distribution with shape `(baselines, ...)`. Unified IG—not
individual backends—validates the baseline distribution.

Every input is attributed from the same baseline distribution. Unified IG
averages its Integrated Gradients paths over the distribution; it never infers
row pairing from equal input and baseline counts. Baseline rows receive equal
weight in V1.

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
named output. Probability attribution is intentionally excluded from V1.
Multiclass classification is not yet supported.

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

When explicitly enabled with `fallback="finite_difference"`, Unified IG uses
central finite differences to approximate gradients for otherwise unsupported
smooth sklearn estimators. The step for coordinate `j` is
`finite_difference_step * max(1, abs(x_j))` at each path point. Perturbed rows
are evaluated in bounded batches.

This fallback requires `predict` for regression or `decision_function` for
binary classification. Probability outputs are never inferred. Known tree and
nearest-neighbor estimators are rejected because their local finite-difference
gradients do not represent path discontinuities reliably. A small completeness
residual is an important numerical diagnostic, but it is not a general proof
that a model is smooth or that every individual attribution is accurate.
