# Unified IG semantics

This document fixes the conventions that every backend must follow. The public
API remains independent of model family:

```python
explanation = unifiedig.Explainer(model, baseline)(data)
```

## Inputs and baselines

`data` is one sample with shape `(features,)` or a batch with shape
`(samples, features)`. Unified IG always stores it as a two-dimensional float
array. A baseline may be a scalar, a feature vector, one matrix row, or one row
per input sample. Unified IG—not individual backends—validates and broadcasts
the baseline to match the input matrix.

Integrated Gradients follows the straight-line path from each normalized
baseline row to its corresponding data row.

## Explanation arrays

For a scalar model output:

- `values`: `(samples, features)`
- `base_values`: `(samples,)`
- `data`: `(samples, features)`

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
