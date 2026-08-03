# Roadmap

## Current development line — 0.1.1.dev0

- Delegate smooth sklearn model support and analytic input Jacobians to skgrad.
- Recognize affine regression and binary or multiclass score-classification
  models through skgrad's unified support API.
- Preserve a constant-Jacobian fast path for exact affine attribution.
- Provide an opt-in, batched finite-difference fallback for other smooth
  sklearn estimators.
- Support weighted baseline distributions and CBaseline `Background` objects
  uniformly across all backends.
- Attribute multiclass classifiers as one centered decision-score vector with
  `K - 1` effective dimensions and provide derived pairwise contrasts.
- Document the complete model inventory, output semantics, baseline behavior,
  numerical diagnostics, and SHAP plotting adapter.

## Next model backend — JAX

Add optional JAX support for differentiable batched prediction functions with:

- parameters supplied explicitly or captured in a closure;
- scalar or centered multiclass score output per sample;
- native JAX automatic gradients;
- Gauss–Legendre path integration;
- shared baseline distributions;
- dtype and 64-bit-mode diagnostics; and
- completeness, cross-framework, packaging, and optional-dependency tests.

JAX libraries such as Flax, Equinox, NNX, and Haiku should be supported through
thin prediction-function adapters rather than separate public explainer
classes.

## Before the next published release

- Stabilize the JAX constructor and optional dependency extra.
- Add a JAX example and dedicated GitHub Actions job.
- Run source-tree and installed-wheel tests for all optional backends.
- Review the README, changelog, error messages, and dependency matrix against
  the packaged artifacts.

## Deferred work

- TensorFlow support.
- Exact piecewise-linear IG for ReLU networks by detecting activation-region
  transitions along the baseline path.
- Further batching and memory optimization for large smooth models.
- A public third-party backend registry, if model coverage eventually makes
  one useful.
