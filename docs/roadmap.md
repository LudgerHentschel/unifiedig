# Roadmap

## Current development line — 0.1.1.dev1

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
- Expose TreeIGNumeric as an explicit approximate fallback for CatBoost and
  recognized piecewise-constant tree models, including weighted backgrounds
  and centered multiclass raw scores.
- Explain probability-only sklearn tree classifiers as binary log odds or
  centered multiclass log scores, with an explicit zero-probability floor.
- Document the complete model inventory, output semantics, baseline behavior,
  numerical diagnostics, and SHAP plotting adapter.
- Distinguish vector class scores from explicitly declared multi-output
  regression for generic automatic-gradient models.
- Add native TensorFlow automatic gradients, direct TensorFlow-backed Keras
  models, and backend-native Keras 3 dispatch for JAX and PyTorch.

## JAX backend

Implemented optional JAX support for differentiable prediction functions with:

- parameters supplied explicitly or captured in a closure;
- scalar, multi-output regression, or centered multiclass score output per
  sample;
- native JAX automatic gradients;
- Gauss–Legendre path integration;
- shared baseline distributions;
- dtype and 64-bit-mode diagnostics; and
- completeness, cross-framework, packaging, and optional-dependency tests.

JAX libraries such as Flax, Equinox, NNX, and Haiku are supported through the
lightweight public `JaxModel` adapter rather than separate explainer classes.

## TensorFlow and Keras

Implemented optional TensorFlow support with native `GradientTape`
derivatives, quadrature nodes batched into each model pass, shared weighted
backgrounds, scalar and vector outputs, and an explicit `TensorFlowModel`
adapter for arbitrary prediction functions. TensorFlow-backed Keras models
work directly. Keras 3 models configured for JAX or PyTorch reuse those native
Unified IG backends instead of introducing a Keras-specific differentiation
layer.

## Before the next published release

- Verify `LossExplainer` squared-error and score-based log-loss parity across
  analytic sklearn, exact TreeIG, PyTorch, JAX, TensorFlow/Keras, and explicit
  finite-difference backends.
- Run source-tree and installed-wheel tests for all optional backends.
- Review the README, changelog, error messages, and dependency matrix against
  the packaged artifacts.

## Deferred work

- Multiple-input and structured-output deep-learning models.
- Exact structural CatBoost, probability-averaging forest, and sklearn
  histogram-gradient-boosting support.
- Exact piecewise-linear IG for ReLU networks by detecting activation-region
  transitions along the baseline path.
- Further batching and memory optimization for large smooth models.
- A public third-party backend registry, if model coverage eventually makes
  one useful.
