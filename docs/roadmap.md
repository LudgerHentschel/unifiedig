# Roadmap

## 0.1.0.dev1 — correctness and release safety

Completeness diagnostics, broader MLP validation, SHAP plot compatibility,
examples, benchmarks, and guarded releases.

## 0.1.0.dev2 — PyTorch

Optional PyTorch and Captum integration behind the existing `Explainer` API,
with raw scalar-output semantics and structured single-tensor inputs.

## 0.1.0.dev3 — trees

Optional integration with TreeIG for exact path attributions on its supported
sklearn, XGBoost, and LightGBM models. TreeIG owns model support, target-output
semantics, and attribution computation; Unified IG owns baseline normalization
and the common `Explanation` result.

## 0.1.0 — stable V1

Stabilize the public API and documented output semantics.

## Future work

Investigate exact piecewise-linear IG for ReLU networks by detecting activation
region transitions along the baseline path. This is analogous to partitioning a
tree path at decision-boundary crossings and may improve both accuracy and
speed when fixed quadrature needs many nodes.
