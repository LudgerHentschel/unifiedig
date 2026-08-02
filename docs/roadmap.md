# Roadmap

## 0.1.0.dev1 — correctness and release safety

Completeness diagnostics, broader MLP validation, SHAP plot compatibility,
examples, benchmarks, and guarded releases.

## 0.1.0.dev2 — PyTorch

Optional PyTorch and Captum integration behind the existing `Explainer` API.

## 0.1.0.dev3 — trees

Integrated Gradients for supported sklearn tree ensembles, with exact path
handling where practical.

## 0.1.0 — stable V1

Stabilize the public API and documented output semantics.

## Future work

Investigate exact piecewise-linear IG for ReLU networks by detecting activation
region transitions along the baseline path. This is analogous to partitioning a
tree path at decision-boundary crossings and may improve both accuracy and
speed when fixed quadrature needs many nodes.
