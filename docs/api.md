---
myst:
  html_meta:
    description: "Look up UnifiedIG Explainer, LossExplainer, Explanation, framework adapters, signatures, and public parameters."
---

# API reference

The public interface consists of two explainers, an explanation container, and
two function adapters. Constructors below are generated from the installed
source, so argument defaults track the implementation.

## Prediction explainer

Call `Explainer(model, baseline, **options)(data)` to obtain an `Explanation`.
The baseline and output contracts are detailed in [baselines](baselines.md)
and [semantics](semantics.md).

| Control | Purpose |
|---|---|
| `baseline_weights` | Relative weights aligned with baseline rows |
| `attribute_after` | Named preprocessing boundary, using original inputs |
| `n_steps` | Fixed quadrature node count, or automatic refinement when omitted |
| `check_completeness`, `completeness_atol`, `completeness_rtol` | Reconstruction checks and automatic refinement trigger |
| `on_incomplete` | `"warn"` (default) emits RuntimeWarning; `"raise"` raises RuntimeError after failed completeness checks and any refinement; applies to both explainers |
| `fallback` | Explicit `finite_difference` or `tree_numeric` route for eligible models |
| `finite_difference_step`, `finite_difference_batch_size` | Central-difference perturbation and batch bound |
| `gradient_batch_size` | Smooth-backend path-row batch bound |
| `tree_grid_size`, `tree_max_refine` | Numerical tree scan and refinement resolution |
| `probability_floor` | Explicit finite-score definition for zero tree probabilities |
| `output_kind` | Framework classification versus multi-output regression semantics |

```{eval-rst}
.. autoclass:: unifiedig.Explainer
   :members: __call__
```

## Loss explainer

```{eval-rst}
.. autoclass:: unifiedig.LossExplainer
   :members: __call__, n_steps
```

## Explanation

```{eval-rst}
.. autoclass:: unifiedig.Explanation
   :members: contrast, to_shap, max_abs_completeness_error
```

## Function adapters

```{eval-rst}
.. autoclass:: unifiedig.JaxModel

.. autoclass:: unifiedig.TensorFlowModel
```
