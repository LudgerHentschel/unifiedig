# Accuracy and troubleshooting

## Automatic integration

Exact affine and structural tree routes do not depend on quadrature resolution.
Smooth numerical routes normally start with 16 Gauss–Legendre nodes. With
completeness checking enabled, UnifiedIG can retry at 32 and 64 nodes when the
residual exceeds the configured tolerance. Polynomial structure can reduce the
required node count. [Loss attribution](loss.md) has additional affine shortcuts.

An explicit `n_steps` fixes the requested resolution and disables automatic
refinement. Disabling completeness checking also disables that refinement.
`explainer.n_steps` reports the current numerical resolution.

```python
explainer = uig.Explainer(
    model, background, n_steps=128,
    completeness_atol=1e-6, completeness_rtol=1e-4,
)
result = explainer(X_eval)
```

## Interpret the diagnostic

A small completeness residual is necessary for a well-resolved path
calculation, but feature-level errors can cancel. Check stability of the
feature values when changing numerical resolution, particularly for nonlinear
models and numerical tree detection. Tightening tolerances alone does not make
a fixed computation more accurate.

| Symptom | What to check |
|---|---|
| No backend supports the model | Confirm the exact estimator and output configuration in the model matrix; a fitted model is required |
| Smooth model requires an explicit fallback | Use `fallback="finite_difference"` only for a suitable smooth sklearn estimator |
| Classifier exposes only probabilities | Generic finite differences require decision scores; supported probability-only trees have their separate explicit tree route |
| Numerical-tree warning | Increase `tree_grid_size` and compare feature values, not only completeness; missed offsetting crossings may leave zero residual |
| Zero class probability on a tree path | Choose an explicit `probability_floor` if finite derived log scores are the desired output |
| Unexpected class/output axis | Check `output_names`, centered-score semantics, and `output_kind` for framework vectors |
| Baseline shape or weight error | Align feature dimensions and row weights; do not supply weights twice for a CBaseline object |
| Framework residual plateau | Check dtype, differentiability, nonsmooth points, and deterministic inference; increasing nodes may not overcome float32 limits |
| Feature names differ after a pipeline boundary | The result describes the selected transformed space; supply original data and baselines |

The [model matrix](supported-models.md) defines which fallbacks apply. The
[methodology](how-it-works.md) explains why numerical tree search can miss
crossings even when completeness holds.

## Performance

Work usually grows with evaluation rows, baseline rows, path nodes, and output
dimensions. Keep the reference population meaningful and manageable, evaluate
observations in batches when necessary, and use `gradient_batch_size` to bound
smooth-backend path work. The finite-difference backend has its own
`finite_difference_batch_size` and `finite_difference_step` controls.

Exact affine paths can collapse a baseline distribution to its weighted input
mean without changing results. Do not apply this shortcut yourself to general
nonlinear models. See [baselines](baselines.md).
