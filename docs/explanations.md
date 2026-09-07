# Reading an explanation

## Prediction contributions

For a scalar regression prediction, a positive contribution increases the
prediction relative to the chosen baseline distribution; a negative contribution
decreases it. These are contributions to the fitted model's output, not causal
effects or measures of model correctness.

```python
explanation = uig.Explainer(model, background)(X_eval)
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
```

For tabular scalar outputs, `values` has shape `(samples, features)` and
`base_values` has shape `(samples,)`. Multi-output explanations add a trailing
output axis: `(samples, features, outputs)` and `(samples, outputs)`. Framework
inputs can have additional sample dimensions; sum all feature dimensions for
completeness in that case. See [the exact shape contract](semantics.md).

`data` and `feature_names` describe the chosen attribution space. DataFrame
column names are preserved for tabular inputs. `output_names` labels class or
output coordinates; `attribute_after` records a selected pipeline boundary.

## The path attribution in one equation

For an observation $x$, one baseline $b$, and a scalar model output $f$, define
the straight-line path $\gamma(\alpha)=b+\alpha(x-b)$. Feature $j$ receives

$$
\mathrm{IG}_j(x;b)
= (x_j-b_j)\int_0^1
\frac{\partial f}{\partial x_j}\bigl(\gamma(\alpha)\bigr)\,d\alpha.
$$

The input difference multiplies the model's sensitivity to that feature,
averaged along the path. Summing across features reconstructs $f(x)-f(b)$
when the path integral is valid and evaluated accurately. For a baseline
distribution, UnifiedIG averages these complete path attributions using the
baseline weights; the reference output becomes the weighted mean of $f(b)$.

The displayed derivative formula describes smooth model paths (including
piecewise-smooth paths where the usual integration conditions hold). TreeIG
accounts explicitly for split-boundary jumps in discontinuous tree outputs;
ordinary gradients alone would miss them. See [how trees fit the path
calculation](how-it-works.md#why-trees-can-be-included).

The closing [loss chapter](loss.md#nearly-the-same-calculation) uses the same
path and adds the loss derivative inside the integral.

## Classification uses scores

**UnifiedIG attributes classification scores, not probabilities.** We recommend
against probability attribution for explaining classification decisions because
probability links compress and couple score changes. The
[classification guide](classification.md) explains why probability links change
the attribution question, why multiclass scores are centered, and how to choose
a reference on the correct scale.

| Model output | What is reconstructed |
|---|---|
| Binary sklearn classifier | Its decision margin or logit, oriented toward `classes_[1]` |
| Multiclass classifier | Each score minus the mean score across classes |
| Two-score framework model | `score[1] - score[0]` |
| Framework vector with at least three class scores | The centered score vector |
| Probability-only supported tree classifier, explicit numerical fallback | Derived log odds or centered log probabilities |

A positive binary contribution supports the second class on the score scale;
it is not a percentage-point probability change. For generic differentiable
functions, supply raw scores or logits rather than probabilities. The model
matrix describes [restrictions and exceptions](supported-models.md).

Vector framework outputs default to classification. Set
`output_kind="regression"` when they represent multiple regression predictions.
Known sklearn and tree estimators declare their own semantics, so this override
is unnecessary and rejected for those models.

## Compare two classes

A multiclass explanation preserves all centered class coordinates. To explain
one class against another without recomputing paths:

```python
contrast = explanation.contrast(0, 1)
```

Integer arguments select output positions; strings select output names.
The result explains the first score minus the second. Use the
[multiclass example](examples.md#multiclass-classification) for a full calculation.

## Check completeness

Inspect `explanation.max_abs_completeness_error`. A small residual checks that
the sum reconstructs the chosen output. It does not prove individual feature
allocations are accurate, especially for approximate tree crossing detection.
Read [numerical accuracy](numerical.md) before interpreting a warning.

## Plot with SHAP

Install `unifiedig[shap]`, convert the result, and use SHAP's plotting API:

```python
import shap

plot_values = explanation.to_shap()
shap.plots.waterfall(plot_values[0])
shap.plots.beeswarm(plot_values)
```

Follow [Plot with SHAP](plotting.md) for a complete example, a visual gallery,
plot selection, multiclass slicing, and saving labeled figures.

This snippet assumes a scalar tabular explanation. For multiclass results,
convert a pairwise `contrast` first. Conversion supplies a plotting container;
it does not turn Integrated Gradients into Shapley values.

## A prediction explanation does not measure prediction quality

A positive prediction contribution tells you that the feature raises the
explained output relative to the reference. Whether that helps depends on the
observed target and the loss function. The closing [loss-attribution chapter](loss.md)
uses the same reference paths to answer that second question, explaining the
model composed with the loss while holding each target fixed.
