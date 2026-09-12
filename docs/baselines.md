---
myst:
  html_meta:
    description: "Choose a reference point or weighted background distribution for UnifiedIG and preserve CBaseline rows, weights, and output scales."
---

# Baselines and CBaseline

A baseline defines the comparison in an attribution. Changing it changes the
question, even when the model and observation stay fixed.

## A reference population for a specified prediction

CBaseline separates two decisions: **which prediction defines the reference**,
and **which observed cases form a relevant population around it**. Choose a
reference output `f0`, localize observed rows near it in prediction space, and
calibrate their weights so the mean output matches `f0` to numerical tolerance.
UnifiedIG then averages the complete IG paths from those rows.

This gives the explanation a coherent total, `f(x) - f0`, and an explicit
reference population for allocating that total to features. Several input rows
can represent the reference, retaining variation among observed cases instead
of making the entire explanation depend on one selected point.

## Why localization and calibration both matter

Localization selects rows whose **predictions** are near the reference. It does
not require proximity to the evaluation observation in input space. Calibration
then constrains the weighted mean output. Matching the mean alone is weaker:
very high and very low predictions can average to `f0` while representing a
broad, heterogeneous reference population.

| Reference choice | What it supplies | What it does not enforce |
|---|---|---|
| A single reference point | One definite starting input and prediction | Variation among reference cases or a specified population-level output |
| A full background sample or random subset | A distribution of reference cases | Localization near a chosen `f0`, or calibration to it |
| A calibrated CBaseline distribution | Observed rows localized near `f0`, with a weighted mean output matching it | Exact neutrality of every row or proximity in input space |

A single point remains appropriate when that particular input is the intended
comparison. A broad background is appropriate when the intended question is
relative to that full population. CBaseline is designed for the more specific
question: what changes the prediction from a localized reference population
centered on a chosen output?

Broad background samples are common in SHAP workflows, but localization is a
property of background construction, not an inherent limitation of SHAP.
CBaseline can also construct backgrounds for SHAP. UnifiedIG makes localized,
calibrated distributions a direct part of the IG workflow through its weighted
baseline interface. [CBaseline's guide](https://ludgerhentschel.github.io/cbaseline/concepts.html)
explains the reference-distribution construction in more detail.

Calibration may need to widen the localized support or reject an infeasible
reference. Inspect its diagnostics: a weighted mean of `f0` does not imply
that every retained row predicts exactly `f0`. Observed baseline endpoints
also do not guarantee that every interpolated point along an IG path is an
observed or plausible input.

## Point, population, or calibrated distribution

| Input to `Explainer` | Meaning |
|---|---|
| One feature vector | Compare every observation with this single reference input |
| A matrix of reference rows | Average each observation's paths from all rows, with equal weights |
| Matrix plus `baseline_weights` | Average paths using the supplied normalized weights |
| CBaseline background object | Use its aligned `rows` and `weights` directly |

A scalar baseline broadcasts across input features. Use it only when that
constant has a meaningful interpretation. A matrix remains a shared
population even if it has exactly as many rows as the evaluation batch.

```python
explanation = uig.Explainer(
    model, reference_rows, baseline_weights=[0.2, 0.3, 0.5]
)(X_eval)
```

Weights must be finite, nonnegative, have a positive sum, and align with the
reference rows. UnifiedIG normalizes them. Do not also pass `baseline_weights`
when supplying a background object that already carries weights.

For a scalar output, the baseline value is the weighted average of model
outputs on the reference rows. It is generally **not** the model output at the
weighted mean input. Nonlinear models require averaging the paths themselves.

## Choose a reference prediction with CBaseline

Sometimes it is easier to specify an output reference, such as the average
prediction of a population. CBaseline constructs a distribution of observed
inputs whose weighted output matches that reference. There is no unique
inverse input corresponding to an output value.

The [getting-started example](getting-started.md) uses a regression prediction.
For classification, [attribute scores, not probabilities](classification.md),
and construct the background on the same score scale that
UnifiedIG explains. This binary example uses `decision_function`:

```{literalinclude} ../examples/logistic_regression.py
:language: python
:lines: 3-
```

For multiclass models use the complete centered score vector, as in the
[multiclass example](examples.md#multiclass-classification). A requested
reference must be feasible for the observed prediction support; inspect
CBaseline's diagnostics and its [reference-prediction guide](https://ludgerhentschel.github.io/cbaseline/reference-predictions.html).

## Practical choices

Use reference cases representing the population or decision threshold relevant
to your analysis. Keep that choice fixed when comparing explanations across
models, and record any calibration weights. For pipelines, supply original
input rows even when attributing in a [transformed feature space](feature-spaces.md).

Larger reference populations increase work because every evaluation point is
compared with every reference row. Start with a substantively meaningful,
manageable population and examine whether changing it alters your conclusions.
