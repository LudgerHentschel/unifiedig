# Choosing the attribution feature space

Use `attribute_after` to choose which pipeline representation to explain.
Leave it as `None` (the default) for original pipeline inputs, or name a fitted
preprocessing step for features after that step.

```python
original = uig.Explainer(pipeline, baseline)(X)
standardized = uig.Explainer(pipeline, baseline, attribute_after="scale")(X)
components = uig.Explainer(pipeline, baseline, attribute_after="pca")(X)
```

**Always pass original pipeline observations and original baseline rows.**
UnifiedIG applies the fitted prefix to both, preserves baseline weights, and
integrates the remaining predictor along straight paths in the selected space.
Do not pre-transform either argument when using `attribute_after`.

## Step names and results

Use names from `pipeline.named_steps`. A nested path such as
`attribute_after="preprocess__scale"` selects a step inside a nested Pipeline;
`attribute_after="preprocess"` selects the output of that entire preprocessing
sub-pipeline. Selecting the final predictor or an unknown name raises an error.

- `explanation.values` attributes to the selected feature axes.
- `explanation.data` contains observations in that same space.
- `explanation.feature_names` comes from fitted preprocessing feature-name maps
  (for example `pca0`, `pca1`, or polynomial terms).
- `explanation.attribute_after` records the chosen step, or `None` for original
  inputs. Class-score contrasts preserve it. Conversion to SHAP retains the data
  and feature names; this extra boundary field is specific to UnifiedIG.

Names from DataFrame inputs must match the fitted pipeline column order. Numeric
arrays follow the fitted positional order. For array-fitted pipelines, generated
names start with `x0`, `x1`, and so on. Raw-space behavior remains unchanged.

A scalar baseline is expanded over the original input columns before
transformation. Weighted distributions and CBaseline-style objects exposing
`rows` and `weights` are transformed row by row. The transformed baseline is not
constructed by transforming the original mean; those operations differ for
nonlinear preprocessing.

## Which transformations change the answer?

Featurewise affine scaling (StandardScaler, RobustScaler, MaxAbsScaler, and
unclipped MinMaxScaler) leaves IG contributions unchanged when baselines are
transformed consistently. Gradient units and displayed feature values do change.

PCA and polynomial expansion expose different attribution axes. Nonlinear or
clipped preprocessing can also change the path: a straight line between
transformed endpoints need not be the transformed original straight path.
These are deliberate alternative attribution questions, not equivalent labels
for the same calculation. After feature selection, dropped coordinates no
longer appear; original-space gradients allocate zero to those coordinates.

## Scope and loss attribution

The fitted prefix currently accepts skgrad's supported continuous transforms:
the four scalers, PolynomialFeatures, PCA (including non-degenerate whitening),
and its registered fitted feature selectors. Categorical encoders,
ColumnTransformer, imputers, arbitrary FunctionTransformer functions, and custom
transformer overrides are rejected. No automatic refitting or inverse mapping
is performed. Do not refit the source pipeline while using an explainer.

`explainer.source_model` retains the full original pipeline;
`explainer.model` is the remaining predictor in the selected coordinates.

The remaining predictor is selected through UnifiedIG's NumPy-input backends. This
allows a supported tree backend after selecting a preprocessing boundary, as
well as analytic sklearn gradients or an explicitly requested numerical fallback.
The default original-space route still requires support for the whole pipeline.

`LossExplainer` offers the same argument and uses the same feature-space choice:

```python
losses = uig.LossExplainer(
    pipeline, baseline, attribute_after="scale", loss="squared_error"
)(X, y)
```

The [runnable example](examples.md#pipeline-feature-spaces) compares original,
standardized, and PCA-component attributions and verifies completeness for all
three. This feature requires skgrad 0.1.5 or later.
