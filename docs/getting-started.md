# Getting started

## Install once

UnifiedIG requires Python 3.10 or newer:

```console
python -m pip install unifiedig
```

This also installs CBaseline, skgrad, and TreeIG. You do not select or install
an attribution backend separately. Install your model's framework separately
when using PyTorch, JAX, TensorFlow, CatBoost, XGBoost, or LightGBM. SHAP is
optional and is needed only for the plotting conversion.

## Explain a fitted model

This complete example fits a Ridge model, builds a distribution of observed
baseline rows whose weighted prediction equals the training mean, and explains
five observations. There is one public attribution call for all supported
model families.

```{literalinclude} ../examples/quickstart.py
:language: python
:lines: 3-
```

`explanation.values` holds feature contributions. For this scalar regressor,
its shape is `(5, 4)`. The shared baseline output is repeated in
`explanation.base_values`, which has shape `(5,)`. The assertion checks that
baseline output plus feature contributions reconstructs each prediction.

This example explains training observations to keep setup compact. For model
assessment, explain held-out observations and choose a reference population
that matches the comparison you intend to make.

## Choose the reference deliberately

Pass a vector for one reference observation, a matrix for a shared reference
population, or a CBaseline object for a calibrated weighted distribution:

```python
one_point = uig.Explainer(model, X_train[0])(X_eval)
shared_population = uig.Explainer(model, X_train[:20])(X_eval)
```

A matrix is never interpreted as one baseline per evaluation row. Read
[baselines](baselines.md) before changing the reference distribution.

## Next steps

- [Interpret outputs and class scores](explanations.md).
- [Find a worked example](examples.md) or check [model support](supported-models.md).
- [Check accuracy and unexpected results](numerical.md).

Once you are comfortable explaining predictions, the optional
[loss-attribution chapter](loss.md) shows how to analyze prediction loss when
observed targets are available.
