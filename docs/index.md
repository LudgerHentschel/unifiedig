# UnifiedIG documentation

**Fast Integrated Gradients feature attribution for the most common Python
machine learning models, including tree models, with a familiar API and a
convenient path to SHAP plotting tools.**

UnifiedIG brings three capabilities together:

1. **One familiar API, including trees.** Explain supported linear models,
   pipelines, neural networks, and tree ensembles through the same interface.
   TreeIG brings tree models into the IG framework by accounting for prediction
   jumps at split boundaries.
2. **Fast attribution from model structure.** Native autograd, skgrad's analytic
   derivatives, and TreeIG's exact split-crossing calculations use the information
   each model makes available. Exact shortcuts avoid unnecessary integration;
   specialized gradients avoid expensive numerical differentiation.
3. **Coherent reference distributions.** CBaseline constructs distributions of
   observed inputs localized around a chosen reference prediction and calibrates
   their weighted output to that reference. Explain against a meaningful
   reference population, with every path contributing to the same prediction
   contrast.

Give UnifiedIG a fitted model and a reference background:

```python
import unifiedig as uig

explanation = uig.Explainer(model, background)(X)
```

The result contains feature contributions, baseline outputs, and completeness
diagnostics. [Convert it with `explanation.to_shap()`](plotting.md) to use
SHAP's waterfall, beeswarm, bar, and scatter plots.

Read [how the computation works](how-it-works.md) for the tree and gradient
machinery, and [baselines and CBaseline](baselines.md) for the reference
distribution. The same prediction-attribution interface brings them together.

## Explain your first model

Start with [installation and a complete example](getting-started.md). Then
learn how to choose a [baseline](baselines.md) and
[interpret the feature contributions](explanations.md).

| What you need | Where to go |
|---|---|
| Check whether your model is supported | [Models and backends](supported-models.md) |
| Explain original or transformed pipeline features | [Feature spaces](feature-spaces.md) |
| Use PyTorch, JAX, or TensorFlow | [Framework adapters](frameworks.md) |
| Turn contributions into familiar SHAP plots | [Plot with SHAP](plotting.md) |
| Follow a complete calculation | [Worked examples](examples.md) |
| Assess accuracy or investigate a warning | [Accuracy and troubleshooting](numerical.md) |

The examples are included from executable files and checked during documentation
builds. For more detail, read the [methodology](how-it-works.md),
[output semantics](semantics.md), or [API reference](api.md).

UnifiedIG also supports [loss attribution](loss.md) when observed targets are
available. This additional capability is covered at the end of the user guide.

```{toctree}
:maxdepth: 1
:caption: User guide

getting-started
baselines
explanations
classification
plotting
supported-models
feature-spaces
frameworks
examples
numerical
loss
```

```{toctree}
:maxdepth: 1
:caption: Concepts

how-it-works
semantics
ig-stack
```

```{toctree}
:maxdepth: 1
:caption: Reference and development

api
releases
roadmap
building
releasing
```
