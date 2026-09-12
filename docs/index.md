---
myst:
  html_meta:
    description: "UnifiedIG computes Integrated Gradients feature attributions across supported Python model families with explicit baselines and completeness diagnostics."
---

# UnifiedIG documentation

UnifiedIG is a Python package for Integrated Gradients feature attribution
across supported scikit-learn, tree, PyTorch, JAX, and TensorFlow/Keras models.
Install and import it as `unifiedig`. Given a fitted model, a reference baseline
or background distribution, and observations, it returns feature contributions,
baseline outputs, and completeness diagnostics.

**Integrated Gradients is the Aumann–Shapley value. UnifiedIG computes it across
the model families used in practice.**

UnifiedIG computes one estimand across linear models, pipelines, SVMs, MLPs,
gradient-boosted trees, random forests, PyTorch, JAX, and TensorFlow. It selects
an exact route wherever model structure permits, avoiding both sampling and
quadrature; reports the completeness residual per sample wherever it does not;
and explains every model against an explicit, auditable reference distribution.

Other attribution libraries typically dispatch to a different algorithm per
model family. In SHAP, tree models use TreeSHAP, neural networks use DeepSHAP,
and the remainder use KernelSHAP. These estimate different quantities under
different assumptions, so attributions from different explainers are not
directly comparable: an apparent disagreement between two models may reflect the
explainers rather than the models.

Give UnifiedIG a fitted model and a reference background:

```python
import unifiedig as uig

explanation = uig.Explainer(model, background)(X)
```

Check the [model/backend matrix](supported-models.md) before choosing a
route. Classification explains scores or logits; multiclass scores are centered.
Exactness depends on model structure, and numerical fallbacks require explicit
opt-in. See [classification conventions](classification.md) and
[accuracy checks](numerical.md). A small completeness residual checks
reconstruction; it does not by itself establish accurate individual allocations.

The result contains feature contributions, baseline outputs, and completeness
diagnostics. [Convert it with `explanation.to_shap()`](plotting.md) to use
SHAP's waterfall, beeswarm, bar, and scatter plots.

Read [how the computation works](how-it-works.md) for the tree and gradient
machinery, and [baselines and CBaseline](baselines.md) for the reference
distribution. The same prediction-attribution interface brings them together.

## Discrete and continuous value theory

The Shapley value is the unique attribution satisfying efficiency, symmetry,
dummy, and additivity for cooperative games with a finite player set. Continuous
features are not a finite player set. Applying the discrete theory to them
requires a value function $v(S)$ specifying the model output when a subset of
features is absent — a modeling choice the axioms do not determine, with
conditional and interventional conventions yielding different answers — followed
by an approximation over $2^p$ coalitions.

The corresponding theory for non-atomic games yields the Aumann–Shapley value,
which for differentiable $F$ is the integral of $\nabla F$ along the
straight-line path (Aumann and Shapley, 1974; Sundararajan, Taly and Yan, 2017).
Read distributionally, that integral is defined for piecewise-constant $F$ as
well, where $\nabla F$ carries an impulse at each split boundary
([Hentschel, 2026b](references.md)).
Tree ensembles therefore belong inside the same theory rather than requiring a
separate method.

The construction requires a reference distribution and a path. Both are stated
explicitly — the reference constructed from observed data to a chosen output
$f_0$ ([Hentschel, 2026a](references.md)), the path fixed by symmetry
(Friedman, 2004) — and the cost is linear in the number of path nodes rather
than exponential in the number of features.

## Related packages

UnifiedIG combines [CBaseline](https://ludgerhentschel.github.io/cbaseline/)
reference distributions, [TreeIG](https://ludgerhentschel.github.io/treeig/)
tree-path attributions, and [skgrad](https://ludgerhentschel.github.io/skgrad/)
analytic input derivatives. Use these packages directly when you need their
individual capabilities; use UnifiedIG for the common attribution interface.
See [the Integrated Gradients stack](ig-stack.md) for their relationships.

For automated readers, [llms.txt](https://ludgerhentschel.github.io/unifiedig/llms.txt)
links to the guides, complete examples, and rendered API reference.

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
references
releases
roadmap
building
publishing
releasing
```
