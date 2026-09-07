# Unified IG semantics

This document fixes the conventions that every backend must follow. The public
API remains independent of model family:

```python
explanation = unifiedig.Explainer(model, baseline)(data)
```

## Inputs and baselines

For sklearn, `data` is one sample with shape `(features,)` or a batch with shape
`(samples, features)`. PyTorch, JAX, and TensorFlow additionally accept
structured single-array inputs with any shape `(samples, ...)`. A baseline may
be a scalar, one sample, or a baseline distribution with shape
`(baselines, ...)`. Optional `baseline_weights` must align with its rows. A
background object exposing `rows` and `weights`, including a CBaseline
`Background`, may be passed directly. Unified IG—not individual backends—
validates and normalizes the distribution.

Every input is attributed from the same baseline distribution. Unified IG
averages its Integrated Gradients paths over the distribution; it never infers
row pairing from equal input and baseline counts. Matrix rows receive equal
weight by default. Explicit weights must be finite and nonnegative with a
positive sum; Unified IG normalizes them to sum to one.

For normalized weights `w_b`, the explanation averages complete paths:

```text
values = sum_b w_b * IG(data; baseline_b)
base_values = sum_b w_b * model_output(baseline_b)
```

Passing a background object and also supplying `baseline_weights` is rejected
so that there is only one source of weighting semantics.

## Explanation arrays

For a scalar model output:

- `values`: the same shape as `data`
- `base_values`: `(samples,)`
- `data`: `(samples, ...)`

For multiple model outputs, including structured single-array inputs:

- `values`: `data.shape + (outputs,)`
- `base_values`: `(samples, outputs)`
- `data`: `(samples, ...)`

This orientation matches SHAP's current multi-output convention. Output names,
when available, identify the final axis.

## Explained output

**Classification completeness is on the score scale, never the probability
scale.** A decision margin is a logit only when the model defines it as one.
See [classification scores](classification.md) for the rationale and the
distinction from the mathematically different probability-IG question.

Regression backends explain the model prediction. Binary classification
backends explain a decision margin or logit, with the positive class as the
named output. If a model returns two raw binary scores, Unified IG explains
their difference, `score[1] - score[0]`.

For multiclass classification, let `z(x)` be the model's vector of `K` raw
class scores. Unified IG explains the centered score vector

```text
s(x) = z(x) - mean(z(x) over classes).
```

The `K` labeled coordinates sum to zero and represent a `K - 1` dimensional
decision-score object. This removes the common-score direction without
choosing an arbitrary reference class. Completeness holds separately for every
centered score:

```text
sum_j values[i, j, k] + base_values[i, k] = s_k(data[i]).
```

The stronger zero-sum identities also hold up to floating-point error:

```text
sum_k values[i, j, k] = 0
sum_k base_values[i, k] = 0.
```

`Explanation.contrast(a, b)` subtracts two stored coordinates to recover IG
for the invariant pairwise margin `z_a - z_b`, without recomputing gradients
or paths. Independent target-class attribution is not a separate Unified IG
estimand. Probability attribution is intentionally excluded.

Generic differentiable frameworks do not encode whether a vector output is a
class-score vector or a multi-output regression prediction. Unified IG treats
vector-valued PyTorch, JAX, and TensorFlow outputs as class scores by default.
Pass `output_kind="regression"` to preserve independent regression outputs
without binary differencing or multiclass centering. Known sklearn and tree
estimators declare their task type and do not use this option.

Keras 3 models use their configured native automatic-gradient backend.
Visible final sigmoid and softmax activations are rejected for classification:
the explained output must be a logit or raw score, not a probability. An
explicit `output_kind="regression"` permits these activations only when the
output is genuinely a bounded regression prediction.

## Completeness

Every explanation targets:

```text
sum(values over features) + base_values = explained model output
```

Closed-form backends satisfy this up to floating-point arithmetic. Numerical
backends approximate the path integral and document their integration method;
their completeness tests use an explicit numerical tolerance.

Supported tree models are delegated to TreeIG, which computes their path
attributions exactly and applies the same shared-distribution semantics.
When explicitly selected with `fallback="tree_numeric"`, TreeIGNumeric instead
searches for discontinuities on a finite path grid. Detected jumps are
allocated locally and averaged over the same weighted baseline distribution.
This route is complete when it recovers all endpoint changes, but its feature
allocation remains approximate when crossings are missed or merged.

For a classifier with probabilities but no native decision score, the
numerical-tree route explains derived scores. With class probabilities `p`,
binary classification uses

```text
score = log(p_1) - log(p_0),
```

and multiclass classification uses

```text
score_k = log(p_k) - mean(log(p) over classes).
```

The multiclass object is therefore centered and pairwise contrasts are log
odds. Because tree probabilities may be exactly zero, Unified IG raises when
the logarithm is not finite unless `probability_floor` was supplied explicitly.
When supplied, each probability is floored and the vector is renormalized;
completeness refers to that explicitly smoothed score function.

`Explanation.completeness_error` stores the signed residual between the model
output and the reconstructed output. `max_abs_completeness_error` summarizes
the worst sample/output. Numerical backends emit a `RuntimeWarning` when this
exceeds the configured absolute and relative tolerances. The check can be
configured with `completeness_atol`, `completeness_rtol`, and
`check_completeness` on `Explainer`.

## Numerical integration

Numerical backends use Gauss–Legendre quadrature on the unit path interval.
When `n_steps` is omitted, Unified IG starts with 16 nodes and automatically
retries with 32, then 64, if the completeness tolerance is not met. A successful
higher resolution is retained by that explainer for later calls. Supplying
`Explainer(..., n_steps=N)` disables this refinement and uses the requested
number of nodes; disabling completeness checking also disables refinement.

More nodes usually improve accuracy but require proportionally more gradient
evaluations. For a supported degree-`d` polynomial pipeline ending in an affine
estimator, Unified IG caps the active resolution at `ceil(d / 2)`, which is
exact for the polynomial gradient along a straight path. An explicitly smaller
value is not raised automatically.

Scalar-output skgrad models batch baseline-observation paths before evaluating
analytic gradients. `gradient_batch_size` bounds the number of path rows in
each call and defaults to 8,192; it does not change the attribution functional
or the quadrature nodes.

For affine prediction functions, averaging IG over a baseline distribution is
exactly equivalent to using its weighted mean baseline. The affine backend
uses this identity to avoid work proportional to the number of baseline rows.

PyTorch, JAX, and TensorFlow evaluate native automatic gradients at the same
Gauss–Legendre quadrature nodes. Prediction
functions must produce samplewise outputs: one scalar or one vector for each
leading input row. By default, two class scores are reduced to their margin
and three or more are centered under the multiclass convention above.

When explicitly enabled with `fallback="finite_difference"`, Unified IG uses
central finite differences to approximate gradients for otherwise unsupported
smooth sklearn estimators. The step for coordinate `j` is
`finite_difference_step * max(1, abs(x_j))` at each path point. Perturbed rows
are evaluated in bounded batches.

This fallback requires `predict` for regression or `decision_function` for
classification. Multiclass decision functions must return one score per class;
pairwise-derived SVC scores are rejected. Probability outputs are never
inferred. Known tree and nearest-neighbor estimators are rejected because their
local finite-difference gradients do not represent path discontinuities
reliably. A small completeness residual is an important numerical diagnostic,
but it is not a general proof that a model is smooth or that every individual
attribution is accurate.

The separate numerical-tree control `tree_grid_size` defaults to 1,024 path
intervals. Increasing it improves the chance of separating nearby tree
crossings but increases model evaluations proportionally.

## Attribution feature space

The default explains the inputs of the supplied model object. For fitted
pipelines, `attribute_after="step_name"` selects features after that step while
still accepting original observations and baselines. Data, names, and feature
axes in the Explanation all refer to the selected space; `attribute_after`
records the choice. See [feature-space selection](feature-spaces.md) for path,
weighting, scaling, and nonlinear-transformation conventions.
