# Classification: attribute scores, not probabilities

**UnifiedIG explains classification decisions on the score scale.** Binary
attributions decompose a margin or logit; multiclass attributions decompose the
complete centered score vector. They do not represent percentage-point changes
in class probabilities. Construct baselines and check completeness on that
same score scale.

## Why this is the right output for UnifiedIG

The classification question is how features account for the model's score
contrast relative to a reference. The score is the quantity being compared
between classes before a probability link compresses and normalizes it.
Using probabilities would change that question and the feature allocations.
UnifiedIG therefore keeps the score definition fixed across its model backends.

**Attributing probabilities is possible, but we recommend against it for
explaining classification decisions.** It is a bad idea when the goal is to
understand feature contributions to the model's score contrasts: the probability
link compresses score changes near zero and one, and multiclass normalization
couples each class to the other classes. The resulting allocations reflect
those transformations as well as changes in the underlying scores.

Probability IG can be mathematically complete for a probability difference.
That does not make it a good explanation of a score difference. Use scores for
that question. Probability attribution is defensible only when the probability
change itself is explicitly the quantity you intend to decompose and you accept
the link-dependent interpretation. UnifiedIG deliberately does not offer it.

## Binary classification: separate score changes from probability compression

For a logistic model, $p(x)=\sigma(z(x))$, where $z$ is the logit. The chain
rule gives

$$
\frac{\partial p}{\partial x_j}
= p(1-p)\frac{\partial z}{\partial x_j}.
$$

Probability IG includes the factor $p(1-p)$ at every point along the path.
Score IG integrates the score derivative itself. The probability link thus
changes the weighting along the path, especially where probabilities approach
zero or one. Integrating the probability derivative does not make that factor
disappear; it reconstructs a different endpoint difference.

For example, raising a logit from 0 to 2 changes probability from 0.500 to
0.881. Raising it from 4 to 6 changes probability from 0.982 to 0.998. Both are
a +2 score change, but their probability changes are approximately 0.381 and
0.0155. Score attribution keeps the change in the decision score explicit,
instead of expressing it through the nonlinear probability scale.

For an affine score, $z(x)=\beta_0+\beta^T x$, each point-baseline contribution
is exactly $\beta_j(x_j-b_j)$. Applying a sigmoid changes the explained
function; the probability attribution is no longer this affine decomposition.

A margin is not universally a logit. For example, an SVM's `decision_function`
is a decision margin without an automatic log-odds interpretation. UnifiedIG
preserves the model's supported score convention; it does not claim that all
classifiers share a calibrated score scale. Compare models only with their
output units and references made explicit.

## Multiclass classification: preserve the full decision contrast

For softmax models, $p_k=\exp(z_k)/\sum_\ell\exp(z_\ell)$. Changing a rival
class's score changes $p_k$ even when $z_k$ itself is unchanged:

$$
\frac{\partial p_k}{\partial x_j}
= p_k\left(\frac{\partial z_k}{\partial x_j}
-\sum_\ell p_\ell\frac{\partial z_\ell}{\partial x_j}\right).
$$

The probabilities introduce path-dependent weighting by all class
probabilities. UnifiedIG instead explains the complete relative score vector.
Adding the same possibly input-dependent offset to every score leaves softmax
probabilities and class comparisons unchanged. To remove that common direction
without selecting a privileged reference class, UnifiedIG uses

$$
s_k(x)=z_k(x)-\frac{1}{K}\sum_{\ell=1}^K z_\ell(x).
$$

Centering is the unique zero-sum representative of a given score vector up to
common shifts. Its coordinates sum to zero and retain all pairwise margins:
$s_a-s_b=z_a-z_b$. By linearity of IG, subtracting the corresponding
attributions gives an attribution of that pairwise margin. Use
`explanation.contrast(a, b)` without a second model pass.

For softmax scores, the margin also equals $\log(p_a/p_b)$. For other
classifiers, retain the model's decision-score interpretation. Centering
removes common offsets; it does not remove arbitrary score rescaling or make
different fitted models' scores interchangeable.

## Baselines must use the same output scale

For binary logistic regression, pass `model.decision_function(X_reference)`
to CBaseline. For multiclass classification, pass the centered score matrix.
The [worked multiclass example](examples.md#multiclass-classification) constructs
that distribution and verifies score reconstruction.

Choose a reference score deliberately. `mean(logit(p))` generally differs from
`logit(mean(p))`. A background calibrated to a mean score does not thereby have
a specified mean probability. If you start from a reference probability, use
the model's appropriate score link and understand that calibration constrains
the **score mean**. See [CBaseline's reference guide](https://ludgerhentschel.github.io/cbaseline/reference-predictions.html).

## Probability-only trees still produce score attributions

Some supported sklearn trees and forests expose probabilities without native
decision scores. With the explicit `fallback="tree_numeric"` route, UnifiedIG
defines binary log odds or centered log probabilities of the complete model:

$$
s=\log p_1-\log p_0,
\qquad
s_k=\log p_k-\frac{1}{K}\sum_\ell\log p_\ell.
$$

These are derived scores, not recovered training-time margins. For a forest,
the transformation applies after probability aggregation across trees. It
cannot be replaced by averaging the transformed scores of individual trees.

Zero probabilities require an explicitly chosen `probability_floor`; the
floored vector is renormalized before taking logs. Completeness then refers
to that explicitly defined finite-score function. This is a score construction
for these supported models, not a probability-attribution mode.

## Framework models, plotting, and loss

Supply logits or raw scores from PyTorch, JAX, and TensorFlow functions.
Two scores become `score[1] - score[0]`; larger class vectors are centered.
Visible Keras sigmoid/softmax classification heads are rejected, but a custom
function can hide a probability transformation that UnifiedIG cannot detect.
The caller is responsible for supplying the documented score output. Do not
use `output_kind="regression"` to relabel class probabilities; that option is
for actual multi-output regression.

[SHAP plots](plotting.md) display the same score contributions. A contribution
of +0.4 is a score change, not a 40-percentage-point probability change. Applying
sigmoid or softmax to each feature contribution separately does not produce
additive probability contributions. With a known link, reconstruct the full
endpoint score first to obtain an endpoint probability.

[Log-loss attribution](loss.md) is a distinct output question. It starts from
scores and applies the loss derivative along the path to explain loss for a
fixed observed target. Its contributions reconstruct loss, rather than either
a score or a probability.

## Further reading

The [original IG paper](https://proceedings.mlr.press/v70/sundararajan17a.html)
defines attribution for a specified output function. The choice of that output
is essential to interpreting its completeness property.
[scikit-learn's probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
explains the distinction between classifier scores and calibrated probabilities.
The chain-rule equations above show why choosing the probability output changes
the path attribution.
