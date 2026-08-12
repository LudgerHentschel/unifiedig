"""Centered decision-score attribution for multiclass classification."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from cbaseline import background
import unifiedig as uig


rng = np.random.default_rng(7)
X = rng.normal(size=(300, 4))
latent_scores = np.column_stack(
    (X[:, 0], X[:, 1] - 0.5 * X[:, 2], -X[:, 0] - X[:, 1])
)
y = np.argmax(latent_scores, axis=1)
model = LogisticRegression(max_iter=1000).fit(X, y)

# Construct one reference distribution for the complete centered score vector.
training_scores = model.decision_function(X)
centered_training_scores = training_scores - training_scores.mean(
    axis=1, keepdims=True
)
f0 = centered_training_scores.mean(axis=0)
bg = background(
    predictions=centered_training_scores,
    f0=f0,
    features=X,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)(X[100:105])

raw_scores = model.decision_function(X[100:105])
centered_scores = raw_scores - raw_scores.mean(axis=1, keepdims=True)
np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    centered_scores,
)

# Pairwise score-margin attribution is derived without another model pass.
first_vs_second = explanation.contrast(
    str(model.classes_[0]), str(model.classes_[1])
)
print(first_vs_second.values)
