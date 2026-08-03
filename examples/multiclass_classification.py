"""Centered decision-score attribution for multiclass classification."""

import numpy as np
from sklearn.linear_model import LogisticRegression

import unifiedig as uig


rng = np.random.default_rng(7)
X = rng.normal(size=(300, 4))
latent_scores = np.column_stack(
    (X[:, 0], X[:, 1] - 0.5 * X[:, 2], -X[:, 0] - X[:, 1])
)
y = np.argmax(latent_scores, axis=1)
model = LogisticRegression(max_iter=1000).fit(X, y)

explanation = uig.Explainer(model, X[:50])(X[100:105])

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
