"""Explain a binary logistic regression on its decision-score scale."""

import numpy as np
from sklearn.linear_model import LogisticRegression

import unifiedig as uig

training = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
model = LogisticRegression().fit(training, np.array([0, 0, 1, 1]))

# A score-neutral background can be constructed explicitly with CBaseline;
# this compact example uses the observed reference sample directly.
explanation = uig.Explainer(model, training)([[0.5, -0.25]])
print(explanation.values)
print(explanation.base_values)
