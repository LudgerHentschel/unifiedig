"""Explain a binary logistic regression on its decision-score scale."""

import numpy as np
from sklearn.linear_model import LogisticRegression

import unifiedig as uig

training = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
model = LogisticRegression().fit(training, np.array([0, 0, 1, 1]))

explanation = uig.Explainer(model, baseline=[0.0, 0.0])([[0.5, -0.25]])
print(explanation.values)
print(explanation.base_values)

