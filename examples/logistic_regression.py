"""Explain a binary logistic regression on its decision-score scale."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from cbaseline import background
import unifiedig as uig

training = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
model = LogisticRegression().fit(training, np.array([0, 0, 1, 1]))
scores = model.decision_function(training)
f0 = float(scores.mean())
bg = background(
    predictions=scores,
    f0=f0,
    features=training,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)([[0.5, -0.25]])
print(explanation.values)
print(explanation.base_values)
