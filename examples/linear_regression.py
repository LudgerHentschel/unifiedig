"""Explain a fitted linear regression model."""

import numpy as np
from sklearn.linear_model import LinearRegression

import unifiedig as uig

training = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
model = LinearRegression().fit(training, np.array([1.0, 3.0, -2.0, 0.0]))

# Use observed reference cases as a shared baseline distribution.
explanation = uig.Explainer(model, training)([[0.25, 0.75]])
print(explanation.values)
print(explanation.max_abs_completeness_error)
