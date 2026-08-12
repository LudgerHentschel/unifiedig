"""Explain a fitted linear regression model."""

import numpy as np
from sklearn.linear_model import LinearRegression

from cbaseline import background
import unifiedig as uig

training = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
model = LinearRegression().fit(training, np.array([1.0, 3.0, -2.0, 0.0]))
predictions = model.predict(training)
f0 = float(predictions.mean())
bg = background(
    predictions=predictions,
    f0=f0,
    features=training,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)([[0.25, 0.75]])
print(explanation.values)
print(explanation.max_abs_completeness_error)
