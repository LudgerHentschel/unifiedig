"""Explain an sklearn MLP regressor with analytic gradients."""

import numpy as np
from sklearn.neural_network import MLPRegressor

from cbaseline import background
import unifiedig as uig

rng = np.random.default_rng(7)
training = rng.normal(size=(100, 3))
targets = training[:, 0] - 0.5 * training[:, 1] + 0.25 * training[:, 2]
model = MLPRegressor(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=1,
).fit(training, targets)
predictions = model.predict(training)
f0 = float(predictions.mean())
bg = background(
    predictions=predictions,
    f0=f0,
    features=training,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)([[0.5, -0.2, 0.8]])
print(explanation.values)
print(explanation.max_abs_completeness_error)
