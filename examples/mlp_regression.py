"""Explain an sklearn MLP regressor with analytic gradients."""

import numpy as np
from sklearn.neural_network import MLPRegressor

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

explanation = uig.Explainer(model, baseline=np.zeros(3), n_steps=64)(
    [[0.5, -0.2, 0.8]]
)
print(explanation.values)
print(explanation.max_abs_completeness_error)
