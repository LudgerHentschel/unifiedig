"""Attribute out-of-sample squared-error loss."""

import numpy as np
from sklearn.linear_model import Ridge

import unifiedig as uig

rng = np.random.default_rng(0)
X = rng.normal(size=(240, 3))
y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=len(X))

model = Ridge(alpha=0.5).fit(X[:160], y[:160])
background = X[:40]
X_test, y_test = X[160:], y[160:]

explanation = uig.LossExplainer(
    model,
    background,
    loss="squared_error",
)(X_test, y_test)

np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    (y_test - model.predict(X_test)) ** 2,
)
