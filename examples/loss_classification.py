"""Attribute out-of-sample multiclass log loss from raw decision scores."""

import numpy as np
from sklearn.linear_model import LogisticRegression

import unifiedig as uig

rng = np.random.default_rng(4)
X = rng.normal(size=(300, 4))
scores = np.column_stack((X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]))
y = np.argmax(scores + rng.normal(scale=0.5, size=scores.shape), axis=1)

model = LogisticRegression().fit(X[:200], y[:200])
background = X[:50]
X_test, y_test = X[200:], y[200:]

explanation = uig.LossExplainer(
    model,
    background,
    loss="log_loss",
)(X_test, y_test)

endpoint_loss = -np.log(model.predict_proba(X_test)[np.arange(len(y_test)), y_test])
np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    endpoint_loss,
    atol=2e-9,
)
