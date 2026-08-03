"""Explain an RBF support-vector regressor with numerical gradients."""

import numpy as np
from sklearn.svm import SVR

import unifiedig as uig


rng = np.random.default_rng(8)
X = rng.normal(size=(80, 4))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
model = SVR(C=5.0, epsilon=0.01).fit(X, y)

explainer = uig.Explainer(
    model,
    X[:5],
    fallback="finite_difference",
    n_steps=32,
)
explanation = explainer(X[20:25])

print(explanation.values)
print(explanation.max_abs_completeness_error)
