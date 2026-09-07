"""Explain a Gaussian-process regressor with numerical gradients."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from cbaseline import background
import unifiedig as uig


rng = np.random.default_rng(8)
X = rng.normal(size=(80, 4))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
model = GaussianProcessRegressor(
    kernel=RBF(1.2), alpha=1e-6, optimizer=None
).fit(X, y)
predictions = model.predict(X)
f0 = float(predictions.mean())
bg = background(
    predictions=predictions,
    f0=f0,
    features=X,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explainer = uig.Explainer(
    model,
    bg,
    fallback="finite_difference",
    n_steps=32,
)
explanation = explainer(X[20:25])

print(explanation.values)
print(explanation.max_abs_completeness_error)
