"""Construct a baseline distribution from a target reference prediction."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cbaseline import background
import unifiedig as uig


rng = np.random.default_rng(12)
X_train = rng.normal(size=(1_000, 4))
y_train = (
    1.5 * X_train[:, 0]
    - X_train[:, 1]
    + 0.75 * X_train[:, 2] ** 2
    + rng.normal(scale=0.2, size=len(X_train))
)
model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)

# Choose the prediction difference the explanation should decompose.
f_train = model.predict(X_train)
f0 = float(f_train.mean())

# CBaseline solves for an observed, weighted reference distribution whose
# average model prediction equals f0. The inverse map from f0 to input rows is
# not unique, so UnifiedIG intentionally delegates this construction.
bg = background(
    predictions=f_train,
    f0=f0,
    features=X_train,
    weighting="calibrated",
)

X_eval = X_train[500:505]
# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)(X_eval)

np.testing.assert_allclose(explanation.base_values, f0)
np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    model.predict(X_eval),
)

print(bg.diagnostics["calibrated_neutrality_norm"])
print(explanation.values)
