"""README quick start; also used to smoke-test the installed wheel."""

import numpy as np
from sklearn.linear_model import Ridge

from cbaseline import background
import unifiedig as uig

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 4))
y_train = 2.0 * X_train[:, 0] - X_train[:, 1] + 0.5 * X_train[:, 2]
model = Ridge(alpha=0.5).fit(X_train, y_train)

# Choose a reference prediction and construct observed baseline inputs whose
# weighted mean model prediction equals that reference.
f_train = model.predict(X_train)
f0 = float(f_train.mean())
bg = background(
    predictions=f_train,
    f0=f0,
    features=X_train,
    weighting="calibrated",
)
X_eval = X_train[100:105]

explanation = uig.Explainer(model, bg)(X_eval)

np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    model.predict(X_eval),
)
