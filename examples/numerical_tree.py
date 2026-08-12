"""Numerical path-event IG for a piecewise-constant tree model."""

from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor

from cbaseline import background
import unifiedig as uig


X, y = make_regression(n_samples=200, n_features=4, random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X, y)
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
    fallback="tree_numeric",
    tree_grid_size=256,
)
explanation = explainer(X[100:105])

print(explanation.values)
print(explanation.max_abs_completeness_error)
