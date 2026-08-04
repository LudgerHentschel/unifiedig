"""Numerical path-event IG for a piecewise-constant tree model."""

from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor

import unifiedig as uig


X, y = make_regression(n_samples=200, n_features=4, random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X, y)

explainer = uig.Explainer(
    model,
    X[:20],
    fallback="tree_numeric",
    tree_grid_size=256,
)
explanation = explainer(X[100:105])

print(explanation.values)
print(explanation.max_abs_completeness_error)
