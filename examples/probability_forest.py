"""Log-score attribution for a probability-only random forest classifier."""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import unifiedig as uig


X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    random_state=0,
)
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)

explainer = uig.Explainer(
    model,
    X[:20],
    fallback="tree_numeric",
    probability_floor=1e-6,
    tree_grid_size=256,
)
explanation = explainer(X[100:105])

print(explanation.values)
print(explanation.max_abs_completeness_error)
