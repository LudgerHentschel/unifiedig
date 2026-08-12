"""Log-score attribution for a probability-only random forest classifier."""

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from cbaseline import background
import unifiedig as uig


X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    random_state=0,
)
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
probability_floor = 1e-6
probabilities = np.maximum(model.predict_proba(X), probability_floor)
scores = np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])
f0 = float(scores.mean())
bg = background(
    predictions=scores,
    f0=f0,
    features=X,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explainer = uig.Explainer(
    model,
    bg,
    fallback="tree_numeric",
    probability_floor=probability_floor,
    tree_grid_size=256,
)
explanation = explainer(X[100:105])

print(explanation.values)
print(explanation.max_abs_completeness_error)
