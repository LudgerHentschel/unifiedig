"""Explain the positive-class logit of a binary sklearn MLP classifier."""

import numpy as np
from sklearn.neural_network import MLPClassifier

from cbaseline import background
import unifiedig as uig

rng = np.random.default_rng(5)
training = rng.normal(size=(100, 2))
labels = (training[:, 0] - 0.5 * training[:, 1] > 0).astype(int)
model = MLPClassifier(
    hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs", random_state=4
).fit(training, labels)

# UnifiedIG explains the positive-class logit, so construct CBaseline on that
# same output scale. This model has one tanh hidden layer.
hidden = np.tanh(training @ model.coefs_[0] + model.intercepts_[0])
logits = (hidden @ model.coefs_[1] + model.intercepts_[1]).ravel()
f0 = float(logits.mean())
bg = background(
    predictions=logits,
    f0=f0,
    features=training,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)([[0.3, -0.2]])
print(explanation.values)
print(explanation.output_names)
