"""Explain the positive-class logit of a binary sklearn MLP classifier."""

import numpy as np
from sklearn.neural_network import MLPClassifier

import unifiedig as uig

rng = np.random.default_rng(5)
training = rng.normal(size=(100, 2))
labels = (training[:, 0] - 0.5 * training[:, 1] > 0).astype(int)
model = MLPClassifier(
    hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs", random_state=4
).fit(training, labels)

explanation = uig.Explainer(model, training[:25])([[0.3, -0.2]])
print(explanation.values)
print(explanation.output_names)
