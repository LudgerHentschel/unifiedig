"""Explain a TensorFlow-backed Keras model."""

import numpy as np
from tensorflow import keras

from cbaseline import background
import unifiedig as uig

keras.utils.set_random_seed(9)
model = keras.Sequential(
    [
        keras.Input((3,)),
        keras.layers.Dense(8, activation="tanh"),
        keras.layers.Dense(1),
    ]
)
data = np.array([[0.5, -0.2, 0.8]], dtype=np.float32)
rng = np.random.default_rng(9)
reference = rng.normal(size=(200, 3)).astype(np.float32)
reference_predictions = model(reference, training=False).numpy()[:, 0]
f0 = float(reference_predictions.mean())
bg = background(
    predictions=reference_predictions,
    f0=f0,
    features=reference,
    weighting="calibrated",
)

# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)(data)

print(explanation.values)
print(explanation.max_abs_completeness_error)
