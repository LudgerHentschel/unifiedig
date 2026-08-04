"""Explain a TensorFlow-backed Keras model."""

import numpy as np
from tensorflow import keras

import unifiedig as uig

model = keras.Sequential(
    [
        keras.Input((3,)),
        keras.layers.Dense(8, activation="tanh"),
        keras.layers.Dense(1),
    ]
)
data = np.array([[0.5, -0.2, 0.8]], dtype=np.float32)
background = np.zeros(3, dtype=np.float32)

explanation = uig.Explainer(model, background)(data)

print(explanation.values)
print(explanation.max_abs_completeness_error)
