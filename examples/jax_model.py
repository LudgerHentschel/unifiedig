"""Integrated Gradients for a parameterized JAX prediction function."""

import jax.numpy as jnp
import numpy as np

import unifiedig as uig


def predict(params, X):
    hidden = jnp.tanh(X @ params["hidden_weights"])
    return hidden @ params["output_weights"]


params = {
    "hidden_weights": jnp.array([[0.8, -0.3], [0.2, 0.7]]),
    "output_weights": jnp.array([1.1, -0.6]),
}
background = np.array([[0.0, 0.0], [0.1, -0.1]], dtype=np.float32)
X_eval = np.array([[0.5, -0.2], [1.0, 0.4]], dtype=np.float32)

model = uig.JaxModel(predict, params=params)
explanation = uig.Explainer(model, background)(X_eval)

print(explanation.values)
print(explanation.max_abs_completeness_error)
