"""Integrated Gradients for a parameterized JAX prediction function."""

import jax.numpy as jnp
import numpy as np

from cbaseline import background
import unifiedig as uig


def predict(params, X):
    hidden = jnp.tanh(X @ params["hidden_weights"])
    return hidden @ params["output_weights"]


params = {
    "hidden_weights": jnp.array([[0.8, -0.3], [0.2, 0.7]]),
    "output_weights": jnp.array([1.1, -0.6]),
}
rng = np.random.default_rng(4)
reference = rng.normal(size=(200, 2)).astype(np.float32)
reference_predictions = np.asarray(predict(params, jnp.asarray(reference)))
f0 = float(reference_predictions.mean())
bg = background(
    predictions=reference_predictions,
    f0=f0,
    features=reference,
    weighting="calibrated",
)
X_eval = np.array([[0.5, -0.2], [1.0, 0.4]], dtype=np.float32)

model = uig.JaxModel(predict, params=params)
# A deliberate single starting point is also valid: uig.Explainer(model, x0).
explanation = uig.Explainer(model, bg)(X_eval)

print(explanation.values)
print(explanation.max_abs_completeness_error)
