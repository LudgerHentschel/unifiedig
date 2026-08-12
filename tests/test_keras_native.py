import importlib.util

import numpy as np
import pytest

import unifiedig as uig

keras_available = importlib.util.find_spec("keras") is not None
pytestmark = pytest.mark.skipif(not keras_available, reason="keras is not installed")

if keras_available:
    import keras


def test_keras_uses_its_configured_native_autodiff_backend():
    backend = keras.config.backend()
    if backend not in {"jax", "torch"}:
        pytest.skip("this test exercises the Keras JAX and Torch backends")

    class LinearModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.output_layer = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.Constant([[1.5], [-0.5]]),
                bias_initializer=keras.initializers.Constant(0.25),
            )

        def call(self, inputs):
            return self.output_layer(inputs)

    model = LinearModel()
    data = np.array([[1.0, 2.0], [-0.5, 0.7]], dtype=np.float32)

    result = uig.Explainer(model, [0.0, 0.0])(data)

    expected = keras.ops.convert_to_numpy(
        model(keras.ops.convert_to_tensor(data), training=False)
    ).reshape(-1)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        expected,
        atol=1e-5,
    )


def test_native_keras_probability_head_is_rejected():
    backend = keras.config.backend()
    if backend not in {"jax", "torch"}:
        pytest.skip("TensorFlow probability heads are tested separately")
    model = keras.Sequential(
        [keras.Input((2,)), keras.layers.Dense(2, activation="softmax")]
    )

    with pytest.raises(ValueError, match="probabilities"):
        uig.Explainer(model, [0.0, 0.0])
