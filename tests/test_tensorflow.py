import builtins
import importlib.util

import numpy as np
import pandas as pd
import pytest

import unifiedig as uig

tensorflow_available = importlib.util.find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(
    not tensorflow_available, reason="tensorflow is not installed"
)

if tensorflow_available:
    import tensorflow as tf
    from tensorflow import keras


def _linear_model(n_outputs=1, *, activation=None, dtype="float64"):
    model = keras.Sequential(
        [
            keras.Input((2,), dtype=dtype),
            keras.layers.Dense(n_outputs, activation=activation, dtype=dtype),
        ]
    )
    kernel = np.arange(1, 2 * n_outputs + 1, dtype=float).reshape(2, n_outputs)
    bias = np.linspace(-0.25, 0.25, n_outputs)
    model.layers[-1].set_weights([kernel, bias])
    return model


def test_tensorflow_keras_scalar_output_is_complete():
    model = keras.Sequential(
        [
            keras.Input((2,), dtype="float64"),
            keras.layers.Dense(4, activation="tanh", dtype="float64"),
            keras.layers.Dense(1, dtype="float64"),
        ]
    )
    model.layers[0].set_weights(
        [
            np.array([[0.5, -0.2, 0.7, 0.1], [0.3, 0.8, -0.4, 0.6]]),
            np.array([0.1, -0.2, 0.05, 0.3]),
        ]
    )
    model.layers[1].set_weights(
        [np.array([[0.4], [-0.7], [0.2], [0.5]]), np.array([0.15])]
    )
    data = np.array([[0.4, -0.2], [-0.5, 0.7]])
    baseline = np.array([0.1, -0.1])

    result = uig.Explainer(model, baseline, n_steps=32)(data)

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model(data, training=False).numpy()[:, 0],
        atol=1e-9,
    )
    assert result.values.dtype == np.float64
    assert result.max_abs_completeness_error < 1e-9


def test_subclassed_tensorflow_keras_model_dispatches_directly():
    class LinearModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.output_layer = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.Constant([[1.0], [-2.0]]),
                bias_initializer="zeros",
            )

        def call(self, inputs):
            return self.output_layer(inputs)

    model = LinearModel()
    data = np.array([[0.5, -0.25]], dtype=np.float32)
    model(data)

    result = uig.Explainer(model, [0.0, 0.0])(data)

    np.testing.assert_allclose(result.values, [[0.5, 0.5]], atol=1e-6)


def test_tensorflow_weighted_baselines_and_dataframe_names():
    model = _linear_model(dtype="float64")
    data = pd.DataFrame([[0.4, -0.2], [-0.5, 0.7]], columns=["age", "income"])
    baselines = np.array([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4]])
    weights = np.array([0.1, 0.2, 0.7])

    result = uig.Explainer(
        model, baselines, baseline_weights=weights
    )(data)

    expected_base = weights @ model(baselines, training=False).numpy()[:, 0]
    np.testing.assert_allclose(result.base_values, expected_base, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model(data.to_numpy(), training=False).numpy()[:, 0],
        atol=1e-12,
    )
    assert result.feature_names == ["age", "income"]


def test_tensorflow_multiclass_scores_are_centered():
    model = _linear_model(n_outputs=3, dtype="float64")
    data = np.array([[1.0, 2.0], [-0.5, 0.7]])

    result = uig.Explainer(model, [0.0, 0.0])(data)

    raw_scores = model(data, training=False).numpy()
    centered = raw_scores - raw_scores.mean(axis=1, keepdims=True)
    assert result.values.shape == (2, 2, 3)
    assert result.output_names == ["0", "1", "2"]
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, centered, atol=1e-12
    )


def test_tensorflow_two_scores_become_one_binary_margin():
    model = _linear_model(n_outputs=2, dtype="float64")
    data = np.array([[1.0, 2.0], [-0.5, 0.7]])

    result = uig.Explainer(model, [0.0, 0.0])(data)

    raw_scores = model(data, training=False).numpy()
    assert result.values.shape == (2, 2)
    assert result.output_names == ["1"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        raw_scores[:, 1] - raw_scores[:, 0],
        atol=1e-12,
    )


def test_tensorflow_multi_output_regression_is_preserved_when_declared():
    model = _linear_model(n_outputs=2, dtype="float64")
    data = np.array([[1.0, 2.0], [-0.5, 0.7]])

    result = uig.Explainer(
        model, [0.0, 0.0], output_kind="regression"
    )(data)

    assert result.values.shape == (2, 2, 2)
    assert result.output_names == ["0", "1"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model(data, training=False).numpy(),
        atol=1e-12,
    )


def test_tensorflow_structured_single_tensor_input_and_scalar_baseline():
    model = uig.TensorFlowModel(
        lambda X: tf.reduce_sum(X, axis=(1, 2, 3)), dtype=tf.float64
    )
    data = np.arange(8, dtype=float).reshape(2, 1, 2, 2)

    result = uig.Explainer(model, 0.0)(data)

    assert result.values.shape == result.data.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(result.values, data, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=(1, 2, 3)) + result.base_values,
        data.sum(axis=(1, 2, 3)),
        atol=1e-12,
    )


def test_tensorflow_model_adapter_supports_kwargs_names_and_integer_inputs():
    def predict(X, *, scale):
        return tf.stack((scale * X[:, 0], X[:, 1] ** 2), axis=1)

    model = uig.TensorFlowModel(
        predict,
        call_kwargs={"scale": 0.5},
        output_names=["level", "curvature"],
    )
    data = [[1, 2], [3, 4]]

    result = uig.Explainer(
        model, 0, n_steps=32, output_kind="regression"
    )(data)

    assert result.values.dtype == np.float32
    assert result.output_names == ["level", "curvature"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        predict(tf.constant(data, dtype=tf.float32), scale=0.5).numpy(),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("activation", "n_outputs"), [("sigmoid", 1), ("softmax", 3)]
)
def test_keras_probability_heads_are_rejected_for_classification(
    activation, n_outputs
):
    model = _linear_model(
        n_outputs=n_outputs, activation=activation, dtype="float32"
    )

    with pytest.raises(ValueError, match="probabilities"):
        uig.Explainer(model, [0.0, 0.0])


def test_keras_probability_head_can_be_explicit_bounded_regression():
    model = _linear_model(n_outputs=2, activation="sigmoid", dtype="float32")

    result = uig.Explainer(
        model, [0.0, 0.0], output_kind="regression", n_steps=32
    )([[0.2, -0.1]])

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model(np.array([[0.2, -0.1]], dtype=np.float32)).numpy(),
        atol=1e-5,
    )


def test_tensorflow_rejects_structured_multi_head_output():
    model = uig.TensorFlowModel(lambda X: [X[:, 0], X[:, 1]])

    with pytest.raises(ValueError, match="one tensor"):
        uig.Explainer(model, [0.0, 0.0])([[1.0, 2.0]])


def test_missing_tensorflow_error_is_actionable(monkeypatch):
    real_import = builtins.__import__

    def without_tensorflow(name, *args, **kwargs):
        if name.startswith("tensorflow"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tensorflow)
    with pytest.raises(ImportError, match=r"unifiedig\[tensorflow\]"):
        uig.Explainer(uig.TensorFlowModel(lambda X: X[:, 0]), [0.0])
