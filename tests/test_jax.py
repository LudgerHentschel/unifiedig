import importlib.util

import numpy as np
import pytest

import unifiedig as uig


jax_available = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not jax_available, reason="jax is not installed")

if jax_available:
    import jax.numpy as jnp


def test_jax_nonlinear_scalar_output_is_complete():
    def predict(X):
        return X[:, 0] ** 2 + jnp.tanh(X[:, 1])

    data = np.array([[0.5, -0.2], [1.2, 0.7]], dtype=np.float32)
    result = uig.Explainer(
        uig.JaxModel(predict), np.zeros(2, dtype=np.float32), n_steps=32
    )(data)

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        np.asarray(predict(jnp.asarray(data))),
        atol=2e-6,
    )
    assert result.values.dtype == np.float32
    assert result.max_abs_completeness_error < 2e-6


def test_jax_explicit_params_and_weighted_baselines():
    def predict(params, X):
        return X @ params["coef"] + params["intercept"]

    params = {
        "coef": jnp.array([2.0, -0.5], dtype=jnp.float32),
        "intercept": jnp.float32(0.25),
    }
    data = np.array([[1.0, 2.0], [-0.5, 0.7]], dtype=np.float32)
    baselines = np.array([[0.0, 0.0], [0.2, -0.1]], dtype=np.float32)
    weights = np.array([0.25, 0.75])

    result = uig.Explainer(
        uig.JaxModel(predict, params=params),
        baselines,
        baseline_weights=weights,
    )(data)

    baseline_outputs = np.asarray(predict(params, jnp.asarray(baselines)))
    np.testing.assert_allclose(result.base_values, weights @ baseline_outputs)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        np.asarray(predict(params, jnp.asarray(data))),
        atol=1e-6,
    )


def test_jax_multiclass_output_uses_centered_scores_and_names():
    weights = jnp.array(
        [[1.0, -0.5, 0.2], [-0.3, 0.8, 0.6]], dtype=jnp.float32
    )

    def predict(X):
        return X @ weights

    data = np.array([[1.0, 2.0], [-0.5, 0.7]], dtype=np.float32)
    model = uig.JaxModel(predict, output_names=["red", "green", "blue"])
    result = uig.Explainer(model, np.zeros(2, dtype=np.float32))(data)

    raw = np.asarray(predict(jnp.asarray(data)))
    centered = raw - raw.mean(axis=1, keepdims=True)
    assert result.values.shape == (2, 2, 3)
    assert result.output_names == ["red", "green", "blue"]
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, centered, atol=1e-6
    )
    contrast = result.contrast("red", "blue")
    np.testing.assert_allclose(
        contrast.values.sum(axis=1) + contrast.base_values,
        raw[:, 0] - raw[:, 2],
        atol=1e-6,
    )


def test_two_score_jax_output_becomes_binary_margin():
    def predict(X):
        return jnp.stack((X[:, 0] - X[:, 1], 2.0 * X[:, 1]), axis=1)

    data = np.array([[1.0, 2.0], [-0.5, 0.7]], dtype=np.float32)
    result = uig.Explainer(
        uig.JaxModel(predict, output_names=["no", "yes"]), [0.0, 0.0]
    )(data)

    raw = np.asarray(predict(jnp.asarray(data)))
    assert result.values.shape == (2, 2)
    assert result.output_names == ["yes"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        raw[:, 1] - raw[:, 0],
        atol=1e-6,
    )


def test_multi_output_jax_regression_is_not_centered_when_declared():
    def predict(X):
        return jnp.stack((X[:, 0] + X[:, 1], X[:, 0] - 2.0 * X[:, 1]), axis=1)

    data = np.array([[1.0, 2.0], [-0.5, 0.7]], dtype=np.float32)
    model = uig.JaxModel(predict, output_names=["level", "change"])
    result = uig.Explainer(
        model,
        np.zeros(2, dtype=np.float32),
        output_kind="regression",
    )(data)

    assert result.values.shape == (2, 2, 2)
    assert result.output_names == ["level", "change"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        np.asarray(predict(jnp.asarray(data))),
        atol=1e-6,
    )


def test_vectorized_single_sample_jax_function():
    def predict_one(x, *, offset):
        return jnp.sin(x[0]) + x[1] + offset

    model = uig.JaxModel(
        predict_one, vectorize=True, call_kwargs={"offset": 0.3}
    )
    data = np.array([[0.2, 0.4], [0.8, -0.3]], dtype=np.float32)
    result = uig.Explainer(model, np.zeros(2, dtype=np.float32), n_steps=24)(data)

    expected = np.asarray(jnp.array([predict_one(row, offset=0.3) for row in data]))
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, expected, atol=1e-6
    )


def test_jax_rejects_unsupported_output_shape():
    def bad_predict(X):
        return jnp.ones((X.shape[0], 2, 2))

    with pytest.raises(ValueError, match="one scalar or one output vector"):
        uig.Explainer(uig.JaxModel(bad_predict), [0.0, 0.0])([[1.0, 2.0]])


def test_jax_output_names_must_match_scores():
    def predict(X):
        return jnp.ones((X.shape[0], 3))

    with pytest.raises(ValueError, match="class names"):
        uig.Explainer(
            uig.JaxModel(predict, output_names=["a", "b"]), [0.0]
        )([[1.0]])
