"""Backend-parity tests for LossExplainer."""

import importlib.util

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor

import unifiedig as uig


def _assert_complete(result, endpoint_loss, *, atol):
    np.testing.assert_allclose(
        result.base_values
        + result.values.sum(axis=tuple(range(1, result.values.ndim))),
        endpoint_loss,
        atol=atol,
    )


def test_loss_quadrature_specializes_affine_models():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])

    regression = Ridge().fit(X, np.array([-3.0, -1.0, 1.0, 3.0]))
    squared_error = uig.LossExplainer(regression, [0.0])
    assert squared_error.n_steps == 1
    result = squared_error(X, np.array([-2.5, -0.5, 0.5, 2.5]))
    _assert_complete(
        result,
        (np.array([-2.5, -0.5, 0.5, 2.5]) - regression.predict(X)) ** 2,
        atol=1e-12,
    )

    classifier = LogisticRegression().fit(X, np.array([0, 0, 1, 1]))
    automatic = uig.LossExplainer(classifier, [0.0], loss="log_loss")
    explicit = uig.LossExplainer(classifier, [0.0], loss="log_loss", n_steps=5)
    assert automatic.n_steps == 8
    assert explicit.n_steps == 5


def test_nonlinear_loss_keeps_general_default_quadrature():
    X = np.linspace(-1.0, 1.0, 20)[:, None]
    model = MLPRegressor(hidden_layer_sizes=(3,), max_iter=500, random_state=4).fit(
        X, np.sin(X[:, 0])
    )
    assert uig.LossExplainer(model, [0.0]).n_steps == 16


def test_finite_difference_loss_fallback():
    X = np.linspace(-1.0, 1.0, 20)[:, None]
    y = np.sin(X[:, 0])
    model = GaussianProcessRegressor(alpha=1e-6).fit(X, y)
    with pytest.warns(RuntimeWarning, match="finite-difference gradients"):
        result = uig.LossExplainer(
            model,
            X[:2],
            fallback="finite_difference",
            n_steps=32,
            completeness_atol=2e-4,
        )(X[10:13], y[10:13])
    _assert_complete(result, (y[10:13] - model.predict(X[10:13])) ** 2, atol=2e-4)


def test_finite_difference_binary_log_loss_uses_scores():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["no", "no", "yes", "yes"])
    model = LinearDiscriminantAnalysis().fit(X, y)
    data = np.array([[-0.5], [1.5]])
    labels = np.array(["no", "yes"])
    with pytest.warns(RuntimeWarning, match="finite-difference gradients"):
        result = uig.LossExplainer(
            model, [0.0], loss="log_loss", fallback="finite_difference", n_steps=32
        )(data, labels)
    score = model.decision_function(data)
    target = (labels == "yes").astype(float)
    _assert_complete(result, np.logaddexp(0.0, score) - target * score, atol=2e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("treeig") is None, reason="treeig is not installed"
)
def test_exact_tree_binary_and_multiclass_log_loss():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(120, 3))

    binary_y = np.where(X[:, 0] > 0, "up", "down")
    binary = GradientBoostingClassifier(random_state=8).fit(X, binary_y)
    data = X[20:24]
    labels = binary_y[20:24]
    binary_result = uig.LossExplainer(binary, X[:2], loss="log_loss")(data, labels)
    binary_loss = -np.log(
        binary.predict_proba(data)[np.arange(len(data)), (labels == "up").astype(int)]
    )
    _assert_complete(binary_result, binary_loss, atol=1e-10)

    multiclass_y = np.digitize(X[:, 0] + X[:, 1], [-0.5, 0.5])
    multiclass = GradientBoostingClassifier(random_state=9).fit(X, multiclass_y)
    labels = multiclass_y[20:24]
    multiclass_result = uig.LossExplainer(multiclass, X[:2], loss="log_loss")(
        data, labels
    )
    multiclass_loss = -np.log(
        multiclass.predict_proba(data)[np.arange(len(data)), labels]
    )
    _assert_complete(multiclass_result, multiclass_loss, atol=1e-10)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)
def test_pytorch_squared_error_and_multiclass_log_loss():
    import torch

    regression = torch.nn.Linear(2, 1, bias=True).double()
    data = torch.tensor([[0.5, -0.2], [-0.4, 0.8]], dtype=torch.float64)
    y = np.array([0.25, -0.5])
    result = uig.LossExplainer(regression, [0.0, 0.0], n_steps=16)(data, y)
    prediction = regression(data).detach().numpy()[:, 0]
    _assert_complete(result, (y - prediction) ** 2, atol=1e-10)

    classifier = torch.nn.Linear(2, 3, bias=True).double()
    labels = np.array([0, 2])
    result = uig.LossExplainer(classifier, [0.0, 0.0], loss="log_loss", n_steps=32)(
        data, labels
    )
    scores = classifier(data).detach().numpy()
    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    _assert_complete(result, -np.log(probability[np.arange(2), labels]), atol=1e-9)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax is not installed"
)
def test_jax_squared_error_loss():
    import jax.numpy as jnp

    model = uig.JaxModel(lambda X: X[:, 0] ** 2 + X[:, 1])
    data = np.array([[0.5, -0.2], [-0.4, 0.8]], dtype=np.float32)
    y = np.array([0.25, -0.5], dtype=np.float32)
    result = uig.LossExplainer(model, [0.0, 0.0], n_steps=32)(data, y)
    prediction = np.asarray(model.predict_fn(jnp.asarray(data)))
    _assert_complete(result, (y - prediction) ** 2, atol=2e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="tensorflow is not installed",
)
def test_tensorflow_binary_log_loss():
    import tensorflow as tf

    model = uig.TensorFlowModel(
        lambda X: tf.square(X[:, 0]) - X[:, 1], dtype=tf.float64
    )
    data = np.array([[0.5, -0.2], [-0.4, 0.8]], dtype=float)
    labels = np.array([1.0, 0.0])
    result = uig.LossExplainer(model, [0.0, 0.0], loss="log_loss", n_steps=32)(
        data, labels
    )
    scores = model.predict_fn(tf.constant(data, dtype=tf.float64)).numpy()
    endpoint_loss = np.logaddexp(0.0, scores) - labels * scores
    _assert_complete(result, endpoint_loss, atol=1e-9)
