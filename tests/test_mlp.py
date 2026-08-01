import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

import unifiedig as uig


def test_tanh_mlp_regressor_is_complete():
    rng = np.random.default_rng(10)
    training = rng.normal(size=(80, 3))
    targets = np.sin(training[:, 0]) + training[:, 1] * training[:, 2]
    model = MLPRegressor(
        hidden_layer_sizes=(6, 4),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=2,
    ).fit(training, targets)
    data = np.array([[0.5, -0.2, 0.8], [-0.7, 0.4, 0.1]])
    baseline = np.array([0.1, -0.1, 0.0])

    result = uig.Explainer(model, baseline, n_steps=64)(data)

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(data),
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.base_values,
        model.predict(np.broadcast_to(baseline, data.shape)),
    )


def test_multi_output_mlp_regressor_shapes_and_completeness():
    rng = np.random.default_rng(3)
    training = rng.normal(size=(60, 2))
    targets = np.column_stack((training[:, 0] + training[:, 1], training[:, 0] - training[:, 1]))
    model = MLPRegressor(
        hidden_layer_sizes=(5,),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=1,
    ).fit(training, targets)
    data = np.array([[0.2, 0.8], [-0.4, 0.3]])

    result = uig.Explainer(model, np.zeros(2))(data)

    assert result.values.shape == (2, 2, 2)
    assert result.base_values.shape == (2, 2)
    assert result.output_names == ["0", "1"]
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(data),
        rtol=1e-7,
        atol=1e-7,
    )


def test_binary_mlp_classifier_is_complete_on_logit_scale():
    rng = np.random.default_rng(5)
    training = rng.normal(size=(100, 2))
    labels = (training[:, 0] - 0.5 * training[:, 1] > 0).astype(int)
    model = MLPClassifier(
        hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs", random_state=4
    ).fit(training, labels)
    data = np.array([[0.3, -0.2], [-0.5, 0.6]])

    result = uig.Explainer(model, np.zeros(2), n_steps=64)(data)
    probabilities = model.predict_proba(data)[:, 1]
    logits = np.log(probabilities / (1.0 - probabilities))

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        logits,
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.output_names == [str(model.classes_[1])]


def test_multiclass_mlp_classifier_is_rejected():
    model = MLPClassifier(hidden_layer_sizes=(2,), solver="lbfgs", random_state=0).fit(
        np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1, 2, 2]),
    )
    with pytest.raises(ValueError, match="binary"):
        uig.Explainer(model, [0.0])


@pytest.mark.parametrize("n_steps", [0, -1, 1.5, True])
def test_invalid_n_steps_is_rejected(n_steps):
    model = MLPRegressor(
        hidden_layer_sizes=(2,), solver="lbfgs", max_iter=1000, random_state=0
    ).fit(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0])
    )
    with pytest.raises(ValueError, match="positive integer"):
        uig.Explainer(model, [0.0], n_steps=n_steps)
