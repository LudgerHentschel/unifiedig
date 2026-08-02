import warnings

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

import unifiedig as uig


def fitted_one_neuron_mlp(activation="relu"):
    model = MLPRegressor(
        hidden_layer_sizes=(1,),
        activation=activation,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(np.array([[-1.0], [0.0], [1.0]]), np.array([-1.0, 0.0, 1.0]))
    model.coefs_ = [np.array([[1.0]]), np.array([[1.0]])]
    model.intercepts_ = [np.array([-0.5]), np.array([0.0])]
    return model


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
    assert result.max_abs_completeness_error < 1e-7


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


@pytest.mark.parametrize("activation", ["identity", "logistic", "tanh", "relu"])
def test_supported_hidden_activations_are_complete(activation):
    model = fitted_one_neuron_mlp(activation)
    # This path stays inside one ReLU region; smooth activations are unrestricted.
    baseline = np.array([0.6]) if activation == "relu" else np.array([-0.2])
    data = np.array([[0.8], [1.2]])

    result = uig.Explainer(model, baseline)(data)

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(data),
        rtol=1e-10,
        atol=1e-10,
    )


def test_per_sample_baselines_work_for_deep_mlp():
    rng = np.random.default_rng(8)
    training = rng.normal(size=(50, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(4, 3),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=3,
    ).fit(training, training[:, 0] - training[:, 1] ** 2)
    data = np.array([[0.5, 0.2], [-0.4, 0.7]])
    baselines = np.array([[0.0, 0.0], [0.2, -0.1]])

    result = uig.Explainer(model, baselines)(data)

    np.testing.assert_allclose(result.base_values, model.predict(baselines))
    assert result.max_abs_completeness_error < 1e-7


def test_relu_completeness_warning_and_quadrature_convergence():
    model = fitted_one_neuron_mlp("relu")
    data = np.array([[1.0]])

    with pytest.warns(RuntimeWarning, match="completeness tolerance"):
        coarse = uig.Explainer(model, [0.0], n_steps=1)(data)
    refined = uig.Explainer(model, [0.0], n_steps=2)(data)

    assert coarse.max_abs_completeness_error == pytest.approx(0.5)
    assert refined.max_abs_completeness_error < 1e-12


def test_completeness_warning_can_be_disabled():
    model = fitted_one_neuron_mlp("relu")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = uig.Explainer(
            model, [0.0], n_steps=1, check_completeness=False
        )([[1.0]])
    assert not recorded
    assert result.max_abs_completeness_error == pytest.approx(0.5)


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


@pytest.mark.parametrize("name", ["completeness_atol", "completeness_rtol"])
@pytest.mark.parametrize("value", [-1.0, True, "small"])
def test_invalid_completeness_tolerance_is_rejected(name, value):
    model = fitted_one_neuron_mlp()
    with pytest.raises(ValueError, match="non-negative"):
        uig.Explainer(model, [0.0], **{name: value})


def test_check_completeness_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        uig.Explainer(fitted_one_neuron_mlp(), [0.0], check_completeness=1)
