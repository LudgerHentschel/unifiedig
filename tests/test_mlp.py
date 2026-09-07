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


def test_baseline_distribution_is_shared_for_deep_mlp():
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
    baselines = np.array([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4]])

    result = uig.Explainer(model, baselines)(data)

    np.testing.assert_allclose(result.base_values, model.predict(baselines).mean())
    assert result.max_abs_completeness_error < 1e-7


def test_scalar_mlp_batches_baseline_paths_through_input_gradient(monkeypatch):
    import unifiedig.backends.skgrad as skgrad_backend

    rng = np.random.default_rng(81)
    training = rng.normal(size=(60, 2))
    model = MLPRegressor(
        hidden_layer_sizes=(4, 3), activation="tanh", solver="lbfgs",
        max_iter=5000, random_state=3,
    ).fit(training, training[:, 0] - training[:, 1] ** 2)
    calls = 0
    original = skgrad_backend.skgrad.input_gradient

    def counting_gradient(model, data, target=None):
        nonlocal calls
        calls += 1
        return original(model, data, target=target)

    monkeypatch.setattr(skgrad_backend.skgrad, "input_gradient", counting_gradient)
    result = uig.Explainer(
        model, training[:10], n_steps=16, gradient_batch_size=4
    )(training[10:12])

    # Two observations permit two baselines per path batch: 5 batches * 16 nodes.
    assert calls == 80
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


def test_omitted_n_steps_refines_from_16_to_32(monkeypatch):
    model = fitted_one_neuron_mlp("tanh")
    explainer = uig.Explainer(model, [0.0])
    monkeypatch.setattr(
        type(explainer),
        "_completeness_failed",
        lambda self, error, output: self.n_steps < 32,
    )

    explainer([[1.0]])

    assert explainer.n_steps == 32


def test_explicit_n_steps_disables_automatic_refinement(monkeypatch):
    model = fitted_one_neuron_mlp("tanh")
    explainer = uig.Explainer(model, [0.0], n_steps=16)
    monkeypatch.setattr(
        type(explainer),
        "_completeness_failed",
        lambda self, error, output: True,
    )

    with pytest.warns(RuntimeWarning, match="completeness tolerance"):
        explainer([[1.0]])

    assert explainer.n_steps == 16


def test_disabling_completeness_disables_automatic_refinement(monkeypatch):
    model = fitted_one_neuron_mlp("tanh")
    explainer = uig.Explainer(model, [0.0], check_completeness=False)
    monkeypatch.setattr(
        type(explainer),
        "_completeness_failed",
        lambda self, error, output: True,
    )

    explainer([[1.0]])

    assert explainer.n_steps == 16


def test_multiclass_mlp_classifier_uses_centered_logits():
    model = MLPClassifier(
        hidden_layer_sizes=(2,), activation="tanh", solver="lbfgs", random_state=0
    ).fit(
        np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1, 2, 2]),
    )
    model.coefs_ = [
        np.array([[0.7, -0.4]]),
        np.array([[0.6, -0.3, 0.2], [-0.5, 0.4, 0.1]]),
    ]
    model.intercepts_ = [np.array([0.1, -0.2]), np.array([0.2, -0.1, 0.3])]
    data = np.array([[-0.5], [0.5], [1.5]])
    baselines = np.array([[-1.0], [0.0], [1.0]])
    weights = np.array([0.2, 0.3, 0.5])

    result = uig.Explainer(
        model, baselines, baseline_weights=weights
    )(data)

    hidden = np.tanh(data @ model.coefs_[0] + model.intercepts_[0])
    logits = hidden @ model.coefs_[1] + model.intercepts_[1]
    centered = logits - logits.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        centered,
        atol=1e-8,
    )


@pytest.mark.parametrize("n_steps", [0, -1, 1.5, True])
def test_invalid_n_steps_is_rejected(n_steps):
    model = MLPRegressor(
        hidden_layer_sizes=(2,), solver="lbfgs", max_iter=1000, random_state=0
    ).fit(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0])
    )
    with pytest.raises(ValueError, match="positive integer"):
        uig.Explainer(model, [0.0], n_steps=n_steps)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_gradient_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="gradient_batch_size"):
        uig.Explainer(
            fitted_one_neuron_mlp(), [0.0], gradient_batch_size=batch_size
        )


@pytest.mark.parametrize("name", ["completeness_atol", "completeness_rtol"])
@pytest.mark.parametrize("value", [-1.0, True, "small"])
def test_invalid_completeness_tolerance_is_rejected(name, value):
    model = fitted_one_neuron_mlp()
    with pytest.raises(ValueError, match="non-negative"):
        uig.Explainer(model, [0.0], **{name: value})


def test_check_completeness_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        uig.Explainer(fitted_one_neuron_mlp(), [0.0], check_completeness=1)


@pytest.mark.parametrize("loss", [False, True])
@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_incomplete_policy_on_relu_path(loss, policy):
    model = fitted_one_neuron_mlp()
    cls = uig.LossExplainer if loss else uig.Explainer
    explainer = cls(model, [0.0], n_steps=1, on_incomplete=policy)
    args = ([[1.0]], [0.0]) if loss else ([[1.0]],)
    if policy == "raise":
        with pytest.raises(RuntimeError, match="completeness tolerance"):
            explainer(*args)
    else:
        with pytest.warns(RuntimeWarning, match="completeness tolerance"):
            result = explainer(*args)
        assert result.max_abs_completeness_error > 0.1
    assert explainer.n_steps == 1


@pytest.mark.parametrize("loss", [False, True])
def test_strict_policy_respects_disabled_check(loss):
    cls = uig.LossExplainer if loss else uig.Explainer
    explainer = cls(fitted_one_neuron_mlp(), [0.0],
                    on_incomplete="raise", check_completeness=False, n_steps=1)
    args = ([[1.0]], [0.0]) if loss else ([[1.0]],)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = explainer(*args)
    assert not recorded
    assert result.max_abs_completeness_error > 0.1


@pytest.mark.parametrize("loss", [False, True])
def test_strict_policy_returns_complete_result(loss):
    cls = uig.LossExplainer if loss else uig.Explainer
    explainer = cls(fitted_one_neuron_mlp(), [0.6], on_incomplete="raise")
    args = ([[1.0]], [0.0]) if loss else ([[1.0]],)
    assert explainer(*args).max_abs_completeness_error < 1e-12


@pytest.mark.parametrize("loss", [False, True])
def test_strict_policy_raises_only_after_refinement_exhausted(loss):
    model = fitted_one_neuron_mlp()
    model.intercepts_[0][:] = -0.37
    cls = uig.LossExplainer if loss else uig.Explainer
    explainer = cls(model, [0.0], on_incomplete="raise",
                    completeness_atol=1e-12, completeness_rtol=1e-12)
    args = ([[1.0]], [0.0]) if loss else ([[1.0]],)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="completeness tolerance"):
            explainer(*args)
        assert explainer.n_steps == 64


@pytest.mark.parametrize("loss", [False, True])
@pytest.mark.parametrize("policy", [None, "ignore", True])
def test_invalid_incomplete_policy(loss, policy):
    cls = uig.LossExplainer if loss else uig.Explainer
    with pytest.raises(ValueError, match="on_incomplete"):
        cls(fitted_one_neuron_mlp(), [0.0], on_incomplete=policy)
