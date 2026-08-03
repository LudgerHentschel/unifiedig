import numpy as np
import pytest
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)

import unifiedig as uig
import unifiedig.backends.skgrad as skgrad_backend


def _data():
    rng = np.random.default_rng(41)
    training = rng.normal(size=(100, 4))
    score = 1.4 * training[:, 0] - 0.8 * training[:, 1] + 0.3 * training[:, 2]
    return training, score


@pytest.mark.parametrize(
    "model",
    [
        Ridge(alpha=0.7),
        Lasso(alpha=0.01, max_iter=5000),
        ElasticNet(alpha=0.01, l1_ratio=0.4, max_iter=5000),
    ],
)
def test_regularized_regressors_are_exact_and_complete(model):
    training, target = _data()
    model.fit(training, target)
    data = training[:6]
    baselines = training[10:14]

    explanation = uig.Explainer(model, baselines)(data)

    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        model.predict(data),
    )
    np.testing.assert_allclose(explanation.base_values, model.predict(baselines).mean())
    np.testing.assert_allclose(explanation.completeness_error, 0.0, atol=1e-12)


@pytest.mark.parametrize("model", [LogisticRegression(), RidgeClassifier(alpha=0.5)])
def test_affine_classifiers_use_binary_decision_scores(model):
    training, score = _data()
    target = (score > np.median(score)).astype(int)
    model.fit(training, target)
    data = training[:6]
    baselines = training[10:14]

    explanation = uig.Explainer(model, baselines)(data)

    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        model.decision_function(data),
    )
    assert explanation.output_names == [str(model.classes_[1])]
    np.testing.assert_allclose(explanation.completeness_error, 0.0, atol=1e-12)


def test_multiclass_ridge_classifier_uses_centered_scores():
    training, score = _data()
    target = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    model = RidgeClassifier().fit(training, target)
    data = training[:6]
    baselines = training[10:14]
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    explanation = uig.Explainer(
        model, baselines, baseline_weights=weights
    )(data)

    scores = model.decision_function(data)
    centered_scores = scores - scores.mean(axis=1, keepdims=True)
    assert explanation.values.shape == (6, training.shape[1], 3)
    assert explanation.output_names == [str(item) for item in model.classes_]
    np.testing.assert_allclose(explanation.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(explanation.base_values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        centered_scores,
        atol=1e-12,
    )


def test_constant_jacobian_fast_path_ignores_quadrature_count(monkeypatch):
    training, target = _data()
    model = Ridge(alpha=0.7).fit(training, target)
    calls = 0
    original = skgrad_backend.skgrad.input_jacobian

    def counting_jacobian(model, data):
        nonlocal calls
        calls += 1
        return original(model, data)

    monkeypatch.setattr(skgrad_backend.skgrad, "input_jacobian", counting_jacobian)
    uig.Explainer(model, training[10:13], n_steps=128)(training[:5])

    assert calls == 3
