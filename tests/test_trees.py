"""Integration tests for the optional TreeIG backend."""

import numpy as np
import pytest

import unifiedig as uig


pytest.importorskip("treeig")


def _regression_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 4))
    y = X[:, 0] - X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]
    return X, y


def test_decision_tree_completeness_and_api():
    from sklearn.tree import DecisionTreeRegressor

    X, y = _regression_data()
    model = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    explanation = uig.Explainer(model, X[0])(X[10:18])

    assert explanation.values.shape == (8, 4)
    np.testing.assert_allclose(
        explanation.base_values,
        np.repeat(model.predict(X[[0]]), 8),
    )
    np.testing.assert_allclose(explanation.completeness_error, 0.0, atol=1e-10)


def test_random_forest_uses_shared_baseline_distribution():
    from sklearn.ensemble import RandomForestRegressor

    X, y = _regression_data(seed=1)
    model = RandomForestRegressor(
        n_estimators=8, max_depth=5, random_state=1
    ).fit(X, y)
    data = X[20:26]
    baselines = X[:3]
    explanation = uig.Explainer(model, baselines)(data)

    expected_base_value = model.predict(baselines).mean()
    np.testing.assert_allclose(explanation.base_values, expected_base_value)
    np.testing.assert_allclose(
        explanation.values.sum(axis=1),
        model.predict(data) - expected_base_value,
        atol=1e-10,
    )


def test_binary_gradient_boosting_uses_decision_scores():
    from sklearn.ensemble import GradientBoostingClassifier

    X, score = _regression_data(seed=2)
    y = (score > np.median(score)).astype(int)
    model = GradientBoostingClassifier(
        n_estimators=12, max_depth=2, random_state=2
    ).fit(X, y)
    data = X[30:37]
    explanation = uig.Explainer(model, X[0])(data)

    assert explanation.output_names == [str(model.classes_[1])]
    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        model.decision_function(data),
        atol=1e-10,
    )


def test_multiclass_tree_classifier_is_rejected():
    from sklearn.ensemble import GradientBoostingClassifier

    X, score = _regression_data(seed=3)
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    model = GradientBoostingClassifier(random_state=3).fit(X, y)

    with pytest.raises(ValueError, match="only binary tree classifiers"):
        uig.Explainer(model, X[0])
