import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)

import unifiedig as uig


pytest.importorskip("treeig")


def test_numeric_tree_regression_uses_weighted_background():
    X, y = make_regression(
        n_samples=80, n_features=3, noise=0.1, random_state=11
    )
    model = HistGradientBoostingRegressor(
        max_iter=20, max_leaf_nodes=7, random_state=11
    ).fit(X, y)
    baselines = X[:2]
    weights = np.array([0.25, 0.75])

    with pytest.warns(RuntimeWarning, match="path-event detection"):
        explainer = uig.Explainer(
            model,
            baselines,
            baseline_weights=weights,
            fallback="tree_numeric",
            tree_grid_size=128,
        )
    result = explainer(X[20:24])

    np.testing.assert_allclose(result.base_values, weights @ model.predict(baselines))
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(X[20:24]),
        atol=1e-10,
    )


def test_numeric_tree_multiclass_uses_centered_raw_scores():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=12,
    )
    model = HistGradientBoostingClassifier(
        max_iter=15, max_leaf_nodes=5, random_state=12
    ).fit(X, y)

    with pytest.warns(RuntimeWarning, match="path-event detection"):
        explainer = uig.Explainer(
            model,
            X[:2],
            fallback="tree_numeric",
            tree_grid_size=128,
        )
    result = explainer(X[20:23])

    raw_scores = model.decision_function(X[20:23])
    centered = raw_scores - raw_scores.mean(axis=1, keepdims=True)
    assert result.output_names == [str(item) for item in model.classes_]
    assert result.values.shape == (3, 4, 3)
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        centered,
        atol=1e-10,
    )


def test_probability_only_tree_classifier_remains_explicitly_rejected():
    X, y = make_classification(n_samples=50, n_features=4, random_state=13)
    model = RandomForestClassifier(n_estimators=5, random_state=13).fit(X, y)

    with pytest.raises(TypeError, match="probability-to-score"):
        uig.Explainer(model, X[0], fallback="tree_numeric")


def test_numeric_tree_options_are_validated():
    X, y = make_regression(n_samples=30, n_features=2, random_state=14)
    model = HistGradientBoostingRegressor(random_state=14).fit(X, y)

    with pytest.raises(ValueError, match="tree_grid_size"):
        uig.Explainer(model, X[0], tree_grid_size=0)
