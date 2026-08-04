import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
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


def _centered_log_probabilities(model, X, floor):
    probabilities = np.maximum(model.predict_proba(X), floor)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(probabilities)
    return log_probabilities - log_probabilities.mean(axis=1, keepdims=True)


def _forest_score_crossing_oracle(model, baseline, data, floor, target=None):
    """Enumerate sklearn split crossings to validate numerical allocations."""
    direction = data - baseline
    crossings = {}
    for estimator in model.estimators_:
        tree = estimator.tree_
        for feature, threshold in zip(tree.feature, tree.threshold):
            if feature < 0 or direction[feature] == 0.0:
                continue
            time = (threshold - baseline[feature]) / direction[feature]
            if 0.0 < time < 1.0:
                crossings.setdefault(float(time), set()).add(int(feature))

    times = sorted(crossings)
    attribution = np.zeros(data.shape[0])
    for index, time in enumerate(times):
        features = crossings[time]
        assert len(features) == 1  # Different-feature ties need a convention.
        left = 0.5 * ((times[index - 1] if index else 0.0) + time)
        right = 0.5 * (time + (times[index + 1] if index + 1 < len(times) else 1.0))
        points = baseline + np.array([left, right])[:, None] * direction
        scores = _centered_log_probabilities(model, points, floor)
        if scores.shape[1] == 2:
            jump = scores[1, 1] - scores[1, 0] - scores[0, 1] + scores[0, 0]
        else:
            jump = scores[1, target] - scores[0, target]
        attribution[features.pop()] += jump
    return attribution


def test_binary_probability_tree_uses_derived_log_odds():
    X, y = make_classification(n_samples=50, n_features=4, random_state=13)
    model = RandomForestClassifier(n_estimators=5, random_state=13).fit(X, y)
    floor = 1e-6
    baselines = X[:2]
    weights = np.array([0.3, 0.7])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explainer = uig.Explainer(
            model,
            baselines,
            baseline_weights=weights,
            fallback="tree_numeric",
            probability_floor=floor,
            tree_grid_size=128,
        )
    assert any("deriving binary log odds" in str(item.message) for item in caught)
    assert any("path-event detection" in str(item.message) for item in caught)
    result = explainer(X[10:14])

    centered = _centered_log_probabilities(model, X[10:14], floor)
    expected_margin = centered[:, 1] - centered[:, 0]
    baseline_scores = _centered_log_probabilities(model, baselines, floor)
    expected_base = weights @ (baseline_scores[:, 1] - baseline_scores[:, 0])
    assert result.output_names == [str(model.classes_[1])]
    np.testing.assert_allclose(result.base_values, expected_base)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        expected_margin,
        atol=1e-12,
    )


def test_binary_probability_forest_allocations_match_split_crossings():
    X, y = make_classification(n_samples=60, n_features=4, random_state=17)
    model = RandomForestClassifier(
        n_estimators=7, max_depth=3, random_state=17
    ).fit(X, y)
    baselines = X[:2]
    weights = np.array([0.4, 0.6])
    explained = X[20]
    floor = 1e-6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = uig.Explainer(
            model,
            baselines,
            baseline_weights=weights,
            fallback="tree_numeric",
            probability_floor=floor,
            tree_grid_size=1024,
        )(explained)

    expected = sum(
        weight * _forest_score_crossing_oracle(
            model, baseline, explained, floor
        )
        for baseline, weight in zip(baselines, weights)
    )
    np.testing.assert_allclose(result.values[0], expected, atol=1e-12)


def test_multiclass_probability_tree_uses_centered_log_scores():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=15,
    )
    model = ExtraTreesClassifier(
        n_estimators=7, max_depth=4, random_state=15
    ).fit(X, y)
    floor = 1e-5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explainer = uig.Explainer(
            model,
            X[:2],
            fallback="tree_numeric",
            probability_floor=floor,
            tree_grid_size=1024,
        )
    assert any("centered multiclass" in str(item.message) for item in caught)
    assert any("path-event detection" in str(item.message) for item in caught)
    result = explainer(X[20:24])

    expected = _centered_log_probabilities(model, X[20:24], floor)
    assert result.output_names == [str(item) for item in model.classes_]
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        expected,
        atol=1e-12,
    )
    crossing_values = np.column_stack(
        [
            np.mean(
                [
                    _forest_score_crossing_oracle(
                        model, baseline, X[20], floor, target
                    )
                    for baseline in X[:2]
                ],
                axis=0,
            )
            for target in range(3)
        ]
    )
    np.testing.assert_allclose(result.values[0], crossing_values, atol=1e-12)


def test_probability_tree_requires_floor_when_path_reaches_zero():
    X = np.array([[0.0], [1.0]])
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=16).fit(X, [0, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explainer = uig.Explainer(
            model, X[0], fallback="tree_numeric", tree_grid_size=16
        )
    assert any("deriving binary log odds" in str(item.message) for item in caught)

    with pytest.raises(ValueError, match="probability_floor"):
        explainer(X)


def test_numeric_tree_options_are_validated():
    X, y = make_regression(n_samples=30, n_features=2, random_state=14)
    model = HistGradientBoostingRegressor(random_state=14).fit(X, y)

    with pytest.raises(ValueError, match="tree_grid_size"):
        uig.Explainer(model, X[0], tree_grid_size=0)
    for floor in (0.0, 1.0, -0.1, np.inf):
        with pytest.raises(ValueError, match="probability_floor"):
            uig.Explainer(model, X[0], probability_floor=floor)
