import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import unifiedig as uig


def _data(seed=70):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 3))
    score = np.sin(X[:, 0]) + 0.4 * X[:, 1] ** 2 - 0.3 * X[:, 2]
    return X, score


def test_gaussian_process_regression_fallback_is_complete():
    X, y = _data()
    model = GaussianProcessRegressor(kernel=RBF(1.2), alpha=1e-6).fit(X, y)

    with pytest.warns(RuntimeWarning, match="finite-difference gradients"):
        explainer = uig.Explainer(
            model,
            X[:2],
            n_steps=32,
            fallback="finite_difference",
            finite_difference_batch_size=4,
            completeness_atol=2e-5,
        )
    result = explainer(X[10:14])

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(X[10:14]),
        atol=2e-5,
    )
    assert result.max_abs_completeness_error < 2e-5


def test_binary_classifier_fallback_uses_decision_scores():
    X, score = _data(seed=71)
    y = (score > np.median(score)).astype(int)
    model = SVC(kernel="rbf", gamma=0.6, C=2.0).fit(X, y)

    with pytest.warns(RuntimeWarning, match="finite-difference gradients"):
        explainer = uig.Explainer(
            model,
            X[0],
            n_steps=32,
            fallback="finite_difference",
            completeness_atol=2e-5,
        )
    result = explainer(X[10:14])

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.decision_function(X[10:14]),
        atol=2e-5,
    )
    assert result.output_names == [str(model.classes_[1])]


def test_fallback_is_explicit_and_rejects_probability_only_classifiers():
    X, score = _data(seed=72)
    y = (score > np.median(score)).astype(int)
    model = GaussianNB().fit(X, y)

    with pytest.raises(TypeError, match="fallback='finite_difference'"):
        uig.Explainer(model, X[0])
    with pytest.raises(TypeError, match="requires decision_function"):
        uig.Explainer(model, X[0], fallback="finite_difference")


def test_fallback_rejects_tree_estimators_not_supported_by_treeig():
    X, score = _data(seed=73)
    y = (score > np.median(score)).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=73).fit(X, y)

    with pytest.raises(TypeError, match="not appropriate"):
        uig.Explainer(model, X[0], fallback="finite_difference")


def test_finite_difference_options_are_validated():
    X, y = _data(seed=74)
    model = GaussianProcessRegressor().fit(X, y)

    with pytest.raises(ValueError, match="fallback"):
        uig.Explainer(model, X[0], fallback="automatic")
    with pytest.raises(ValueError, match="step"):
        uig.Explainer(model, X[0], finite_difference_step=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        uig.Explainer(model, X[0], finite_difference_batch_size=0)


def test_specialized_backend_takes_precedence_over_requested_fallback():
    X, y = _data(seed=75)
    model = Ridge().fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = uig.Explainer(
            model, X[:2], fallback="finite_difference", n_steps=1
        )(X[10:14])

    assert not caught
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(X[10:14]),
    )
