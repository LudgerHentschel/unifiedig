import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import unifiedig as uig


catboost = pytest.importorskip("catboost")


def test_catboost_regressor_uses_raw_prediction_events():
    X, y = make_regression(
        n_samples=60, n_features=3, noise=0.1, random_state=21
    )
    model = catboost.CatBoostRegressor(
        iterations=8,
        depth=3,
        verbose=False,
        random_seed=21,
        allow_writing_files=False,
    ).fit(X, y)

    with pytest.warns(RuntimeWarning, match="path-event detection"):
        explainer = uig.Explainer(
            model,
            X[:2],
            fallback="tree_numeric",
            tree_grid_size=128,
        )
    result = explainer(X[20:23])

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(X[20:23], prediction_type="RawFormulaVal"),
        atol=1e-10,
    )


def test_catboost_multiclass_uses_centered_raw_formula_values():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=22,
    )
    model = catboost.CatBoostClassifier(
        iterations=8,
        depth=3,
        verbose=False,
        random_seed=22,
        allow_writing_files=False,
    ).fit(X, y)

    with pytest.warns(RuntimeWarning, match="path-event detection"):
        explainer = uig.Explainer(
            model,
            X[:2],
            fallback="tree_numeric",
            tree_grid_size=128,
        )
    result = explainer(X[20:23])

    raw = np.asarray(
        model.predict(X[20:23], prediction_type="RawFormulaVal")
    )
    centered = raw - raw.mean(axis=1, keepdims=True)
    assert result.output_names == [str(item) for item in model.classes_]
    np.testing.assert_allclose(result.values.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        centered,
        atol=1e-10,
    )
