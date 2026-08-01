import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import unifiedig as uig


def test_public_api_and_unsupported_model_error():
    assert uig.Explainer is not None
    assert uig.Explanation is not None
    with pytest.raises(TypeError, match="no Unified IG backend"):
        uig.Explainer(DummyRegressor(), 0.0)


def test_bad_baseline_shape_is_rejected():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="baseline"):
        uig.Explainer(model, [0.0, 0.0, 0.0])(np.eye(2))


def test_to_shap_has_actionable_optional_dependency_error(monkeypatch):
    explanation = uig.Explanation(
        values=np.zeros((1, 1)), base_values=np.zeros(1), data=np.zeros((1, 1))
    )
    real_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    with pytest.raises(ImportError, match=r"unifiedig\[shap\]"):
        explanation.to_shap()


def test_to_shap_returns_compatible_explanation():
    shap = pytest.importorskip("shap")
    explanation = uig.Explanation(
        values=np.array([[1.0, -2.0]]),
        base_values=np.array([3.0]),
        data=np.array([[4.0, 5.0]]),
        feature_names=["a", "b"],
    )

    converted = explanation.to_shap()

    assert isinstance(converted, shap.Explanation)
    np.testing.assert_array_equal(converted.values, explanation.values)
    assert converted.feature_names == ["a", "b"]


def test_dataframe_columns_become_feature_names():
    training = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["age", "income"])
    model = LinearRegression().fit(training, np.array([0.0, 1.0]))
    result = uig.Explainer(model, [0.0, 0.0])(training.iloc[[1]])
    assert result.feature_names == ["age", "income"]


def test_unfitted_supported_model_is_rejected():
    with pytest.raises(ValueError, match="fitted"):
        uig.Explainer(LinearRegression(), [0.0])
