import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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


def test_empty_data_batch_is_rejected():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-empty"):
        uig.Explainer(model, [0.0, 0.0])(np.empty((0, 2)))


def test_missing_tree_extra_has_actionable_error(monkeypatch):
    model = DecisionTreeRegressor(random_state=0).fit(
        np.eye(2), np.array([1.0, 2.0])
    )
    real_import = builtins.__import__

    def without_treeig(name, *args, **kwargs):
        if name == "treeig":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_treeig)
    with pytest.raises(ImportError, match=r"unifiedig\[trees\]"):
        uig.Explainer(model, [0.0, 0.0])


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


def test_shap_waterfall_and_beeswarm_accept_unifiedig_explanation():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shap = pytest.importorskip("shap")
    model = LinearRegression().fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]),
        np.array([0.0, 2.0, 1.0]),
    )
    result = uig.Explainer(model, [0.0, 0.0])(
        np.array([[1.0, 0.5], [0.5, -0.5]])
    ).to_shap()

    shap.plots.waterfall(result[0], show=False)
    shap.plots.beeswarm(result, show=False)

    import matplotlib.pyplot as plt

    plt.close("all")


def test_dataframe_columns_become_feature_names():
    training = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["age", "income"])
    model = LinearRegression().fit(training, np.array([0.0, 1.0]))
    result = uig.Explainer(model, [0.0, 0.0])(training.iloc[[1]])
    assert result.feature_names == ["age", "income"]


def test_unfitted_supported_model_is_rejected():
    with pytest.raises(ValueError, match="fitted"):
        uig.Explainer(LinearRegression(), [0.0])


def test_explanation_rejects_inconsistent_parallel_array_shapes():
    with pytest.raises(ValueError, match="non-empty"):
        uig.Explanation(
            values=np.empty((0, 2)),
            base_values=np.empty(0),
            data=np.empty((0, 2)),
        )

    with pytest.raises(ValueError, match="values must match data"):
        uig.Explanation(
            values=np.zeros((2, 3)),
            base_values=np.zeros(2),
            data=np.zeros((2, 2)),
        )


def test_explanation_normalizes_array_like_fields():
    explanation = uig.Explanation(
        values=[[1.0, -1.0]],
        base_values=[2.0],
        data=[[3.0, 4.0]],
        completeness_error=[0.0],
    )

    assert isinstance(explanation.values, np.ndarray)
    assert isinstance(explanation.base_values, np.ndarray)
    assert isinstance(explanation.data, np.ndarray)
    assert isinstance(explanation.completeness_error, np.ndarray)

    with pytest.raises(ValueError, match="base_values must have shape"):
        uig.Explanation(
            values=np.zeros((2, 2, 3)),
            base_values=np.zeros(2),
            data=np.zeros((2, 2)),
        )


def test_multiclass_contrast_validation():
    explanation = uig.Explanation(
        values=np.zeros((2, 3, 3)),
        base_values=np.zeros((2, 3)),
        data=np.zeros((2, 3)),
        output_names=["a", "b", "c"],
    )

    assert explanation.contrast(2, 0).output_names == ["c - a"]
    with pytest.raises(ValueError, match="different"):
        explanation.contrast("a", "a")
    with pytest.raises(ValueError, match="unknown"):
        explanation.contrast("missing", "a")
    with pytest.raises(IndexError, match="output index"):
        explanation.contrast(3, 0)
    with pytest.raises(TypeError, match="integer indices or names"):
        explanation.contrast(1.5, 0)


def test_output_names_must_align_with_output_axis():
    with pytest.raises(ValueError, match="output_names"):
        uig.Explanation(
            values=np.zeros((2, 3, 3)),
            base_values=np.zeros((2, 3)),
            data=np.zeros((2, 3)),
            output_names=["a", "b"],
        )


def test_feature_names_must_align_with_tabular_features():
    with pytest.raises(ValueError, match="feature_names"):
        uig.Explanation(
            values=np.zeros((2, 3)),
            base_values=np.zeros(2),
            data=np.zeros((2, 3)),
            feature_names=["a", "b"],
        )


def test_output_kind_validation_and_known_model_semantics():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="output_kind must be"):
        uig.Explainer(model, [0.0, 0.0], output_kind="scores")
    with pytest.raises(ValueError, match="only needed"):
        uig.Explainer(model, [0.0, 0.0], output_kind="regression")


def test_multiclass_to_shap_preserves_output_axis():
    shap = pytest.importorskip("shap")
    explanation = uig.Explanation(
        values=np.zeros((2, 3, 3)),
        base_values=np.zeros((2, 3)),
        data=np.zeros((2, 3)),
        output_names=["a", "b", "c"],
    )

    converted = explanation.to_shap()

    assert isinstance(converted, shap.Explanation)
    assert converted.values.shape == (2, 3, 3)
    assert converted.output_names == ["a", "b", "c"]


def test_multiclass_contrast_works_with_scalar_shap_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("shap")
    explanation = uig.Explanation(
        values=np.zeros((2, 3, 3)),
        base_values=np.zeros((2, 3)),
        data=np.zeros((2, 3)),
        output_names=["a", "b", "c"],
    )

    contrast = explanation.contrast("a", "c").to_shap()

    import shap
    import matplotlib.pyplot as plt

    assert contrast.output_names == "a - c"
    shap.plots.beeswarm(contrast, show=False)
    shap.plots.waterfall(contrast[0], show=False)
    plt.close("all")
