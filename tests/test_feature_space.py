from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, LogisticRegression
import unifiedig as uig


def data():
    rng = np.random.default_rng(52)
    X = rng.normal(size=(70, 3)) * [1, 4, 9] + [3, -2, 1]
    return X, X[:, 0] - .3*X[:, 1] + .1*X[:, 2]


@pytest.mark.parametrize("scaler", [StandardScaler(), RobustScaler(), MaxAbsScaler(), MinMaxScaler()])
def test_affine_scaling_invariance_and_weighted_background(scaler):
    X, y = data()
    model = Pipeline([("scale", scaler), ("regressor", Ridge())]).fit(X, y)
    background = SimpleNamespace(rows=X[10:13], weights=np.array([.2, .8, 0.]))
    original = uig.Explainer(model, background)(X[:4])
    selected = uig.Explainer(model, background, attribute_after="scale")(X[:4])
    np.testing.assert_allclose(selected.values, original.values, atol=1e-12)
    np.testing.assert_allclose(selected.base_values, original.base_values)
    np.testing.assert_allclose(selected.data, scaler.transform(X[:4]))
    assert selected.attribute_after == "scale"
    assert original.attribute_after is None
    assert selected.feature_names == ["x0", "x1", "x2"]
    with pytest.raises(ValueError, match="omitted"):
        uig.Explainer(model, background, attribute_after="scale", baseline_weights=[1, 1, 1])(X[:4])


@pytest.mark.parametrize("transformer", [PCA(2, whiten=True), PolynomialFeatures(2), SelectKBest(f_regression, k=2)])
def test_selected_space_matches_manual_attribution(transformer):
    X, y = data()
    model = Pipeline([("transform", transformer), ("reg", Ridge())]).fit(X, y)
    baselines = X[10:13]
    transformed = transformer.transform(X[:4])
    expected = uig.Explainer(model[-1], transformer.transform(baselines))(transformed)
    actual = uig.Explainer(model, baselines, attribute_after="transform")(X[:4])
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)
    np.testing.assert_allclose(actual.base_values, expected.base_values, atol=1e-12)
    assert actual.values.shape == transformed.shape
    assert len(actual.feature_names) == transformed.shape[1]
    np.testing.assert_allclose(actual.values.sum(axis=1)+actual.base_values, model.predict(X[:4]), atol=1e-12)
    # Baselines must be transformed row-by-row, not averaged first.
    if isinstance(transformer, PolynomialFeatures):
        mean_prediction = model.predict(baselines).mean()
        np.testing.assert_allclose(actual.base_values, mean_prediction)
        assert not np.isclose(mean_prediction, model.predict(baselines.mean(axis=0)[None])[0])


def test_nested_boundary_and_scalar_baseline():
    X, y = data()
    model = Pipeline([("pre", Pipeline([("scale", StandardScaler()), ("pca", PCA(2))])),
                      ("reg", Ridge())]).fit(X, y)
    for name, prefix, suffix in [("pre__scale", model[0][:1], Pipeline([("pca",model[0][1]),("reg",model[1])])),
                                 ("pre", model[0], model[1])]:
        actual = uig.Explainer(model, 0., attribute_after=name)(X[:4])
        expected = uig.Explainer(suffix, prefix.transform(np.zeros((1, 3))))(prefix.transform(X[:4]))
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)
        np.testing.assert_allclose(actual.base_values, expected.base_values)


def test_classification_contrast_and_loss_preserve_selected_space():
    X, _ = data()
    y = np.argmax((X-X.mean(axis=0))/X.std(axis=0), axis=1)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y)
    result = uig.Explainer(model, X[10:13], attribute_after="scale")(X[:4])
    assert result.values.shape == (4, 3, 3)
    assert result.contrast(0, 1).attribute_after == "scale"
    actual = uig.LossExplainer(model, X[10:13], attribute_after="scale", loss="log_loss", n_steps=64)(X[:4], y[:4])
    expected = uig.LossExplainer(model[-1], model[0].transform(X[10:13]), loss="log_loss", n_steps=64)(model[0].transform(X[:4]), y[:4])
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)
    assert actual.attribute_after == "scale"
    np.testing.assert_allclose(actual.data, model[0].transform(X[:4]))


def test_validation():
    X, y = data()
    model = Pipeline([("scale", StandardScaler()), ("reg", Ridge())]).fit(X, y)
    with pytest.raises(ValueError, match="unknown"):
        uig.Explainer(model, 0., attribute_after="missing")
    with pytest.raises(ValueError, match="final predictor"):
        uig.Explainer(model, 0., attribute_after="reg")
    with pytest.raises(TypeError, match="after"):
        uig.Explainer(model, 0., attribute_after=True)
    with pytest.raises(TypeError, match="Pipeline"):
        uig.Explainer(model[-1], 0., attribute_after="scale")


def test_dataframe_names_and_baseline_order():
    pd = pytest.importorskip("pandas")
    X, y = data()
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = Pipeline([("scale", StandardScaler()), ("reg", Ridge())]).fit(frame, y)
    result = uig.Explainer(model, frame.iloc[10:13], attribute_after="scale")(frame.iloc[:4])
    assert result.feature_names == ["a", "b", "c"]
    with pytest.raises(ValueError, match="order"):
        uig.Explainer(model, frame.iloc[10:13][["b", "a", "c"]], attribute_after="scale")(frame.iloc[:4])


def test_clipped_scaling_matches_explicit_processed_path():
    X, y = data()
    model = Pipeline([("scale", MinMaxScaler(clip=True)), ("reg", Ridge())]).fit(X, y)
    baseline = X.min(axis=0)-10
    actual = uig.Explainer(model, baseline, attribute_after="scale")(X[:4])
    expected = uig.Explainer(model[-1], model[0].transform(baseline[None]))(model[0].transform(X[:4]))
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)


def test_selected_space_uses_explicit_numerical_fallback():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = data()
    model = Pipeline([("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]).fit(X, y > np.median(y))
    with pytest.warns(RuntimeWarning, match="finite-difference"):
        explainer = uig.Explainer(model, X[10:13], attribute_after="scale", fallback="finite_difference")
    result = explainer(X[:4])
    assert result.attribute_after == "scale"
    np.testing.assert_allclose(result.values.sum(axis=1)+result.base_values,
                               model.decision_function(X[:4]), atol=1e-8)


def test_selected_space_routes_trees_to_treeig():
    pytest.importorskip("treeig")
    from sklearn.tree import DecisionTreeRegressor
    X, y = data()
    model = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeRegressor(max_depth=3, random_state=4))]).fit(X, y)
    result = uig.Explainer(model, X[10:13], attribute_after="scale")(X[:4])
    np.testing.assert_allclose(result.values.sum(axis=1)+result.base_values,
                               model.predict(X[:4]), atol=1e-10)
    assert result.attribute_after == "scale"
