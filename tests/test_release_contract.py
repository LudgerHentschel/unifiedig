"""Release-level invariants across real dependencies and public interfaces."""

import numpy as np
import pytest
from cbaseline import background
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

import unifiedig as uig


@pytest.mark.parametrize("model", [Ridge(alpha=0.2), DecisionTreeRegressor(max_depth=3, random_state=0)])
def test_real_cbaseline_matches_explicit_distribution_and_weighted_paths(model):
    rng = np.random.default_rng(19)
    training = rng.normal(size=(30, 3))
    model.fit(training, training[:, 0] ** 2 - training[:, 1])
    predictions = model.predict(training)
    f0 = float(np.quantile(predictions, 0.4))
    bg = background(predictions=predictions, f0=f0, features=training, weighting="calibrated")
    data = training[10:12]
    result = uig.Explainer(model, bg)(data)
    explicit = uig.Explainer(model, bg.rows, baseline_weights=bg.weights)(data)
    paths = [uig.Explainer(model, row)(data) for row in bg.rows]
    weights = np.asarray(bg.weights) / np.sum(bg.weights)
    np.testing.assert_allclose(result.values, explicit.values, atol=1e-10)
    np.testing.assert_allclose(result.values, np.einsum("b,bnf->nf", weights, [p.values for p in paths]), atol=1e-10)
    np.testing.assert_allclose(result.base_values, f0, atol=1e-8)
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), model.predict(data), atol=1e-8)
    point = uig.Explainer(model, bg.rows[0])(data)
    singleton = uig.Explainer(model, bg.rows[:1])(data)
    np.testing.assert_allclose(point.values, singleton.values)


class SmoothRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X) @ np.array([2.0, -3.0]) + 0.5


def test_analytic_and_finite_difference_interfaces_agree():
    training = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    smooth = SmoothRegressor().fit(training)
    analytic = LinearRegression().fit(training, smooth.predict(training))
    baseline = np.array([[-0.2, 0.4], [0.6, -0.1]])
    expected = uig.Explainer(analytic, baseline, baseline_weights=[1, 3])(training)
    with pytest.warns(RuntimeWarning, match="finite-difference gradients"):
        numerical = uig.Explainer(smooth, baseline, baseline_weights=[1, 3], fallback="finite_difference")(training)
    np.testing.assert_allclose(numerical.values, expected.values, atol=1e-9)
    np.testing.assert_allclose(numerical.base_values, expected.base_values, atol=1e-9)


def test_sklearn_and_torch_interfaces_agree():
    torch = pytest.importorskip("torch")
    training = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    model = LinearRegression().fit(training, training @ [2., -3.] + 0.5)
    native = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        native.weight.copy_(torch.tensor(model.coef_[None, :]))
        native.bias.copy_(torch.tensor([model.intercept_]))
    kwargs = {"baseline_weights": [1, 3]}
    analytic = uig.Explainer(model, training[:2], **kwargs)(training)
    autodiff = uig.Explainer(native, training[:2], **kwargs)(training)
    np.testing.assert_allclose(autodiff.values, analytic.values, atol=1e-10)
    np.testing.assert_allclose(autodiff.base_values, analytic.base_values, atol=1e-10)


@pytest.mark.parametrize("framework", ["xgboost", "lightgbm"])
def test_external_tree_regressor_wrapper_and_booster_agree(framework):
    package = pytest.importorskip(framework)
    rng = np.random.default_rng(9)
    training = rng.normal(size=(40, 3))
    target = training[:, 0] - training[:, 1] ** 2
    if framework == "xgboost":
        model = package.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1).fit(training, target)
        booster = model.get_booster()
    else:
        model = package.LGBMRegressor(n_estimators=3, max_depth=2, min_child_samples=2, verbosity=-1, n_jobs=1).fit(training, target)
        booster = model.booster_
    data = training[5:8]
    baseline = training[:2]
    wrapped = uig.Explainer(model, baseline)(data)
    native = uig.Explainer(booster, baseline)(data)
    np.testing.assert_allclose(wrapped.values, native.values, atol=2e-6)
    np.testing.assert_allclose(wrapped.base_values, native.base_values, atol=2e-6)
    np.testing.assert_allclose(wrapped.base_values + wrapped.values.sum(axis=1), model.predict(data), atol=2e-6)
