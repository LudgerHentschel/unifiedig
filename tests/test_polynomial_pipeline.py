import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import unifiedig as uig
import unifiedig.backends.skgrad as skgrad_backend


def _cubic_pipeline():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] ** 3 - 0.8 * X[:, 0] * X[:, 1] + X[:, 2] ** 2
    model = make_pipeline(
        PolynomialFeatures(3, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    ).fit(X, y)
    return model, X


def test_default_quadrature_is_capped_at_exact_polynomial_order(monkeypatch):
    model, X = _cubic_pipeline()
    calls = 0
    original = skgrad_backend.skgrad.input_gradient

    def counting_gradient(model, data, target=None):
        nonlocal calls
        calls += 1
        return original(model, data, target=target)

    monkeypatch.setattr(skgrad_backend.skgrad, "input_gradient", counting_gradient)
    result = uig.Explainer(model, X[10:14], n_steps=128)(X[:6])

    # A cubic needs two Gauss-Legendre nodes for each baseline, not 128.
    assert calls == 2
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(X[:6]),
        atol=1e-11,
    )


def test_user_can_request_fewer_than_the_exact_polynomial_steps(monkeypatch):
    model, X = _cubic_pipeline()
    calls = 0
    original = skgrad_backend.skgrad.input_gradient

    def counting_gradient(model, data, target=None):
        nonlocal calls
        calls += 1
        return original(model, data, target=target)

    monkeypatch.setattr(skgrad_backend.skgrad, "input_gradient", counting_gradient)
    uig.Explainer(
        model, X[10:14], n_steps=1, check_completeness=False
    )(X[:2])

    assert calls == 1
