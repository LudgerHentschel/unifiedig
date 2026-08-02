import numpy as np
from sklearn.linear_model import LinearRegression

import unifiedig as uig


def test_linear_regression_is_complete_for_shared_baseline():
    model = LinearRegression().fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([1.0, 3.0, -2.0, 0.0]),
    )
    data = np.array([[0.25, 0.75], [2.0, -1.0]])
    baseline = np.array([0.5, 0.5])

    result = uig.Explainer(model, baseline)(data)

    np.testing.assert_allclose(result.values.sum(axis=1), model.predict(data) - result.base_values)
    np.testing.assert_allclose(result.base_values, model.predict(baseline.reshape(1, -1)).repeat(2))
    assert result.data.shape == result.values.shape == (2, 2)
    assert result.max_abs_completeness_error < 1e-12


def test_scalar_baseline_and_single_sample_are_normalized():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    result = uig.Explainer(model, 0.0)(np.array([1.0, 2.0]))
    assert len(result) == 1
    assert result.data.shape == (1, 2)
    np.testing.assert_allclose(result.values.sum(axis=1) + result.base_values, model.predict(result.data))


def test_per_sample_baselines_are_complete():
    model = LinearRegression().fit(np.eye(2), np.array([1.0, 2.0]))
    data = np.array([[1.0, 2.0], [3.0, -1.0]])
    baselines = np.array([[0.0, 0.0], [1.0, 1.0]])

    result = uig.Explainer(model, baselines)(data)

    np.testing.assert_allclose(result.base_values, model.predict(baselines))
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, model.predict(data)
    )


def test_multi_output_regression_is_complete():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    targets = np.array([[1.0, -1.0], [3.0, 0.0], [-2.0, 4.0]])
    model = LinearRegression().fit(data, targets)
    explained = np.array([[0.25, 0.75], [2.0, -1.0]])
    baseline = np.array([0.5, 0.5])

    result = uig.Explainer(model, baseline)(explained)

    assert result.values.shape == (2, 2, 2)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values, model.predict(explained)
    )
