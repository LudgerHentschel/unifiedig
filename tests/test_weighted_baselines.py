from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import unifiedig as uig


@dataclass
class Background:
    """Minimal stand-in for the public CBaseline interface."""

    rows: np.ndarray
    weights: np.ndarray


def _problem():
    training = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    model = LinearRegression().fit(
        training, np.array([1.0, 3.0, -2.0, 0.0])
    )
    rows = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
    weights = np.array([1.0, 2.0, 7.0])
    data = np.array([[0.25, 0.75], [2.0, -1.0]])
    return model, rows, weights, data


def test_explicit_weighted_baseline_is_complete():
    model, rows, weights, data = _problem()

    result = uig.Explainer(
        model, rows, baseline_weights=weights
    )(data)

    normalized = weights / weights.sum()
    expected_base = normalized @ model.predict(rows)
    np.testing.assert_allclose(result.base_values, expected_base)
    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.predict(data),
        atol=1e-12,
    )


def test_background_object_uses_rows_and_weights():
    model, rows, weights, data = _problem()

    direct = uig.Explainer(
        model, rows, baseline_weights=weights
    )(data)
    from_background = uig.Explainer(
        model, Background(rows=rows, weights=weights)
    )(data)

    np.testing.assert_allclose(from_background.values, direct.values)
    np.testing.assert_allclose(
        from_background.base_values, direct.base_values
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0], "align"),
        ([1.0, -1.0, 1.0], "nonnegative"),
        ([0.0, 0.0, 0.0], "positive sum"),
        ([1.0, np.nan, 1.0], "finite"),
    ],
)
def test_invalid_baseline_weights_are_rejected(weights, message):
    model, rows, _, data = _problem()

    with pytest.raises(ValueError, match=message):
        uig.Explainer(model, rows, baseline_weights=weights)(data)


def test_explicit_weights_cannot_override_background_weights():
    model, rows, weights, data = _problem()

    with pytest.raises(ValueError, match="must be omitted"):
        uig.Explainer(
            model,
            Background(rows=rows, weights=weights),
            baseline_weights=weights,
        )(data)
