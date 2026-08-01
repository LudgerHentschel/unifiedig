import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import unifiedig as uig


def test_binary_logistic_regression_is_complete_on_logit_scale():
    training_data = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    model = LogisticRegression().fit(training_data, np.array([0, 0, 1, 1]))
    data = np.array([[-0.5, 2.0], [1.5, -0.5]])
    baseline = np.array([0.25, -0.25])

    result = uig.Explainer(model, baseline)(data)

    np.testing.assert_allclose(
        result.values.sum(axis=1) + result.base_values,
        model.decision_function(data),
    )
    assert result.output_names == [str(model.classes_[1])]


def test_multiclass_logistic_regression_is_rejected():
    model = LogisticRegression().fit(
        np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1, 2, 2]),
    )
    with pytest.raises(ValueError, match="binary"):
        uig.Explainer(model, np.array([0.0]))

