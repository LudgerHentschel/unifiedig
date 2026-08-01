"""Closed-form Integrated Gradients for sklearn linear estimators."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, LogisticRegression

from .base import BackendResult


class SklearnLinearBackend:
    """Exact IG for models whose explained output is affine in the inputs."""

    @classmethod
    def supports(cls, model: object) -> bool:
        return isinstance(model, (LinearRegression, LogisticRegression))

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        if not self.supports(model):
            raise TypeError("SklearnLinearBackend received an unsupported model")
        if not hasattr(model, "coef_"):
            raise ValueError("model must be fitted before creating an Explainer")
        if isinstance(model, LogisticRegression) and len(model.classes_) != 2:
            raise ValueError("V1 supports only binary LogisticRegression")
        self.model = model

    def explain(
        self, data: NDArray[np.floating], baseline: NDArray[np.floating]
    ) -> BackendResult:
        coefficients = np.asarray(self.model.coef_, dtype=float)
        intercept = np.asarray(self.model.intercept_, dtype=float)

        if isinstance(self.model, LogisticRegression):
            weights = coefficients.reshape(-1)
            values = (data - baseline) * weights
            base_values = baseline @ weights + intercept.reshape(-1)[0]
            return values, base_values, [str(self.model.classes_[1])]

        if coefficients.ndim == 1:
            values = (data - baseline) * coefficients
            base_values = baseline @ coefficients + intercept.reshape(-1)[0]
            return values, base_values, None

        # Multi-output regression: (samples, features, outputs).
        values = (data - baseline)[:, :, None] * coefficients.T[None, :, :]
        base_values = baseline @ coefficients.T + intercept
        output_names: Optional[Sequence[str]] = [str(i) for i in range(coefficients.shape[0])]
        return values, base_values, output_names
