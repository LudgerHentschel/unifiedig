"""Integrated Gradients for fitted scikit-learn multilayer perceptrons."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import skgrad
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import BackendResult


FloatArray = NDArray[np.floating]


class SklearnMLPBackend:
    """Analytic MLP gradients integrated with Gauss–Legendre quadrature."""

    @classmethod
    def supports(cls, model: object) -> bool:
        return isinstance(model, (MLPRegressor, MLPClassifier))

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        if not self.supports(model):
            raise TypeError("SklearnMLPBackend received an unsupported model")
        if not hasattr(model, "n_features_in_"):
            raise ValueError("model must be fitted before creating an Explainer")
        if isinstance(model, MLPClassifier) and len(model.classes_) != 2:
            raise ValueError("V1 supports only binary MLPClassifier")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")

        # Validate fitted state and the model configuration at construction.
        skgrad.model_output(model, np.zeros((1, int(model.n_features_in_))))

        self.model = model
        self.n_steps = n_steps
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = (nodes + 1.0) / 2.0
        self._weights = weights / 2.0

    def explain(self, data: FloatArray, baseline: FloatArray) -> BackendResult:
        if data.shape[1] != self.model.n_features_in_:
            raise ValueError(f"data must have {self.model.n_features_in_} features")

        values: Optional[FloatArray] = None
        for baseline_row in baseline:
            difference = data - baseline_row
            integrated_gradient: Optional[FloatArray] = None
            for node, weight in zip(self._nodes, self._weights):
                path_data = baseline_row + node * difference
                gradient = np.transpose(
                    skgrad.input_jacobian(self.model, path_data), (0, 2, 1)
                )
                if integrated_gradient is None:
                    integrated_gradient = weight * gradient
                else:
                    integrated_gradient += weight * gradient

            assert integrated_gradient is not None
            if integrated_gradient.shape[-1] == 1:
                baseline_values = difference * integrated_gradient[..., 0]
            else:
                baseline_values = difference[:, :, None] * integrated_gradient
            if values is None:
                values = baseline_values
            else:
                values += baseline_values

        assert values is not None
        values /= baseline.shape[0]
        mean_base_value = skgrad.model_output(self.model, baseline).mean(axis=0)
        base_values = np.broadcast_to(
            mean_base_value, (data.shape[0], mean_base_value.size)
        ).copy()
        output_values = skgrad.model_output(self.model, data)
        if output_values.shape[-1] == 1:
            return BackendResult(
                values, base_values[:, 0], output_values[:, 0], self._output_names()
            )

        return BackendResult(values, base_values, output_values, self._output_names())

    def _output_names(self) -> Optional[Sequence[str]]:
        if isinstance(self.model, MLPClassifier):
            return [str(self.model.classes_[1])]
        n_outputs = int(self.model.n_outputs_)
        return [str(index) for index in range(n_outputs)] if n_outputs > 1 else None
