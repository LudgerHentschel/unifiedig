"""Integrated Gradients for smooth models supported by skgrad."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import skgrad

from .base import BackendResult


FloatArray = NDArray[np.floating]


class SkgradBackend:
    """Integrate analytic skgrad Jacobians, with an exact affine fast path."""

    @classmethod
    def supports(cls, model: object) -> bool:
        return skgrad.supports(model)

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        if not self.supports(model):
            raise TypeError("SkgradBackend received an unsupported model")
        if not hasattr(model, "n_features_in_"):
            raise ValueError("model must be fitted before creating an Explainer")

        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) != 2:
            raise ValueError("V1 supports only binary classifiers")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")

        # Validate fitted state and model configuration at construction.
        skgrad.model_output(model, np.zeros((1, int(model.n_features_in_))))

        self.model = model
        self.n_steps = n_steps
        self._constant_jacobian = skgrad.gradient_properties(
            model
        ).constant_jacobian
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = (nodes + 1.0) / 2.0
        self._weights = weights / 2.0

    def explain(self, data: FloatArray, baseline: FloatArray) -> BackendResult:
        if data.shape[1] != self.model.n_features_in_:
            raise ValueError(f"data must have {self.model.n_features_in_} features")

        values: Optional[FloatArray] = None
        for baseline_row in baseline:
            difference = data - baseline_row
            integrated_jacobian = self._integrated_jacobian(data, baseline_row)
            baseline_values = difference[:, :, None] * integrated_jacobian
            values = baseline_values if values is None else values + baseline_values

        assert values is not None
        values /= baseline.shape[0]
        output_values = skgrad.model_output(self.model, data)
        mean_base_value = skgrad.model_output(self.model, baseline).mean(axis=0)
        base_values = np.broadcast_to(
            mean_base_value, (data.shape[0], mean_base_value.size)
        ).copy()
        output_names = self._output_names(output_values.shape[1])

        if output_values.shape[1] == 1:
            return BackendResult(
                values[:, :, 0],
                base_values[:, 0],
                output_values[:, 0],
                output_names,
            )
        return BackendResult(values, base_values, output_values, output_names)

    def _integrated_jacobian(
        self, data: FloatArray, baseline_row: FloatArray
    ) -> FloatArray:
        if self._constant_jacobian:
            jacobian = skgrad.input_jacobian(self.model, baseline_row[None, :])[0]
            return np.broadcast_to(
                jacobian.T, (data.shape[0], data.shape[1], jacobian.shape[0])
            )

        difference = data - baseline_row
        integrated: Optional[FloatArray] = None
        for node, weight in zip(self._nodes, self._weights):
            path_data = baseline_row + node * difference
            jacobian = np.transpose(
                skgrad.input_jacobian(self.model, path_data), (0, 2, 1)
            )
            if integrated is None:
                integrated = weight * jacobian
            else:
                integrated += weight * jacobian

        assert integrated is not None
        return integrated

    def _output_names(self, n_outputs: int) -> Optional[Sequence[str]]:
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            return [str(classes[1])]
        return [str(index) for index in range(n_outputs)] if n_outputs > 1 else None
