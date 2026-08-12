"""Integrated Gradients for smooth models supported by skgrad."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import skgrad

from .base import BackendResult, classification_score_result


FloatArray = NDArray[np.floating]


class SkgradBackend:
    """Integrate analytic skgrad Jacobians, with an exact affine fast path."""

    @classmethod
    def supports(cls, model: object) -> bool:
        return skgrad.supports(model)

    def __init__(
        self, model: object, *, n_steps: int = 64, batch_size: int = 8192
    ) -> None:
        if not self.supports(model):
            raise TypeError("SkgradBackend received an unsupported model")
        if not hasattr(model, "n_features_in_"):
            raise ValueError("model must be fitted before creating an Explainer")

        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")

        # Validate fitted state and model configuration at construction.
        sample_output = skgrad.model_output(
            model, np.zeros((1, int(model.n_features_in_)))
        )

        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self._n_outputs = sample_output.shape[1]
        properties = skgrad.gradient_properties(model)
        self._constant_jacobian = properties.constant_jacobian
        exact_steps = getattr(properties, "exact_quadrature_steps", None)
        self.n_steps = min(n_steps, exact_steps) if exact_steps is not None else n_steps
        nodes, weights = np.polynomial.legendre.leggauss(self.n_steps)
        self._nodes = (nodes + 1.0) / 2.0
        self._weights = weights / 2.0

    def explain(
        self,
        data: FloatArray,
        baseline: FloatArray,
        baseline_weights: FloatArray,
    ) -> BackendResult:
        if data.shape[1] != self.model.n_features_in_:
            raise ValueError(f"data must have {self.model.n_features_in_} features")

        if self._constant_jacobian:
            mean_baseline = baseline_weights @ baseline
            difference = data - mean_baseline
            jacobian = skgrad.input_jacobian(
                self.model, mean_baseline[None, :]
            )[0]
            values = difference[:, :, None] * jacobian.T[None, :, :]
            mean_base_value = skgrad.model_output(
                self.model, mean_baseline[None, :]
            )[0]
        elif self._n_outputs == 1:
            values = self._integrated_scalar_distribution(
                data, baseline, baseline_weights
            )[:, :, None]
            mean_base_value = baseline_weights @ skgrad.model_output(
                self.model, baseline
            )
        else:
            values = None
            for baseline_row, baseline_weight in zip(baseline, baseline_weights):
                difference = data - baseline_row
                integrated_jacobian = self._integrated_jacobian(data, baseline_row)
                baseline_values = (
                    baseline_weight * difference[:, :, None] * integrated_jacobian
                )
                values = (
                    baseline_values if values is None else values + baseline_values
                )
            assert values is not None
            mean_base_value = baseline_weights @ skgrad.model_output(
                self.model, baseline
            )

        output_values = skgrad.model_output(self.model, data)
        base_values = np.broadcast_to(
            mean_base_value, (data.shape[0], mean_base_value.size)
        ).copy()
        output_names = self._output_names(output_values.shape[1])

        classes = getattr(self.model, "classes_", None)
        if classes is not None and output_values.shape[1] >= 2:
            return classification_score_result(
                values,
                base_values,
                output_values,
                [str(item) for item in classes],
            )

        if output_values.shape[1] == 1:
            return BackendResult(
                values[:, :, 0],
                base_values[:, 0],
                output_values[:, 0],
                output_names,
            )
        return BackendResult(values, base_values, output_values, output_names)

    def _integrated_scalar_distribution(
        self,
        data: FloatArray,
        baseline: FloatArray,
        baseline_weights: FloatArray,
    ) -> FloatArray:
        """Batch scalar reverse passes across baseline-observation paths."""

        n_samples, n_features = data.shape
        baselines_per_batch = max(1, self.batch_size // n_samples)
        values = np.zeros_like(data, dtype=np.result_type(data, baseline))
        for start in range(0, len(baseline), baselines_per_batch):
            stop = min(start + baselines_per_batch, len(baseline))
            baseline_rows = baseline[start:stop]
            weights = baseline_weights[start:stop]
            difference = data[None, :, :] - baseline_rows[:, None, :]
            integrated = np.zeros_like(difference)
            for node, weight in zip(self._nodes, self._weights):
                path = baseline_rows[:, None, :] + node * difference
                gradient = skgrad.input_gradient(
                    self.model, path.reshape(-1, n_features)
                ).reshape(len(baseline_rows), n_samples, n_features)
                integrated += weight * gradient
            values += np.sum(
                weights[:, None, None] * difference * integrated,
                axis=0,
            )
        return values

    def _integrated_jacobian(
        self, data: FloatArray, baseline_row: FloatArray
    ) -> FloatArray:
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
            if n_outputs == 1:
                return [str(classes[1])]
            return [str(item) for item in classes]
        return [str(index) for index in range(n_outputs)] if n_outputs > 1 else None
