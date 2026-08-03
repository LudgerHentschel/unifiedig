"""Opt-in Integrated Gradients using batched central finite differences."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.validation import check_is_fitted

from .base import BackendResult


FloatArray = NDArray[np.floating]


class FiniteDifferenceBackend:
    """Numerical IG fallback for smooth fitted sklearn estimators."""

    @classmethod
    def supports(cls, model: object) -> bool:
        if cls._is_known_nonsmooth(model):
            return False
        if is_classifier(model):
            return callable(getattr(model, "decision_function", None))
        return is_regressor(model) and callable(getattr(model, "predict", None))

    def __init__(
        self,
        model: object,
        *,
        n_steps: int = 64,
        relative_step: float = 1e-5,
        batch_size: int = 8192,
    ) -> None:
        if self._is_known_nonsmooth(model):
            raise TypeError(
                "finite differences are not appropriate for known tree or "
                "piecewise-constant estimators"
            )
        if is_classifier(model) and not callable(
            getattr(model, "decision_function", None)
        ):
            raise TypeError(
                "the finite-difference fallback requires decision_function for "
                "classifiers; probability outputs are not supported"
            )
        if not self.supports(model):
            raise TypeError(
                "the finite-difference fallback requires a fitted sklearn "
                "regressor with predict or binary classifier with decision_function"
            )

        check_is_fitted(model)
        if not hasattr(model, "n_features_in_"):
            raise ValueError("model must expose n_features_in_")
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) != 2:
            raise ValueError("V1 supports only binary classifiers")

        self.model = model
        self.n_steps = n_steps
        self.relative_step = relative_step
        self.batch_size = batch_size
        self._is_classifier = is_classifier(model)
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = (nodes + 1.0) / 2.0
        self._weights = weights / 2.0

    def explain(
        self,
        data: FloatArray,
        baseline: FloatArray,
        baseline_weights: FloatArray,
    ) -> BackendResult:
        n_samples, n_features = data.shape
        if n_features != self.model.n_features_in_:
            raise ValueError(f"data must have {self.model.n_features_in_} features")

        output_values = self._model_output(data)
        n_outputs = output_values.shape[1]
        values = np.zeros((n_samples, n_features, n_outputs), dtype=float)
        path_chunk_size = max(1, int(np.sqrt(self.batch_size)))

        for baseline_row, baseline_weight in zip(baseline, baseline_weights):
            difference = data - baseline_row
            integrated = np.zeros_like(values)
            n_path_points = self.n_steps * n_samples

            for start in range(0, n_path_points, path_chunk_size):
                flat_indices = np.arange(
                    start, min(start + path_chunk_size, n_path_points)
                )
                node_indices = flat_indices // n_samples
                sample_indices = flat_indices % n_samples
                path_data = baseline_row + self._nodes[node_indices, None] * difference[
                    sample_indices
                ]
                jacobian = self._finite_difference_jacobian(path_data, n_outputs)
                weighted = np.transpose(jacobian, (0, 2, 1)) * self._weights[
                    node_indices, None, None
                ]
                np.add.at(integrated, sample_indices, weighted)

            values += baseline_weight * difference[:, :, None] * integrated

        mean_base_value = baseline_weights @ self._model_output(baseline)
        base_values = np.broadcast_to(
            mean_base_value, (n_samples, n_outputs)
        ).copy()
        output_names = self._output_names(n_outputs)

        if n_outputs == 1:
            return BackendResult(
                values[:, :, 0],
                base_values[:, 0],
                output_values[:, 0],
                output_names,
            )
        return BackendResult(values, base_values, output_values, output_names)

    def _finite_difference_jacobian(
        self, data: FloatArray, n_outputs: int
    ) -> FloatArray:
        n_samples, n_features = data.shape
        jacobian = np.empty((n_samples, n_outputs, n_features), dtype=float)
        steps = self.relative_step * np.maximum(1.0, np.abs(data))
        feature_batch_size = max(1, self.batch_size // n_samples)

        for start in range(0, n_features, feature_batch_size):
            features = np.arange(start, min(start + feature_batch_size, n_features))
            n_selected = features.size
            perturbed = np.repeat(data, n_selected, axis=0)
            rows = np.arange(perturbed.shape[0])
            columns = np.tile(features, n_samples)
            deltas = steps[:, features].reshape(-1)

            perturbed[rows, columns] += deltas
            plus = self._model_output(perturbed).reshape(
                n_samples, n_selected, n_outputs
            )
            perturbed[rows, columns] -= 2.0 * deltas
            minus = self._model_output(perturbed).reshape(
                n_samples, n_selected, n_outputs
            )
            derivative = (plus - minus) / (2.0 * steps[:, features, None])
            jacobian[:, :, features] = np.transpose(derivative, (0, 2, 1))

        return jacobian

    def _model_output(self, data: FloatArray) -> FloatArray:
        if self._is_classifier:
            raw = self.model.decision_function(data)
        else:
            raw = self.model.predict(data)
        output = np.asarray(raw, dtype=float)
        if output.ndim == 1:
            output = output[:, None]
        if output.ndim != 2 or output.shape[0] != data.shape[0]:
            raise ValueError(
                "model output must have shape (samples,) or (samples, outputs)"
            )
        if self._is_classifier and output.shape[1] != 1:
            raise ValueError("V1 requires one binary decision score per sample")
        if not np.isfinite(output).all():
            raise ValueError("model output must contain only finite values")
        return output

    def _output_names(self, n_outputs: int) -> Optional[Sequence[str]]:
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            return [str(classes[1])]
        return [str(index) for index in range(n_outputs)] if n_outputs > 1 else None

    @staticmethod
    def _is_known_nonsmooth(model: object) -> bool:
        steps = getattr(model, "steps", None)
        candidate = steps[-1][1] if steps else model
        module = type(candidate).__module__
        name = type(candidate).__name__
        blocked_modules = (
            "sklearn.tree",
            "sklearn.ensemble._forest",
            "sklearn.ensemble._gb",
            "sklearn.ensemble._hist_gradient_boosting",
            "sklearn.ensemble._iforest",
            "sklearn.ensemble._weight_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
        )
        blocked_names = (
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "RadiusNeighborsClassifier",
            "RadiusNeighborsRegressor",
        )
        return module.startswith(blocked_modules) or name in blocked_names
