"""The model-agnostic Unified IG entry point."""

import warnings
from numbers import Real
from typing import Any, Optional, Sequence, Type

import numpy as np

from .backends import (
    Backend,
    PyTorchBackend,
    SkgradBackend,
    TreeIGBackend,
)
from .baselines import normalize_inputs, normalize_torch_inputs
from .explanation import Explanation


_BACKENDS: Sequence[Type[Backend]] = (
    SkgradBackend,
    TreeIGBackend,
    PyTorchBackend,
)


class Explainer:
    """Explain a model with Integrated Gradients through one stable interface.

    ``baseline`` is either one reference sample or an equally weighted shared
    baseline distribution. Every input is attributed against every baseline;
    matching input and baseline row counts do not imply pairing. ``n_steps``
    controls numerical backends and is ignored by exact backends.

    Binary classifiers are explained on their decision-score (logit) scale in
    V1. Probability attributions are intentionally not offered.
    """

    def __init__(
        self,
        model: object,
        baseline: Any,
        *,
        n_steps: int = 64,
        check_completeness: bool = True,
        completeness_atol: float = 1e-6,
        completeness_rtol: float = 1e-4,
    ) -> None:
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(check_completeness, bool):
            raise ValueError("check_completeness must be a boolean")
        self.model = model
        self.baseline = baseline
        self.n_steps = n_steps
        self.check_completeness = check_completeness
        self.completeness_atol = self._validate_tolerance(
            "completeness_atol", completeness_atol
        )
        self.completeness_rtol = self._validate_tolerance(
            "completeness_rtol", completeness_rtol
        )
        backend_type = next((item for item in _BACKENDS if item.supports(model)), None)
        if backend_type is None:
            raise TypeError(f"no Unified IG backend supports {type(model).__name__}")
        self._backend = backend_type(model, n_steps=n_steps)

    def __call__(self, data: Any) -> Explanation:
        feature_names = self._feature_names(data)
        if getattr(self._backend, "input_kind", "numpy") == "torch":
            normalized_data, normalized_baseline = normalize_torch_inputs(
                data,
                self.baseline,
                device=self._backend.device,
                dtype=self._backend.dtype,
            )
            explanation_data = normalized_data.detach().cpu().numpy()
        else:
            normalized_data, normalized_baseline = normalize_inputs(data, self.baseline)
            explanation_data = normalized_data
        backend_result = self._backend.explain(normalized_data, normalized_baseline)
        if backend_result.output_values.ndim == 1:
            attribution_axes = tuple(range(1, backend_result.values.ndim))
        else:
            attribution_axes = tuple(range(1, backend_result.values.ndim - 1))
        attributed_difference = backend_result.values.sum(axis=attribution_axes)
        completeness_error = backend_result.output_values - (
            attributed_difference + backend_result.base_values
        )
        if self.check_completeness and not np.allclose(
            completeness_error,
            0.0,
            atol=self.completeness_atol,
            rtol=0.0,
        ):
            allowed_error = self.completeness_atol + self.completeness_rtol * np.abs(
                backend_result.output_values
            )
            if np.any(np.abs(completeness_error) > allowed_error):
                warnings.warn(
                    "Integrated Gradients completeness tolerance was not met; "
                    f"maximum absolute error is {np.max(np.abs(completeness_error)):.3g}. "
                    "Increase n_steps for numerical backends.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return Explanation(
            values=backend_result.values,
            base_values=backend_result.base_values,
            data=explanation_data,
            feature_names=feature_names,
            output_names=backend_result.output_names,
            completeness_error=completeness_error,
        )

    @staticmethod
    def _feature_names(data: Any) -> Optional[Sequence[str]]:
        columns = getattr(data, "columns", None)
        return [str(column) for column in columns] if columns is not None else None

    @staticmethod
    def _validate_tolerance(name: str, value: float) -> float:
        if not isinstance(value, Real) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
        return float(value)
