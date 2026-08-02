"""A small, plotting-library-independent explanation container."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Explanation:
    """Parallel arrays describing feature attributions.

    The field names intentionally mirror the useful subset of
    :class:`shap.Explanation`. Arrays use a leading sample dimension.
    """

    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    data: NDArray[Any]
    feature_names: Optional[Sequence[str]] = None
    output_names: Optional[Sequence[str]] = None
    completeness_error: Optional[NDArray[np.floating]] = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        data = np.asarray(self.data)
        base_values = np.asarray(self.base_values)
        if data.ndim != 2:
            raise ValueError("data must have shape (samples, features)")
        if values.ndim not in (2, 3) or values.shape[:2] != data.shape:
            raise ValueError(
                "values must have shape (samples, features) or "
                "(samples, features, outputs)"
            )
        expected_base_shape = (
            (data.shape[0],)
            if values.ndim == 2
            else (data.shape[0], values.shape[2])
        )
        if base_values.shape != expected_base_shape:
            raise ValueError(f"base_values must have shape {expected_base_shape}")
        if self.completeness_error is not None:
            completeness_error = np.asarray(self.completeness_error)
            if completeness_error.shape != base_values.shape:
                raise ValueError(
                    "completeness_error must have the same shape as base_values"
                )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def max_abs_completeness_error(self) -> Optional[float]:
        """Largest absolute completeness residual, or ``None`` if unavailable."""

        if self.completeness_error is None:
            return None
        return float(np.max(np.abs(self.completeness_error)))

    def to_shap(self) -> Any:
        """Return an equivalent ``shap.Explanation`` when SHAP is installed."""

        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is optional. Install it with `pip install unifiedig[shap]`."
            ) from exc
        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
            output_names=self.output_names,
        )
