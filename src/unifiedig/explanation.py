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

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        data = np.asarray(self.data)
        base_values = np.asarray(self.base_values)
        if values.shape[0] != data.shape[0]:
            raise ValueError("values and data must contain the same number of samples")
        if base_values.ndim == 0 or base_values.shape[0] != data.shape[0]:
            raise ValueError("base_values must contain one entry per sample")

    def __len__(self) -> int:
        return len(self.data)

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

