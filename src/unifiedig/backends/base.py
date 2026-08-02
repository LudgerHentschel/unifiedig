"""Contract implemented by model-family-specific attribution backends."""

from typing import NamedTuple, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class BackendResult(NamedTuple):
    """Internal backend result, including the output needed for diagnostics."""

    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    output_values: NDArray[np.floating]
    output_names: Optional[Sequence[str]]


class Backend(Protocol):
    """Internal protocol hidden behind :class:`unifiedig.Explainer`."""

    @classmethod
    def supports(cls, model: object) -> bool:
        """Return whether this backend can explain ``model``."""
        ...

    def __init__(self, model: object, *, n_steps: int = 64) -> None: ...

    def explain(
        self, data: NDArray[np.floating], baseline: NDArray[np.floating]
    ) -> BackendResult:
        """Return attribution values, baseline outputs, and output names."""
        ...
