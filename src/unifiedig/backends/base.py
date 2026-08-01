"""Contract implemented by model-family-specific attribution backends."""

from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


BackendResult = Tuple[
    NDArray[np.floating], NDArray[np.floating], Optional[Sequence[str]]
]


class Backend(Protocol):
    """Internal protocol hidden behind :class:`unifiedig.Explainer`."""

    @classmethod
    def supports(cls, model: object) -> bool:
        """Return whether this backend can explain ``model``."""
        ...

    def __init__(self, model: object) -> None: ...

    def explain(
        self, data: NDArray[np.floating], baseline: NDArray[np.floating]
    ) -> BackendResult:
        """Return attribution values, baseline outputs, and output names."""
        ...

