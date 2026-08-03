"""Contract implemented by model-family-specific attribution backends."""

from typing import Any, NamedTuple, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class BackendResult(NamedTuple):
    """Internal backend result, including the output needed for diagnostics."""

    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    output_values: NDArray[np.floating]
    output_names: Optional[Sequence[str]]


def classification_score_result(
    values: NDArray[np.floating],
    base_values: NDArray[np.floating],
    output_values: NDArray[np.floating],
    class_names: Sequence[str],
) -> BackendResult:
    """Represent a vector of raw class scores in its nonredundant space.

    Two-class score vectors become the single margin ``score[1] - score[0]``.
    For three or more classes, scores and their attributions are centered
    across classes. The stored ``K`` labeled coordinates therefore span the
    classifier's ``K - 1`` dimensional decision-score space.
    """

    values = np.asarray(values)
    base_values = np.asarray(base_values)
    output_values = np.asarray(output_values)
    names = [str(name) for name in class_names]
    n_outputs = output_values.shape[-1]
    if n_outputs < 2:
        raise ValueError("classification score vectors require at least two outputs")
    if len(names) != n_outputs:
        raise ValueError("class names must align with classification scores")
    if values.shape[-1] != n_outputs or base_values.shape[-1] != n_outputs:
        raise ValueError("classification score arrays must share an output axis")

    if n_outputs == 2:
        return BackendResult(
            values[..., 1] - values[..., 0],
            base_values[..., 1] - base_values[..., 0],
            output_values[..., 1] - output_values[..., 0],
            [names[1]],
        )

    return BackendResult(
        values - values.mean(axis=-1, keepdims=True),
        base_values - base_values.mean(axis=-1, keepdims=True),
        output_values - output_values.mean(axis=-1, keepdims=True),
        names,
    )


class Backend(Protocol):
    """Internal protocol hidden behind :class:`unifiedig.Explainer`."""

    @classmethod
    def supports(cls, model: object) -> bool:
        """Return whether this backend can explain ``model``."""
        ...

    def __init__(self, model: object, *, n_steps: int = 64) -> None: ...

    def explain(
        self, data: Any, baseline: Any, baseline_weights: Any
    ) -> BackendResult:
        """Return attribution values, baseline outputs, and output names."""
        ...
