"""Public adapter for differentiable TensorFlow prediction functions."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TensorFlowModel:
    """Describe a TensorFlow prediction function for ``unifiedig.Explainer``.

    ``predict_fn`` must accept a batch as its first argument and return one
    scalar or one output vector per sample. ``call_kwargs`` are forwarded on
    every inference call. Direct TensorFlow-backed Keras models do not need
    this adapter.
    """

    predict_fn: Callable[..., Any]
    call_kwargs: Optional[Mapping[str, Any]] = None
    output_names: Optional[Sequence[str]] = None
    dtype: Any = None

    def __post_init__(self) -> None:
        if not callable(self.predict_fn):
            raise TypeError("predict_fn must be callable")
        if self.call_kwargs is not None:
            object.__setattr__(self, "call_kwargs", dict(self.call_kwargs))
        if self.output_names is not None:
            object.__setattr__(
                self, "output_names", [str(name) for name in self.output_names]
            )
