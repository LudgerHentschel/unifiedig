"""Public adapter for differentiable JAX prediction functions."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class JaxModel:
    """Describe a JAX prediction function for :class:`unifiedig.Explainer`.

    ``predict_fn`` is called as ``predict_fn(X, **call_kwargs)`` when
    ``params`` is omitted and as ``predict_fn(params, X, **call_kwargs)``
    otherwise. It must return one scalar or one output vector per sample.
    Set ``vectorize=True`` only when the function accepts a single sample.
    """

    predict_fn: Callable[..., Any]
    params: Any = None
    vectorize: bool = False
    call_kwargs: Optional[Mapping[str, Any]] = None
    output_names: Optional[Sequence[str]] = None
    dtype: Any = None

    def __post_init__(self) -> None:
        if not callable(self.predict_fn):
            raise TypeError("predict_fn must be callable")
        if not isinstance(self.vectorize, bool):
            raise TypeError("vectorize must be a boolean")
        if self.call_kwargs is not None:
            object.__setattr__(self, "call_kwargs", dict(self.call_kwargs))
        if self.output_names is not None:
            object.__setattr__(
                self, "output_names", [str(name) for name in self.output_names]
            )
