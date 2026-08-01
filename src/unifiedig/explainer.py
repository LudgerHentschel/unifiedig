"""The model-agnostic Unified IG entry point."""

from typing import Any, Optional, Sequence, Type

from .backends import Backend, SklearnLinearBackend
from .baselines import normalize_inputs
from .explanation import Explanation


_BACKENDS: Sequence[Type[Backend]] = (SklearnLinearBackend,)


class Explainer:
    """Select an IG backend and expose a consistent callable interface.

    Binary classifiers are explained on their decision-score (logit) scale in
    V1. Probability attributions are intentionally not offered.
    """

    def __init__(self, model: object, baseline: Any) -> None:
        self.model = model
        self.baseline = baseline
        backend_type = next((item for item in _BACKENDS if item.supports(model)), None)
        if backend_type is None:
            raise TypeError(f"no Unified IG backend supports {type(model).__name__}")
        self._backend = backend_type(model)

    def __call__(self, data: Any) -> Explanation:
        feature_names = self._feature_names(data)
        normalized_data, normalized_baseline = normalize_inputs(data, self.baseline)
        values, base_values, output_names = self._backend.explain(
            normalized_data, normalized_baseline
        )
        return Explanation(
            values=values,
            base_values=base_values,
            data=normalized_data,
            feature_names=feature_names,
            output_names=output_names,
        )

    @staticmethod
    def _feature_names(data: Any) -> Optional[Sequence[str]]:
        columns = getattr(data, "columns", None)
        return [str(column) for column in columns] if columns is not None else None

