"""Explicit numerical path-event fallback powered by TreeIGNumeric."""

import warnings
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base import BackendResult, classification_score_result


class TreeIGNumericBackend:
    """Approximate IG for known piecewise-constant model families."""

    @classmethod
    def supports(cls, model: object) -> bool:
        model_type = type(model)
        if model_type.__module__.startswith("catboost"):
            return model_type.__name__ in {
                "CatBoost",
                "CatBoostClassifier",
                "CatBoostRegressor",
            }
        try:
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
                RandomForestClassifier,
            )
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:  # pragma: no cover - sklearn is a core dependency
            return False
        return isinstance(
            model,
            (
                DecisionTreeClassifier,
                ExtraTreesClassifier,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
                RandomForestClassifier,
            ),
        )

    def __init__(
        self,
        model: object,
        *,
        n_steps: int = 1024,
        probability_floor: Optional[float] = None,
    ) -> None:
        try:
            import treeig
        except ImportError as exc:
            raise ImportError(
                "Numerical tree support requires TreeIG 0.1.10 or newer. "
                "Install it with `pip install unifiedig[trees]`."
            ) from exc
        if not self.supports(model):
            raise TypeError(
                "the numerical tree fallback supports known piecewise-constant "
                "tree families only"
            )
        classes = getattr(model, "classes_", None)
        self._classes = None if classes is None else list(classes)
        self._probability_to_score = bool(
            self._classes and not self._has_raw_score_output(model)
        )
        if self._probability_to_score:
            warnings.warn(
                f"{type(model).__name__} has no native decision score; deriving "
                "binary log odds or centered multiclass log scores from its "
                "class probabilities.",
                RuntimeWarning,
                stacklevel=3,
            )
        self.model = model
        self.grid_size = n_steps
        self.probability_floor = probability_floor
        self._treeig = treeig

    def explain(
        self,
        data: NDArray[np.floating],
        baseline: NDArray[np.floating],
        baseline_weights: NDArray[np.floating],
    ) -> BackendResult:
        if self._classes is not None and len(self._classes) > 2:
            n_outputs = len(self._classes)
            results = [
                self._attribute_target(
                    data, baseline, baseline_weights, target=target
                )
                for target in range(n_outputs)
            ]
            values = np.stack([result[0] for result in results], axis=-1)
            base_values = np.column_stack([result[1] for result in results])
            output_values = np.column_stack([result[2] for result in results])
            return classification_score_result(
                values,
                base_values,
                output_values,
                [str(item) for item in self._classes],
            )

        values, base_values, output_values = self._attribute_target(
            data, baseline, baseline_weights, target=None
        )
        output_names: Optional[Sequence[str]] = None
        if self._classes:
            output_names = [str(self._classes[1])]
        return BackendResult(values, base_values, output_values, output_names)

    def _attribute_target(
        self,
        data: NDArray[np.floating],
        baseline: NDArray[np.floating],
        baseline_weights: NDArray[np.floating],
        *,
        target: Optional[int],
    ):
        values = np.zeros_like(data, dtype=float)
        mean_base = 0.0
        output_values = None
        for baseline_row, weight in zip(baseline, baseline_weights):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="No native decision score is available.*",
                    category=RuntimeWarning,
                )
                explainer = self._treeig.TreeIGNumeric(
                    self.model,
                    baseline=baseline_row,
                    target=target,
                    probability_to_score=self._probability_to_score,
                    probability_floor=(
                        self.probability_floor
                        if self._probability_to_score
                        else None
                    ),
                    grid_size=self.grid_size,
                    warn_residual=False,
                )
            values += float(weight) * explainer.attribute(data)
            mean_base += float(weight) * explainer.model_output(
                baseline_row[None, :]
            )[0]
            if output_values is None:
                output_values = explainer.model_output(data)
        assert output_values is not None
        return (
            values,
            np.full(data.shape[0], mean_base),
            output_values,
        )

    @staticmethod
    def _has_raw_score_output(model: object) -> bool:
        if hasattr(model, "decision_function"):
            return True
        return type(model).__module__.startswith("catboost")
