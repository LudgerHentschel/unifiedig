"""Thin adapter to the optional TreeIG package."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base import BackendResult


class TreeIGBackend:
    """Exact IG for tree models supported by TreeIG."""

    @classmethod
    def supports(cls, model: object) -> bool:
        try:
            import treeig
        except ImportError:
            return cls._looks_like_supported_model(model)
        return bool(treeig.supports(model))

    @staticmethod
    def _looks_like_supported_model(model: object) -> bool:
        """Recognize likely TreeIG models when the optional package is absent."""
        try:
            from sklearn.ensemble import (
                ExtraTreesRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestRegressor,
            )
            from sklearn.tree import DecisionTreeRegressor
        except ImportError:  # pragma: no cover - sklearn is a core dependency
            sklearn_types = ()
        else:
            sklearn_types = (
                DecisionTreeRegressor,
                ExtraTreesRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestRegressor,
            )
        if isinstance(model, sklearn_types):
            return True

        model_type = type(model)
        optional_types = {
            ("xgboost.core", "Booster"),
            ("xgboost.sklearn", "XGBClassifier"),
            ("xgboost.sklearn", "XGBRegressor"),
            ("lightgbm.basic", "Booster"),
            ("lightgbm.sklearn", "LGBMClassifier"),
            ("lightgbm.sklearn", "LGBMRegressor"),
        }
        return (model_type.__module__, model_type.__name__) in optional_types

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        try:
            import treeig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Tree support is optional. Install it with "
                "`pip install unifiedig[trees]`."
            ) from exc
        if not treeig.supports(model):
            raise TypeError("TreeIGBackend received an unsupported model")

        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) != 2:
            raise ValueError("V1 supports only binary tree classifiers")
        self.model = model
        self._explainer = treeig.TreeIG(model)

    def explain(
        self,
        data: NDArray[np.floating],
        baseline: NDArray[np.floating],
    ) -> BackendResult:
        values = self._explainer.attribute(data, baseline=baseline)

        output_values = self._explainer.model_output(data)
        mean_base_value = self._explainer.model_output(baseline).mean()
        base_values = np.full(data.shape[0], mean_base_value)
        output_names: Optional[Sequence[str]] = None
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            output_names = [str(classes[1])]
        return BackendResult(values, base_values, output_values, output_names)
