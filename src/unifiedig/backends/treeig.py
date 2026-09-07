"""Thin adapter to the required TreeIG package."""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .._loss import LossName, loss_values
from .base import BackendResult, classification_score_result


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
        """Recognize likely TreeIG models when the required package is absent."""
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
                "TreeIG is a required dependency. Repair the installation with "
                "`pip install --upgrade unifiedig`."
            ) from exc
        if not treeig.supports(model):
            raise TypeError("TreeIGBackend received an unsupported model")

        self.model = model
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) > 2:
            self._explainer = treeig.TreeIG(model, target=0)
            self._n_outputs = len(classes)
            self._class_names = [str(item) for item in classes]
        else:
            try:
                self._explainer = treeig.TreeIG(model)
            except ValueError as original_error:
                try:
                    candidate = treeig.TreeIG(model, target=0)
                except (TypeError, ValueError):
                    raise original_error
                arrays = getattr(candidate, "_arrays", {})
                if arrays.get("output_kind") != "multiclass_margin":
                    raise original_error
                self._explainer = candidate
                self._n_outputs = int(arrays["n_outputs"])
                self._class_names = [str(index) for index in range(self._n_outputs)]
            else:
                self._n_outputs = 1
                self._class_names = None

    def explain(
        self,
        data: NDArray[np.floating],
        baseline: NDArray[np.floating],
        baseline_weights: NDArray[np.floating],
    ) -> BackendResult:
        if self._n_outputs > 1:
            values = np.stack(
                [
                    self._explainer.attribute(
                        data,
                        baseline=baseline,
                        baseline_weights=baseline_weights,
                        target=target,
                    )
                    for target in range(self._n_outputs)
                ],
                axis=-1,
            )
            output_values = np.column_stack(
                [
                    self._explainer.model_output(data, target=target)
                    for target in range(self._n_outputs)
                ]
            )
            baseline_outputs = np.column_stack(
                [
                    self._explainer.model_output(baseline, target=target)
                    for target in range(self._n_outputs)
                ]
            )
            mean_base_value = baseline_weights @ baseline_outputs
            base_values = np.broadcast_to(
                mean_base_value, (data.shape[0], self._n_outputs)
            ).copy()
            assert self._class_names is not None
            return classification_score_result(
                values,
                base_values,
                output_values,
                self._class_names,
            )

        values = self._explainer.attribute(
            data,
            baseline=baseline,
            baseline_weights=baseline_weights,
        )

        output_values = self._explainer.model_output(data)
        mean_base_value = float(
            baseline_weights @ self._explainer.model_output(baseline)
        )
        base_values = np.full(data.shape[0], mean_base_value)
        output_names: Optional[Sequence[str]] = None
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            output_names = [str(classes[1])]
        return BackendResult(values, base_values, output_values, output_names)

    def explain_loss(
        self,
        data: NDArray[np.floating],
        baseline: NDArray[np.floating],
        baseline_weights: NDArray[np.floating],
        y: np.ndarray,
        loss: LossName,
    ) -> BackendResult:
        """Attribute loss changes with TreeIG's exact path-event kernels."""

        if self._n_outputs > 1:
            if loss != "log_loss":
                raise ValueError("squared_error requires exactly one model output")
            result = self._explainer.multiclass_loss_attribution(
                data,
                y,
                baseline=baseline,
                baseline_weights=baseline_weights,
                n_classes=self._n_outputs,
            )
            endpoint_scores = np.column_stack(
                [
                    self._explainer.model_output(data, target=target)
                    for target in range(self._n_outputs)
                ]
            )
            baseline_scores = np.stack(
                [
                    np.column_stack(
                        [
                            self._explainer.model_output(
                                baseline_row[None, :], target=target
                            )
                            for target in range(self._n_outputs)
                        ]
                    )[0]
                    for baseline_row in baseline
                ]
            )
        else:
            result = self._explainer.loss_attribution(
                data,
                y,
                baseline=baseline,
                baseline_weights=baseline_weights,
                loss=loss,
            )
            endpoint_scores = np.asarray(result["endpoint_prediction"])[:, None]
            baseline_scores = np.asarray(
                self._explainer.model_output(baseline)
            ).reshape(-1, 1)

        values = -np.asarray(result["observation_values"])
        output_values = loss_values(endpoint_scores, y, loss=loss)
        base_values = np.zeros(data.shape[0], dtype=float)
        for scores, weight in zip(baseline_scores, baseline_weights):
            repeated = np.broadcast_to(scores, (data.shape[0], len(scores)))
            base_values += weight * loss_values(repeated, y, loss=loss)
        return BackendResult(values, base_values, output_values, None)
