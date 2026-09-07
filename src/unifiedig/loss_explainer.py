"""Loss attribution through UnifiedIG's existing backend machinery."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ._loss import LossName, validate_targets
from .backends.base import BackendResult
from .baselines import (
    normalize_jax_inputs,
    normalize_tensorflow_inputs,
    normalize_torch_inputs,
)
from .explainer import _AUTOMATIC_N_STEPS, Explainer
from .explanation import Explanation


class LossExplainer:
    """Attribute squared-error or log-loss changes with Integrated Gradients.

    Model selection, baselines, numerical controls, and output semantics match
    :class:`unifiedig.Explainer`. Classification consumes raw decision scores,
    margins, or logits; probability outputs are not attributed.

    Values normally sum to endpoint loss minus baseline loss, so negative
    values reduce loss. ``direction="loss_reduction"`` negates only the final
    attribution values. Base values and diagnostics retain the default
    loss-change direction.

    Parameters
    ----------
    model : object
        Fitted model accepted by :class:`unifiedig.Explainer` and a loss-aware
        backend.
    baseline : array-like or background object
        One baseline sample or a shared weighted baseline distribution.
    baseline_weights : array-like, optional
        Nonnegative weights aligned with baseline rows.
    loss : {"squared_error", "log_loss"}, default="squared_error"
        Squared error for scalar regression outputs, or binary/multiclass log
        loss for raw classification scores.
    direction : {"loss_change", "loss_reduction"}, default="loss_change"
        Orientation of returned attribution values. The alternative reverses
        values only.
    attribute_after : str, optional
        Attribute after a named pipeline preprocessing step. Supply original
        data and baselines; both are transformed together. None keeps inputs.
    n_steps : int, optional
        Gauss--Legendre nodes for numerical gradient backends. When omitted,
        loss attribution starts at 16 and may refine to 32 or 64.

    Notes
    -----
    The remaining numerical and output arguments have the same meanings as on
    :class:`unifiedig.Explainer`. Calling ``explainer(data, y)`` returns an
    :class:`unifiedig.Explanation` with one scalar loss attribution per input
    feature and observation.
    """

    def __init__(
        self,
        model: object,
        baseline: Any,
        *,
        baseline_weights: Any | None = None,
        attribute_after: str | None = None,
        loss: LossName = "squared_error",
        direction: Literal["loss_change", "loss_reduction"] = "loss_change",
        n_steps: int | None = None,
        check_completeness: bool = True,
        on_incomplete: Literal["warn", "raise"] = "warn",
        completeness_atol: float = 1e-6,
        completeness_rtol: float = 1e-4,
        fallback: str | None = None,
        finite_difference_step: float = 1e-5,
        finite_difference_batch_size: int = 8192,
        gradient_batch_size: int = 8192,
        probability_floor: float | None = None,
        output_kind: Literal["auto", "regression", "classification"] = "auto",
    ) -> None:
        if loss not in ("squared_error", "log_loss"):
            raise ValueError("loss must be 'squared_error' or 'log_loss'")
        if direction not in ("loss_change", "loss_reduction"):
            raise ValueError("direction must be 'loss_change' or 'loss_reduction'")
        self.loss = loss
        self.direction = direction
        self._explainer = Explainer(
            model,
            baseline,
            baseline_weights=baseline_weights,
            attribute_after=attribute_after,
            n_steps=n_steps,
            check_completeness=check_completeness,
            on_incomplete=on_incomplete,
            completeness_atol=completeness_atol,
            completeness_rtol=completeness_rtol,
            fallback=fallback,
            finite_difference_step=finite_difference_step,
            finite_difference_batch_size=finite_difference_batch_size,
            gradient_batch_size=gradient_batch_size,
            probability_floor=probability_floor,
            output_kind=output_kind,
        )
        if not callable(getattr(self._explainer._backend, "explain_loss", None)):
            raise TypeError(
                f"no Unified IG loss backend supports {type(model).__name__}"
            )
        configure = getattr(self._explainer._backend, "configure_loss_quadrature", None)
        if callable(configure):
            configure(loss, automatic=self._explainer._automatic_steps)

    @property
    def n_steps(self) -> int:
        """Current numerical integration resolution."""

        return int(
            getattr(
                self._explainer._backend,
                "loss_n_steps",
                self._explainer.n_steps,
            )
        )

    def __call__(self, data: Any, y: Any) -> Explanation:
        """Return observation-level feature attributions for realized loss."""

        feature_names = self._explainer._attribution_feature_names(data)
        normalized_data, baseline, weights, explanation_data = self._normalize(data)
        targets = np.asarray(y).reshape(-1)
        if len(targets) != len(explanation_data):
            raise ValueError("y and data must have the same number of observations")
        targets = validate_targets(
            self._explainer.model,
            targets,
            loss=self.loss,
            n_outputs=self._n_outputs(normalized_data),
        )
        result = self._explain(normalized_data, baseline, weights, targets)
        error = self._explainer._completeness_error(result)
        if self._explainer.check_completeness and self._explainer._automatic_steps:
            candidates = (
                (16, *_AUTOMATIC_N_STEPS) if self.n_steps < 16 else _AUTOMATIC_N_STEPS
            )
            for candidate in candidates:
                if not self._explainer._completeness_failed(
                    error, result.output_values
                ):
                    break
                factory = self._explainer._backend_factory
                if factory is None or candidate <= self._explainer.n_steps:
                    continue
                self._explainer._backend = factory(candidate)
                self._explainer.n_steps = candidate
                result = self._explain(normalized_data, baseline, weights, targets)
                error = self._explainer._completeness_error(result)
        if self._explainer.check_completeness and self._explainer._completeness_failed(
            error, result.output_values
        ):
            message = (
                "Loss Integrated Gradients completeness tolerance was not met; "
                f"maximum absolute error is {np.max(np.abs(error)):.3g}. "
                "Increase n_steps for numerical gradient integration."
            )
            if self._explainer.on_incomplete == "raise":
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        values = -result.values if self.direction == "loss_reduction" else result.values
        return Explanation(
            values=values,
            base_values=result.base_values,
            data=explanation_data,
            feature_names=feature_names,
            completeness_error=error,
            attribute_after=self._explainer.attribute_after,
        )

    def _explain(
        self, data: Any, baseline: Any, weights: Any, y: np.ndarray
    ) -> BackendResult:
        return self._explainer._backend.explain_loss(
            data, baseline, weights, y, self.loss
        )

    def _n_outputs(self, data: Any) -> int:
        backend = self._explainer._backend
        if hasattr(backend, "_n_outputs"):
            return int(backend._n_outputs)
        if hasattr(backend, "_model_output"):
            return int(backend._model_output(data).shape[1])
        output = backend._forward(data)
        return 1 if output.ndim == 1 else int(output.shape[1])

    def _normalize(self, data: Any) -> tuple[Any, Any, Any, np.ndarray]:
        backend = self._explainer._backend
        kind = getattr(backend, "input_kind", "numpy")
        if kind == "torch":
            normalized = normalize_torch_inputs(
                data,
                self._explainer.baseline,
                self._explainer.baseline_weights,
                device=backend.device,
                dtype=backend.dtype,
            )
            explanation_data = normalized[0].detach().cpu().numpy()
        elif kind == "jax":
            normalized = normalize_jax_inputs(
                data,
                self._explainer.baseline,
                self._explainer.baseline_weights,
                dtype=backend.dtype,
            )
            explanation_data = np.asarray(normalized[0])
        elif kind == "tensorflow":
            normalized = normalize_tensorflow_inputs(
                data,
                self._explainer.baseline,
                self._explainer.baseline_weights,
                dtype=backend.dtype,
            )
            explanation_data = normalized[0].numpy()
        else:
            normalized = self._explainer._normalize_numpy_inputs(data)
            explanation_data = normalized[0]
        return *normalized, explanation_data
