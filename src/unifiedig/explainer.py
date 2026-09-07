"""The model-agnostic Unified IG entry point."""

import warnings
from numbers import Real
from typing import Any, Callable, Literal, Optional, Sequence, Type

import numpy as np

from .backends import (
    Backend,
    FiniteDifferenceBackend,
    JaxBackend,
    PyTorchBackend,
    SkgradBackend,
    TensorFlowBackend,
    TreeIGBackend,
    TreeIGNumericBackend,
)
from .baselines import (
    _baseline_parts,
    normalize_baseline,
    normalize_inputs,
    normalize_jax_inputs,
    normalize_tensorflow_inputs,
    normalize_torch_inputs,
)
from .explanation import Explanation

_BACKENDS: Sequence[Type[Backend]] = (
    SkgradBackend,
    PyTorchBackend,
    JaxBackend,
    TensorFlowBackend,
    TreeIGBackend,
)
_DEFAULT_N_STEPS = 16
_AUTOMATIC_N_STEPS = (32, 64)


class Explainer:
    """Explain a model with Integrated Gradients through one stable interface.

    ``baseline`` is either one reference sample, a shared baseline matrix, or
    a background object exposing aligned ``rows`` and ``weights`` properties.
    Every input is attributed against every baseline; matching input and
    baseline row counts do not imply pairing. ``n_steps`` controls numerical
    backends and is ignored by exact backends. When omitted, gradient backends
    start at 16 nodes and retry at 32 or 64 only if completeness fails. An
    explicit integer disables this automatic refinement.

    Classifiers are explained on their decision-score scale. Binary outputs use
    one margin; multiclass outputs use the complete centered score vector.
    Probability attributions are intentionally not offered. Set
    ``fallback="finite_difference"`` to explain an otherwise unsupported smooth
    sklearn estimator numerically, or ``fallback="tree_numeric"`` to use
    approximate path-event detection for a recognized piecewise-constant tree
    model. Probability-only tree classifiers are transformed to log scores;
    ``probability_floor`` must be set explicitly if any path probability is
    zero. Specialized backends always take precedence.
    Differentiable JAX functions are selected explicitly by wrapping them in
    :class:`unifiedig.JaxModel`; arbitrary TensorFlow functions use
    :class:`unifiedig.TensorFlowModel`. TensorFlow-backed Keras models work
    directly.

    For a fitted sklearn pipeline, ``attribute_after="scale"`` attributes to
    features after the named preprocessing step. Nested paths such as
    ``"preprocess__scale"`` are accepted. Always supply original observations and
    baselines: both are transformed together before constructing paths in the
    selected space. The default ``None`` explains original model inputs.

    Vector-valued automatic-gradient outputs are treated as class scores by
    default. Set ``output_kind="regression"`` for multi-output regression.
    Known sklearn and tree estimators declare their own output semantics and
    do not need this option. Keras 3 models use their configured TensorFlow,
    JAX, or PyTorch backend automatically.
    """

    def __init__(
        self,
        model: object,
        baseline: Any,
        *,
        baseline_weights: Optional[Any] = None,
        attribute_after: Optional[str] = None,
        n_steps: Optional[int] = None,
        check_completeness: bool = True,
        completeness_atol: float = 1e-6,
        completeness_rtol: float = 1e-4,
        fallback: Optional[str] = None,
        finite_difference_step: float = 1e-5,
        finite_difference_batch_size: int = 8192,
        gradient_batch_size: int = 8192,
        tree_grid_size: int = 1024,
        tree_max_refine: int = 4,
        probability_floor: Optional[float] = None,
        output_kind: Literal["auto", "regression", "classification"] = "auto",
    ) -> None:
        automatic_steps = n_steps is None
        resolved_n_steps = _DEFAULT_N_STEPS if automatic_steps else n_steps
        if (
            not isinstance(resolved_n_steps, int)
            or isinstance(resolved_n_steps, bool)
            or resolved_n_steps < 1
        ):
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(check_completeness, bool):
            raise ValueError("check_completeness must be a boolean")
        if fallback not in (None, "finite_difference", "tree_numeric"):
            raise ValueError(
                "fallback must be None, 'finite_difference', or 'tree_numeric'"
            )
        if output_kind not in ("auto", "regression", "classification"):
            raise ValueError(
                "output_kind must be 'auto', 'regression', or 'classification'"
            )
        if (
            not isinstance(finite_difference_step, Real)
            or isinstance(finite_difference_step, bool)
            or finite_difference_step <= 0
        ):
            raise ValueError("finite_difference_step must be a positive number")
        if (
            not isinstance(finite_difference_batch_size, int)
            or isinstance(finite_difference_batch_size, bool)
            or finite_difference_batch_size < 1
        ):
            raise ValueError("finite_difference_batch_size must be a positive integer")
        if (
            not isinstance(gradient_batch_size, int)
            or isinstance(gradient_batch_size, bool)
            or gradient_batch_size < 1
        ):
            raise ValueError("gradient_batch_size must be a positive integer")
        if (
            not isinstance(tree_grid_size, int)
            or isinstance(tree_grid_size, bool)
            or tree_grid_size < 1
        ):
            raise ValueError("tree_grid_size must be a positive integer")
        if (
            not isinstance(tree_max_refine, int)
            or isinstance(tree_max_refine, bool)
            or tree_max_refine < 0
        ):
            raise ValueError("tree_max_refine must be a nonnegative integer")
        if probability_floor is not None and (
            not isinstance(probability_floor, Real)
            or isinstance(probability_floor, bool)
            or not 0.0 < probability_floor < 1.0
        ):
            raise ValueError(
                "probability_floor must be strictly between 0 and 1"
            )
        self.source_model = model
        self.attribute_after = attribute_after
        self._pipeline_view = None
        if attribute_after is not None:
            from skgrad import pipeline_view
            self._pipeline_view = pipeline_view(model, after=attribute_after)
            model = self._pipeline_view.model
        self.model = model
        self.baseline = baseline
        self.baseline_weights = baseline_weights
        self.n_steps = resolved_n_steps
        self._automatic_steps = automatic_steps
        self._backend_factory: Optional[Callable[[int], Backend]] = None
        self.output_kind = output_kind
        self.check_completeness = check_completeness
        self.completeness_atol = self._validate_tolerance(
            "completeness_atol", completeness_atol
        )
        self.completeness_rtol = self._validate_tolerance(
            "completeness_rtol", completeness_rtol
        )
        backend_type = next((item for item in _BACKENDS if item.supports(model)), None)
        if backend_type is None:
            if output_kind != "auto":
                raise ValueError(
                    "output_kind is only needed for ambiguous automatic-"
                    "gradient outputs; sklearn and tree models declare their "
                    "output semantics"
                )
            if fallback is None:
                raise TypeError(
                    f"no Unified IG backend supports {type(model).__name__}; "
                    "set fallback='finite_difference' for a smooth sklearn "
                    "estimator or fallback='tree_numeric' for a supported "
                    "piecewise-constant tree model"
                )
            if fallback == "tree_numeric":
                self._backend = TreeIGNumericBackend(
                    model,
                    n_steps=tree_grid_size,
                    max_refine=tree_max_refine,
                    probability_floor=(
                        None
                        if probability_floor is None
                        else float(probability_floor)
                    ),
                )
                warnings.warn(
                    "Using numerical tree path-event detection; feature allocations "
                    "are approximate and depend on detecting all path crossings.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self._backend_factory = lambda steps: FiniteDifferenceBackend(
                    model, n_steps=steps,
                    relative_step=float(finite_difference_step),
                    batch_size=finite_difference_batch_size,
                )
                self._backend = self._backend_factory(resolved_n_steps)
                warnings.warn(
                    "Using finite-difference gradients; attribution may be substantially "
                    "slower than a specialized backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            if backend_type in (JaxBackend, TensorFlowBackend):
                self._backend_factory = lambda steps: backend_type(
                    model, n_steps=steps, output_kind=output_kind
                )
                self._backend = self._backend_factory(resolved_n_steps)
            elif backend_type is PyTorchBackend:
                self._backend_factory = lambda steps: backend_type(
                    model, n_steps=steps,
                    batch_size=gradient_batch_size,
                    output_kind=output_kind,
                )
                self._backend = self._backend_factory(resolved_n_steps)
            elif backend_type is SkgradBackend:
                if output_kind != "auto":
                    raise ValueError(
                        "output_kind is only needed for ambiguous automatic-"
                        "gradient outputs; sklearn and tree models declare their "
                        "output semantics"
                    )
                self._backend_factory = lambda steps: backend_type(
                    model, n_steps=steps, batch_size=gradient_batch_size
                )
                self._backend = self._backend_factory(resolved_n_steps)
            else:
                if output_kind != "auto":
                    raise ValueError(
                        "output_kind is only needed for ambiguous automatic-"
                        "gradient outputs; sklearn and tree models declare their "
                        "output semantics"
                    )
                self._backend = backend_type(model, n_steps=resolved_n_steps)

        if self._pipeline_view is not None and getattr(
            self._backend, "input_kind", "numpy"
        ) != "numpy":
            raise TypeError("attribute_after currently requires a NumPy-input backend")

    def __call__(self, data: Any) -> Explanation:
        feature_names = self._attribution_feature_names(data)
        input_kind = getattr(self._backend, "input_kind", "numpy")
        if input_kind == "torch":
            (
                normalized_data,
                normalized_baseline,
                normalized_weights,
            ) = normalize_torch_inputs(
                data,
                self.baseline,
                self.baseline_weights,
                device=self._backend.device,
                dtype=self._backend.dtype,
            )
            explanation_data = normalized_data.detach().cpu().numpy()
        elif input_kind == "jax":
            normalized_data, normalized_baseline, normalized_weights = (
                normalize_jax_inputs(
                    data,
                    self.baseline,
                    self.baseline_weights,
                    dtype=self._backend.dtype,
                )
            )
            explanation_data = np.asarray(normalized_data)
        elif input_kind == "tensorflow":
            normalized_data, normalized_baseline, normalized_weights = (
                normalize_tensorflow_inputs(
                    data,
                    self.baseline,
                    self.baseline_weights,
                    dtype=self._backend.dtype,
                )
            )
            explanation_data = normalized_data.numpy()
        else:
            normalized_data, normalized_baseline, normalized_weights = (
                self._normalize_numpy_inputs(data)
            )
            explanation_data = normalized_data
        backend_result = self._backend.explain(
            normalized_data, normalized_baseline, normalized_weights
        )
        completeness_error = self._completeness_error(backend_result)
        if (
            self.check_completeness
            and self._automatic_steps
            and self._backend_factory is not None
        ):
            for candidate_steps in _AUTOMATIC_N_STEPS:
                if not self._completeness_failed(
                    completeness_error, backend_result.output_values
                ):
                    break
                if candidate_steps <= self.n_steps:
                    continue
                self._backend = self._backend_factory(candidate_steps)
                self.n_steps = candidate_steps
                backend_result = self._backend.explain(
                    normalized_data, normalized_baseline, normalized_weights
                )
                completeness_error = self._completeness_error(backend_result)
        if self.check_completeness and self._completeness_failed(
            completeness_error, backend_result.output_values
        ):
            warnings.warn(
                "Integrated Gradients completeness tolerance was not met; "
                f"maximum absolute error is {np.max(np.abs(completeness_error)):.3g}. "
                "Increase the relevant numerical resolution (n_steps for "
                "gradient integration, or tree_grid_size/tree_max_refine "
                "for numerical trees).",
                RuntimeWarning,
                stacklevel=2,
            )
        return Explanation(
            values=backend_result.values,
            base_values=backend_result.base_values,
            data=explanation_data,
            feature_names=feature_names,
            output_names=backend_result.output_names,
            completeness_error=completeness_error,
            attribute_after=self.attribute_after,
        )

    def _attribution_feature_names(self, data: Any) -> Optional[Sequence[str]]:
        names = self._feature_names(data)
        if self._pipeline_view is None:
            return names
        return list(self._pipeline_view.get_feature_names_out(names))

    def _normalize_numpy_inputs(self, data: Any):
        if self._pipeline_view is None:
            return normalize_inputs(data, self.baseline, self.baseline_weights)
        view = self._pipeline_view
        rows, weights = _baseline_parts(self.baseline, self.baseline_weights)
        baseline = normalize_baseline(
            rows, n_features=self.source_model.n_features_in_
        )
        # Retain DataFrame labels for schema validation. Scalar baselines are
        # expanded in original coordinates, before applying preprocessing.
        if getattr(rows, "columns", None) is not None:
            baseline = rows
        transformed_data = view.transform(data)
        transformed_baseline = view.transform(baseline)
        return normalize_inputs(transformed_data, transformed_baseline, weights)

    def _completeness_failed(
        self, completeness_error: np.ndarray, output_values: np.ndarray
    ) -> bool:
        allowed_error = self.completeness_atol + self.completeness_rtol * np.abs(
            output_values
        )
        return bool(np.any(np.abs(completeness_error) > allowed_error))

    @staticmethod
    def _completeness_error(backend_result: Any) -> np.ndarray:
        if backend_result.output_values.ndim == 1:
            attribution_axes = tuple(range(1, backend_result.values.ndim))
        else:
            attribution_axes = tuple(range(1, backend_result.values.ndim - 1))
        attributed_difference = backend_result.values.sum(axis=attribution_axes)
        return backend_result.output_values - (
            attributed_difference + backend_result.base_values
        )

    @staticmethod
    def _feature_names(data: Any) -> Optional[Sequence[str]]:
        columns = getattr(data, "columns", None)
        return [str(column) for column in columns] if columns is not None else None

    @staticmethod
    def _validate_tolerance(name: str, value: float) -> float:
        if not isinstance(value, Real) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
        return float(value)
