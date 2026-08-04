"""Input and baseline normalization shared by all backends."""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def normalize_data(data: Any) -> NDArray[np.floating]:
    """Convert one sample or a sample matrix to a finite 2-D float array."""

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("data must be a non-empty 1-D sample or 2-D sample matrix")
    if not np.isfinite(array).all():
        raise ValueError("data must contain only finite values")
    return array


def normalize_baseline(
    baseline: Any, *, n_features: int
) -> NDArray[np.floating]:
    """Normalize a scalar, feature vector, or shared baseline distribution."""

    array = np.asarray(baseline, dtype=float)
    if array.ndim == 0:
        array = np.full((1, n_features), array.item())
    elif array.ndim == 1:
        if array.shape[0] != n_features:
            raise ValueError(f"baseline must have {n_features} features")
        array = array.reshape(1, -1)
    elif array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(
            f"baseline must have shape ({n_features},) or "
            f"(n_baselines, {n_features})"
        )
    if array.shape[0] == 0:
        raise ValueError("baseline distribution must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError("baseline must contain only finite values")
    return np.ascontiguousarray(array)


def _baseline_parts(
    baseline: Any, baseline_weights: Optional[Any]
) -> Tuple[Any, Optional[Any]]:
    """Resolve a matrix or a background object exposing rows and weights."""

    has_rows = hasattr(baseline, "rows")
    has_weights = hasattr(baseline, "weights")
    if has_rows != has_weights:
        raise TypeError(
            "a baseline background object must expose both rows and weights"
        )
    if has_rows:
        if baseline_weights is not None:
            raise ValueError(
                "baseline_weights must be omitted when baseline supplies weights"
            )
        return baseline.rows, baseline.weights
    return baseline, baseline_weights


def normalize_baseline_weights(
    weights: Optional[Any], *, n_baselines: int
) -> NDArray[np.floating]:
    """Validate and normalize weights for a shared baseline distribution."""

    if weights is None:
        return np.full(n_baselines, 1.0 / n_baselines, dtype=float)
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.shape[0] != n_baselines:
        raise ValueError("baseline_weights must align with baseline rows")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("baseline_weights must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("baseline_weights must have a positive sum")
    return np.ascontiguousarray(array / total)


def normalize_inputs(
    data: Any, baseline: Any, baseline_weights: Optional[Any] = None
) -> Tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
]:
    """Normalize data and a weighted baseline distribution."""

    normalized_data = normalize_data(data)
    baseline, baseline_weights = _baseline_parts(baseline, baseline_weights)
    normalized_baseline = normalize_baseline(
        baseline,
        n_features=normalized_data.shape[1],
    )
    normalized_weights = normalize_baseline_weights(
        baseline_weights, n_baselines=normalized_baseline.shape[0]
    )
    positive = normalized_weights > 0
    return (
        normalized_data,
        normalized_baseline[positive],
        normalized_weights[positive],
    )


def normalize_torch_inputs(
    data: Any,
    baseline: Any,
    baseline_weights: Optional[Any] = None,
    *,
    device: Any,
    dtype: Any,
) -> Tuple[Any, Any, Any]:
    """Normalize PyTorch inputs and a weighted baseline distribution."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - selected only with PyTorch present
        raise ImportError(
            "PyTorch support is optional. Install it with `pip install unifiedig[torch]`."
        ) from exc

    data_source = data.to_numpy() if hasattr(data, "to_numpy") else data
    if isinstance(data_source, np.ndarray) and not data_source.flags.writeable:
        data_source = data_source.copy()
    if dtype is None:
        source = torch.as_tensor(data_source)
        dtype = (
            source.dtype
            if source.is_floating_point()
            else torch.get_default_dtype()
        )
    normalized_data = torch.as_tensor(
        data_source, device=device, dtype=dtype
    ).detach()
    if normalized_data.ndim == 1:
        normalized_data = normalized_data.reshape(1, -1)
    if normalized_data.ndim < 2 or normalized_data.shape[0] == 0:
        raise ValueError("data must contain a non-empty leading sample dimension")
    if not torch.isfinite(normalized_data).all():
        raise ValueError("data must contain only finite values")

    baseline, baseline_weights = _baseline_parts(baseline, baseline_weights)
    baseline_source = (
        baseline.to_numpy() if hasattr(baseline, "to_numpy") else baseline
    )
    if isinstance(baseline_source, np.ndarray) and not baseline_source.flags.writeable:
        baseline_source = baseline_source.copy()
    normalized_baseline = torch.as_tensor(
        baseline_source,
        device=normalized_data.device,
        dtype=normalized_data.dtype,
    ).detach()
    sample_shape = normalized_data.shape[1:]
    if normalized_baseline.ndim == 0:
        normalized_baseline = torch.full(
            (1, *sample_shape),
            normalized_baseline.item(),
            device=normalized_data.device,
            dtype=normalized_data.dtype,
        )
    elif tuple(normalized_baseline.shape) == tuple(sample_shape):
        normalized_baseline = normalized_baseline.unsqueeze(0)
    elif (
        normalized_baseline.ndim != normalized_data.ndim
        or tuple(normalized_baseline.shape[1:]) != tuple(sample_shape)
    ):
        raise ValueError(
            "baseline must be scalar, have the input sample shape, or be a "
            "baseline distribution with matching sample shape"
        )

    if normalized_baseline.shape[0] == 0:
        raise ValueError("baseline distribution must contain at least one row")
    if not torch.isfinite(normalized_baseline).all():
        raise ValueError("baseline must contain only finite values")
    normalized_weights_array = normalize_baseline_weights(
        baseline_weights, n_baselines=normalized_baseline.shape[0]
    )
    positive = normalized_weights_array > 0
    normalized_baseline = normalized_baseline[
        torch.as_tensor(positive, device=normalized_data.device)
    ]
    normalized_weights = torch.as_tensor(
        normalized_weights_array[positive],
        device=normalized_data.device,
        dtype=normalized_data.dtype,
    )
    return normalized_data, normalized_baseline, normalized_weights


def normalize_jax_inputs(
    data: Any,
    baseline: Any,
    baseline_weights: Optional[Any] = None,
    *,
    dtype: Any = None,
) -> Tuple[Any, Any, Any]:
    """Normalize JAX inputs and a weighted baseline distribution."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - selected only with JAX present
        raise ImportError(
            "JAX support is optional. Install it with `pip install unifiedig[jax]`."
        ) from exc

    data_source = data.to_numpy() if hasattr(data, "to_numpy") else data
    source_dtype = np.asarray(data_source).dtype
    if dtype is None:
        requested = (
            source_dtype
            if np.issubdtype(source_dtype, np.floating)
            else np.float32
        )
        dtype = jax.dtypes.canonicalize_dtype(requested)
    normalized_data = jnp.asarray(data_source, dtype=dtype)
    if normalized_data.ndim == 1:
        normalized_data = normalized_data.reshape(1, -1)
    if normalized_data.ndim < 2 or normalized_data.shape[0] == 0:
        raise ValueError("data must contain a non-empty leading sample dimension")
    if not bool(jnp.isfinite(normalized_data).all()):
        raise ValueError("data must contain only finite values")

    baseline, baseline_weights = _baseline_parts(baseline, baseline_weights)
    baseline_source = (
        baseline.to_numpy() if hasattr(baseline, "to_numpy") else baseline
    )
    normalized_baseline = jnp.asarray(baseline_source, dtype=normalized_data.dtype)
    sample_shape = normalized_data.shape[1:]
    if normalized_baseline.ndim == 0:
        normalized_baseline = jnp.full(
            (1, *sample_shape), normalized_baseline.item(), dtype=normalized_data.dtype
        )
    elif tuple(normalized_baseline.shape) == tuple(sample_shape):
        normalized_baseline = normalized_baseline[None, ...]
    elif (
        normalized_baseline.ndim != normalized_data.ndim
        or tuple(normalized_baseline.shape[1:]) != tuple(sample_shape)
    ):
        raise ValueError(
            "baseline must be scalar, have the input sample shape, or be a "
            "baseline distribution with matching sample shape"
        )
    if normalized_baseline.shape[0] == 0:
        raise ValueError("baseline distribution must contain at least one row")
    if not bool(jnp.isfinite(normalized_baseline).all()):
        raise ValueError("baseline must contain only finite values")

    weight_array = normalize_baseline_weights(
        baseline_weights, n_baselines=normalized_baseline.shape[0]
    )
    positive = weight_array > 0
    return (
        normalized_data,
        normalized_baseline[jnp.asarray(positive)],
        jnp.asarray(weight_array[positive], dtype=normalized_data.dtype),
    )
