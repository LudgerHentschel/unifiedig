"""Input and baseline normalization shared by all backends."""

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def normalize_data(data: Any) -> NDArray[np.floating]:
    """Convert one sample or a sample matrix to a finite 2-D float array."""

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("data must be a non-empty 1-D sample or 2-D sample matrix")
    if not np.isfinite(array).all():
        raise ValueError("data must contain only finite values")
    return array


def normalize_baseline(
    baseline: Any, *, n_samples: int, n_features: int
) -> NDArray[np.floating]:
    """Broadcast a scalar, feature vector, or baseline matrix over samples."""

    array = np.asarray(baseline, dtype=float)
    if array.ndim == 0:
        array = np.full((1, n_features), array.item())
    elif array.ndim == 1:
        if array.shape[0] != n_features:
            raise ValueError(f"baseline must have {n_features} features")
        array = array.reshape(1, -1)
    elif array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(f"baseline must have shape ({n_features},), (1, {n_features}), or (n_samples, {n_features})")

    if array.shape[0] not in (1, n_samples):
        raise ValueError("baseline must contain either one row or one row per sample")
    if not np.isfinite(array).all():
        raise ValueError("baseline must contain only finite values")
    return np.broadcast_to(array, (n_samples, n_features)).copy()


def normalize_inputs(data: Any, baseline: Any) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Normalize data and its baseline into matching sample matrices."""

    normalized_data = normalize_data(data)
    normalized_baseline = normalize_baseline(
        baseline,
        n_samples=normalized_data.shape[0],
        n_features=normalized_data.shape[1],
    )
    return normalized_data, normalized_baseline

