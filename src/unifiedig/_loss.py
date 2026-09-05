"""Canonical loss values and score derivatives used by loss attribution."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

LossName = Literal["squared_error", "log_loss"]


def validate_targets(
    model: object, y: Any, *, loss: LossName, n_outputs: int
) -> np.ndarray:
    """Normalize regression outcomes or classification labels."""

    targets = np.asarray(y).reshape(-1)
    if loss == "squared_error":
        if n_outputs != 1:
            raise ValueError("squared_error requires exactly one model output")
        try:
            targets = targets.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("squared_error outcomes must be numeric") from exc
        if not np.isfinite(targets).all():
            raise ValueError("squared_error outcomes must be finite")
        return targets

    if n_outputs == 1:
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 2:
            negative, positive = classes
            valid = (targets == negative) | (targets == positive)
            if not np.all(valid):
                raise ValueError("y contains a label outside the model classes")
            return (targets == positive).astype(float)
        try:
            binary = targets.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("binary log_loss requires labels in {0, 1}") from exc
        if not np.all((binary == 0.0) | (binary == 1.0)):
            raise ValueError("binary log_loss requires labels in {0, 1}")
        return binary

    classes = np.asarray(getattr(model, "classes_", np.arange(n_outputs)))
    if len(classes) != n_outputs:
        raise ValueError("model classes must align with its score outputs")
    indices = np.empty(len(targets), dtype=int)
    for index, label in enumerate(targets):
        matches = np.flatnonzero(classes == label)
        if len(matches) != 1:
            raise ValueError("y contains a label outside the model classes")
        indices[index] = matches[0]
    return indices


def loss_values(output: np.ndarray, y: np.ndarray, *, loss: LossName) -> np.ndarray:
    """Return one loss value per observation from raw model outputs."""

    scores = np.asarray(output)
    if scores.ndim == 1:
        scores = scores[:, None]
    if loss == "squared_error":
        return (y - scores[:, 0]) ** 2
    if scores.shape[1] == 1:
        return np.logaddexp(0.0, scores[:, 0]) - y * scores[:, 0]
    maximum = scores.max(axis=1)
    return (
        maximum
        + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
        - scores[np.arange(len(y)), y]
    )


def loss_output_gradient(
    output: np.ndarray, y: np.ndarray, *, loss: LossName
) -> np.ndarray:
    """Return the analytical derivative of loss with respect to raw outputs."""

    scores = np.asarray(output)
    if scores.ndim == 1:
        scores = scores[:, None]
    if loss == "squared_error":
        return (2.0 * (scores[:, 0] - y))[:, None]
    if scores.shape[1] == 1:
        probability = np.exp(-np.logaddexp(0.0, -scores[:, 0]))
        return (probability - y)[:, None]
    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    probability[np.arange(len(y)), y] -= 1.0
    return probability
