"""Private helpers for dispatching Keras 3 models to native backends."""

from typing import Any, Optional


def keras_backend(model: object) -> Optional[str]:
    """Return the active backend for a Keras model, without importing it eagerly."""

    modules = {base.__module__ for base in type(model).__mro__}
    if any(name.startswith("keras.") for name in modules):
        try:
            import keras
        except ImportError:  # pragma: no cover - model could not exist normally
            return None
        if isinstance(model, keras.Model):
            return str(keras.config.backend())
    if any(
        name.startswith(("tf_keras.", "tensorflow.python.keras."))
        for name in modules
    ):
        try:
            import tensorflow as tf
        except ImportError:  # pragma: no cover - model could not exist normally
            return None
        if isinstance(model, tf.keras.Model):
            return "tensorflow"
    return None


def keras_probability_activation(model: object) -> Optional[str]:
    """Identify a directly visible sigmoid or softmax output activation."""

    if keras_backend(model) is None:
        return None
    layers = getattr(model, "layers", None)
    if not layers:
        return None
    final_layer = layers[-1]
    activation = getattr(final_layer, "activation", None)
    name = getattr(activation, "__name__", None)
    if name is None:
        name = type(final_layer).__name__.lower()
    return name if name in {"sigmoid", "softmax"} else None


def validate_keras_output(model: object, output_kind: str) -> None:
    """Reject visible probability heads for score-classification attribution."""

    activation = keras_probability_activation(model)
    if activation is None or output_kind == "regression":
        return
    raise ValueError(
        f"Keras model output uses {activation}, which produces probabilities; "
        "Unified IG classification requires logits or raw scores. Explain a "
        "model that exposes the pre-activation output, or set "
        "output_kind='regression' if this is genuinely a regression output."
    )


def keras_dtype(model: object) -> Any:
    """Return a Keras model's computation dtype when available."""

    input_dtype = getattr(model, "input_dtype", None)
    if input_dtype is not None:
        return input_dtype
    inputs = getattr(model, "inputs", None)
    if inputs:
        return getattr(inputs[0], "dtype", None)
    return getattr(model, "compute_dtype", None)
