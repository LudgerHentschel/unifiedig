"""Optional automatic-gradient Integrated Gradients for TensorFlow and Keras."""

from typing import Any, List, Literal, Optional

import numpy as np

from .._keras import keras_backend, keras_dtype, validate_keras_output
from .._loss import LossName
from ..tensorflow import TensorFlowModel
from .base import BackendResult, classification_score_result


class TensorFlowBackend:
    """Explain scalar or vector TensorFlow outputs with automatic gradients."""

    input_kind = "tensorflow"

    @classmethod
    def supports(cls, model: object) -> bool:
        if isinstance(model, TensorFlowModel):
            return True
        return keras_backend(model) == "tensorflow"

    def __init__(
        self,
        model: object,
        *,
        n_steps: int = 64,
        output_kind: Literal["auto", "regression", "classification"] = "auto",
    ) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow support requires optional dependencies. Install "
                "them with `pip install unifiedig[tensorflow]`."
            ) from exc

        self._tf = tf
        self.n_steps = n_steps
        self.output_kind = output_kind
        if isinstance(model, TensorFlowModel):
            self.model = model.predict_fn
            validate_keras_output(self.model, output_kind)
            self.call_kwargs = (
                {} if model.call_kwargs is None else dict(model.call_kwargs)
            )
            if keras_backend(self.model) == "tensorflow":
                self.call_kwargs.setdefault("training", False)
            self.output_names = model.output_names
            requested_dtype = (
                model.dtype if model.dtype is not None else keras_dtype(self.model)
            )
            self.dtype = self._resolve_dtype(requested_dtype)
        else:
            validate_keras_output(model, output_kind)
            self.model = model
            self.call_kwargs = {"training": False}
            self.output_names = None
            self.dtype = self._resolve_dtype(keras_dtype(model))

        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = 0.5 * (nodes + 1.0)
        self._weights = 0.5 * weights

    def explain(self, data: Any, baseline: Any, baseline_weights: Any) -> BackendResult:
        tf = self._tf
        output_values = self._forward(data)
        baseline_outputs = self._forward(baseline)
        n_outputs = 1 if output_values.shape.rank == 1 else int(output_values.shape[1])
        output_weights = tf.cast(baseline_weights, baseline_outputs.dtype)

        if n_outputs == 1:
            values = self._attribute_output(
                data, baseline, baseline_weights, target=None
            )
            mean_base = tf.reduce_sum(output_weights * baseline_outputs)
            base_values = tf.broadcast_to(mean_base, (data.shape[0],))
        else:
            by_output = [
                self._attribute_output(data, baseline, baseline_weights, target=target)
                for target in range(n_outputs)
            ]
            values = tf.stack(by_output, axis=-1)
            mean_base = tf.reduce_sum(
                output_weights[:, None] * baseline_outputs, axis=0
            )
            base_values = tf.broadcast_to(mean_base, (data.shape[0], n_outputs))

        values_array = values.numpy()
        base_array = base_values.numpy()
        output_array = output_values.numpy()
        if n_outputs >= 2 and self.output_kind != "regression":
            names = self.output_names
            if names is None:
                names = [str(index) for index in range(n_outputs)]
            return classification_score_result(
                values_array, base_array, output_array, names
            )

        if self.output_names not in (None, [], ()):
            if len(self.output_names) != n_outputs:
                raise ValueError(
                    f"TensorFlow output requires {n_outputs} output name"
                    f"{'s' if n_outputs != 1 else ''}"
                )
            output_names: Optional[List[str]] = list(self.output_names)
        else:
            output_names = (
                [str(index) for index in range(n_outputs)] if n_outputs >= 2 else None
            )
        return BackendResult(values_array, base_array, output_array, output_names)

    def explain_loss(
        self,
        data: Any,
        baseline: Any,
        baseline_weights: Any,
        y: np.ndarray,
        loss: LossName,
    ) -> BackendResult:
        """Attribute a scalar loss with TensorFlow automatic gradients."""

        tf = self._tf
        targets = tf.convert_to_tensor(y)
        nodes = tf.convert_to_tensor(self._nodes, dtype=data.dtype)
        weights = tf.convert_to_tensor(self._weights, dtype=data.dtype)
        node_shape = (self.n_steps,) + (1,) * data.shape.rank
        nodes = tf.reshape(nodes, node_shape)
        values = tf.zeros_like(data)
        for baseline_row, baseline_weight in zip(baseline, baseline_weights):
            delta = data - baseline_row
            path = baseline_row + nodes * delta[None, ...]
            flat_path = tf.reshape(path, (-1,) + tuple(data.shape[1:]))
            path_targets = tf.tile(targets, [self.n_steps])
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(flat_path)
                path_loss = self._loss_values(
                    self._forward(flat_path), path_targets, loss
                )
            gradient = tape.gradient(
                path_loss,
                flat_path,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            gradients = tf.reshape(gradient, tf.shape(path))
            weight_shape = (self.n_steps,) + (1,) * data.shape.rank
            integrated = tf.reduce_sum(
                tf.reshape(weights, weight_shape) * gradients, axis=0
            )
            values += baseline_weight * delta * integrated
        endpoint_loss = self._loss_values(self._forward(data), targets, loss)
        base_values = tf.zeros_like(endpoint_loss)
        for baseline_row, weight in zip(baseline, baseline_weights):
            repeated = tf.broadcast_to(baseline_row, tf.shape(data))
            base_values += weight * self._loss_values(
                self._forward(repeated), targets, loss
            )
        return BackendResult(
            values.numpy(), base_values.numpy(), endpoint_loss.numpy(), None
        )

    def _loss_values(self, output: Any, y: Any, loss: LossName) -> Any:
        tf = self._tf
        if loss == "squared_error":
            scalar = output if output.shape.rank == 1 else output[:, 0]
            return tf.square(scalar - tf.cast(y, scalar.dtype))
        if output.shape.rank == 1:
            return tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(y, output.dtype), logits=output
            )
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=output
        )

    def _attribute_output(
        self,
        data: Any,
        baseline: Any,
        baseline_weights: Any,
        *,
        target: Optional[int],
    ) -> Any:
        tf = self._tf
        nodes = tf.convert_to_tensor(self._nodes, dtype=data.dtype)
        weights = tf.convert_to_tensor(self._weights, dtype=data.dtype)
        node_shape = (self.n_steps,) + (1,) * data.shape.rank
        nodes = tf.reshape(nodes, node_shape)
        total = tf.zeros_like(data)

        for baseline_row, baseline_weight in zip(baseline, baseline_weights):
            delta = data - baseline_row
            path = baseline_row + nodes * delta[None, ...]
            flat_shape = (-1,) + tuple(data.shape[1:])
            flat_path = tf.reshape(path, flat_shape)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(flat_path)
                output = self._forward(flat_path)
                selected = output if target is None else output[:, target]
                output_sum = tf.reduce_sum(selected)
            gradient = tape.gradient(
                output_sum,
                flat_path,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            gradients = tf.reshape(gradient, tf.shape(path))
            weight_shape = (self.n_steps,) + (1,) * data.shape.rank
            integrated = tf.reduce_sum(
                tf.reshape(weights, weight_shape) * gradients, axis=0
            )
            total = total + baseline_weight * delta * integrated
        return total

    def _forward(self, data: Any) -> Any:
        tf = self._tf
        output = self.model(data, **self.call_kwargs)
        if isinstance(output, (list, tuple, dict)):
            raise ValueError(
                "TensorFlow models must return one tensor, not a structured "
                "collection of output tensors"
            )
        output = tf.convert_to_tensor(output)
        rank = output.shape.rank
        if rank == 1 and output.shape[0] == data.shape[0]:
            return output
        if rank == 2 and output.shape == (data.shape[0], 1):
            return output[:, 0]
        if rank == 2 and output.shape[0] == data.shape[0] and output.shape[1] >= 2:
            return output
        raise ValueError(
            "TensorFlow models must return one scalar or one output vector "
            "for every sample"
        )

    def _resolve_dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return None
        resolved = self._tf.as_dtype(dtype)
        if not resolved.is_floating:
            raise TypeError("TensorFlow input dtype must be floating point")
        return resolved
