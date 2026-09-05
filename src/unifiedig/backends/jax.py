"""Optional automatic-gradient Integrated Gradients for JAX functions."""

import warnings
from typing import Any, List, Literal, Optional

import numpy as np

from .._keras import keras_backend, keras_dtype, validate_keras_output
from .._loss import LossName
from ..jax import JaxModel
from .base import BackendResult, classification_score_result


class JaxBackend:
    """Explain scalar or vector JAX outputs with automatic gradients."""

    input_kind = "jax"

    @classmethod
    def supports(cls, model: object) -> bool:
        return isinstance(model, JaxModel) or keras_backend(model) == "jax"

    def __init__(
        self,
        model: object,
        *,
        n_steps: int = 64,
        output_kind: Literal["auto", "regression", "classification"] = "auto",
    ) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "JAX support requires optional dependencies. Install them "
                "with `pip install unifiedig[jax]`."
            ) from exc

        if isinstance(model, JaxModel):
            validate_keras_output(model.predict_fn, output_kind)
            self.model = model
        else:
            validate_keras_output(model, output_kind)
            self.model = JaxModel(
                lambda data: model(data, training=False),
                dtype=keras_dtype(model),
            )
        self._jax = jax
        self._jnp = jnp
        self.n_steps = n_steps
        self.output_kind = output_kind
        self.dtype = self._resolve_dtype(self.model.dtype)
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = 0.50 * (nodes + 1.00)
        self._weights = 0.50 * weights

    def explain(self, data: Any, baseline: Any, baseline_weights: Any) -> BackendResult:
        jnp = self._jnp
        output_values = self._forward(data)
        baseline_outputs = self._forward(baseline)
        n_outputs = 1 if output_values.ndim == 1 else output_values.shape[1]

        if n_outputs == 1:
            values = self._attribute_output(
                data, baseline, baseline_weights, target=None
            )
            mean_base = jnp.sum(baseline_weights * baseline_outputs)
            base_values = jnp.broadcast_to(mean_base, (data.shape[0],))
        else:
            by_output = [
                self._attribute_output(data, baseline, baseline_weights, target=target)
                for target in range(n_outputs)
            ]
            values = jnp.stack(by_output, axis=-1)
            mean_base = jnp.sum(baseline_weights[:, None] * baseline_outputs, axis=0)
            base_values = jnp.broadcast_to(mean_base, (data.shape[0], n_outputs))

        values_array = np.asarray(values)
        base_array = np.asarray(base_values)
        output_array = np.asarray(output_values)
        if n_outputs >= 2 and self.output_kind != "regression":
            names = self.model.output_names
            if names is None:
                names = [str(index) for index in range(n_outputs)]
            return classification_score_result(
                values_array, base_array, output_array, names
            )
        if self.model.output_names not in (None, [], ()):
            expected_names = n_outputs
            if len(self.model.output_names) != expected_names:
                raise ValueError(
                    f"JAX output requires {expected_names} output name"
                    f"{'s' if expected_names != 1 else ''}"
                )
            output_names: Optional[List[str]] = list(self.model.output_names)
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
        """Attribute a scalar loss with JAX automatic gradients."""

        jax, jnp = self._jax, self._jnp
        targets = jnp.asarray(y)
        nodes = jnp.asarray(self._nodes, dtype=data.dtype)
        weights = jnp.asarray(self._weights, dtype=data.dtype)
        values = jnp.zeros_like(data)

        def summed_loss(path_data: Any) -> Any:
            return jnp.sum(self._loss_values(self._forward(path_data), targets, loss))

        gradient = jax.grad(summed_loss)
        for baseline_row, baseline_weight in zip(baseline, baseline_weights):
            delta = data - baseline_row

            def accumulate(
                current: Any,
                node_weight: Any,
                baseline_row: Any = baseline_row,
                delta: Any = delta,
            ) -> Any:
                node, weight = node_weight
                return current + weight * gradient(baseline_row + node * delta), None

            integrated, _ = jax.lax.scan(
                accumulate, jnp.zeros_like(data), (nodes, weights)
            )
            values = values + baseline_weight * delta * integrated
        endpoint_loss = self._loss_values(self._forward(data), targets, loss)
        base_values = jnp.zeros_like(endpoint_loss)
        for baseline_row, weight in zip(baseline, baseline_weights):
            repeated = jnp.broadcast_to(baseline_row, data.shape)
            base_values = base_values + weight * self._loss_values(
                self._forward(repeated), targets, loss
            )
        return BackendResult(
            np.asarray(values), np.asarray(base_values), np.asarray(endpoint_loss), None
        )

    def _loss_values(self, output: Any, y: Any, loss: LossName) -> Any:
        jnp = self._jnp
        if loss == "squared_error":
            scalar = output if output.ndim == 1 else output[:, 0]
            return (scalar - y.astype(scalar.dtype)) ** 2
        if output.ndim == 1:
            return jnp.logaddexp(0.0, output) - y.astype(output.dtype) * output
        maximum = jnp.max(output, axis=1)
        return (
            maximum
            + jnp.log(jnp.sum(jnp.exp(output - maximum[:, None]), axis=1))
            - output[jnp.arange(y.shape[0]), y.astype(int)]
        )

    def _attribute_output(
        self,
        data: Any,
        baseline: Any,
        baseline_weights: Any,
        *,
        target: Optional[int],
    ) -> Any:
        jax = self._jax
        jnp = self._jnp
        nodes = jnp.asarray(self._nodes, dtype=data.dtype)
        weights = jnp.asarray(self._weights, dtype=data.dtype)
        total = jnp.zeros_like(data)

        def output_sum(path_data: Any) -> Any:
            output = self._forward(path_data)
            return jnp.sum(output if target is None else output[:, target])

        gradient = jax.grad(output_sum)
        for baseline_row, baseline_weight in zip(baseline, baseline_weights):
            delta = data - baseline_row

            def accumulate(
                current: Any,
                node_weight: Any,
                baseline_row: Any = baseline_row,
                delta: Any = delta,
            ) -> Any:
                node, weight = node_weight
                return current + weight * gradient(baseline_row + node * delta), None

            integrated, _ = jax.lax.scan(
                accumulate, jnp.zeros_like(data), (nodes, weights)
            )
            total = total + baseline_weight * delta * integrated
        return total

    def _forward(self, data: Any) -> Any:
        jax = self._jax
        jnp = self._jnp
        kwargs = {} if self.model.call_kwargs is None else self.model.call_kwargs

        def call_one_or_batch(value: Any) -> Any:
            if self.model.params is None:
                return self.model.predict_fn(value, **kwargs)
            return self.model.predict_fn(self.model.params, value, **kwargs)

        output = (
            jax.vmap(call_one_or_batch)(data)
            if self.model.vectorize
            else call_one_or_batch(data)
        )
        output = jnp.asarray(output)
        if output.ndim == 1 and output.shape[0] == data.shape[0]:
            return output
        if output.ndim == 2 and output.shape == (data.shape[0], 1):
            return output[:, 0]
        if (
            output.ndim == 2
            and output.shape[0] == data.shape[0]
            and output.shape[1] >= 2
        ):
            return output
        raise ValueError(
            "JAX models must return one scalar or one output vector for every sample"
        )

    def _resolve_dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return None
        requested = self._jnp.dtype(dtype)
        if not self._jnp.issubdtype(requested, self._jnp.floating):
            raise TypeError("JAX input dtype must be floating point")
        actual = self._jax.dtypes.canonicalize_dtype(requested)
        if actual != requested:
            warnings.warn(
                f"Requested dtype {requested} was downcast to {actual} because "
                "JAX 64-bit mode is disabled.",
                RuntimeWarning,
                stacklevel=3,
            )
        return actual
