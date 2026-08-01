"""Integrated Gradients for fitted scikit-learn multilayer perceptrons."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import BackendResult


FloatArray = NDArray[np.floating]


class SklearnMLPBackend:
    """Analytic MLP gradients integrated with Gauss–Legendre quadrature."""

    @classmethod
    def supports(cls, model: object) -> bool:
        return isinstance(model, (MLPRegressor, MLPClassifier))

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        if not self.supports(model):
            raise TypeError("SklearnMLPBackend received an unsupported model")
        if not hasattr(model, "coefs_"):
            raise ValueError("model must be fitted before creating an Explainer")
        if isinstance(model, MLPClassifier) and len(model.classes_) != 2:
            raise ValueError("V1 supports only binary MLPClassifier")
        if isinstance(model, MLPRegressor) and model.out_activation_ != "identity":
            raise ValueError("V1 supports only identity-output MLPRegressor models")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")

        self.model = model
        self.n_steps = n_steps
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = (nodes + 1.0) / 2.0
        self._weights = weights / 2.0

    def explain(self, data: FloatArray, baseline: FloatArray) -> BackendResult:
        if data.shape[1] != self.model.n_features_in_:
            raise ValueError(f"data must have {self.model.n_features_in_} features")

        difference = data - baseline
        integrated_gradient: Optional[FloatArray] = None
        for node, weight in zip(self._nodes, self._weights):
            path_data = baseline + node * difference
            gradient = self._input_gradient(path_data)
            if integrated_gradient is None:
                integrated_gradient = weight * gradient
            else:
                integrated_gradient += weight * gradient

        assert integrated_gradient is not None
        base_values = self._explained_output(baseline)
        if integrated_gradient.shape[-1] == 1:
            values = difference * integrated_gradient[..., 0]
            return values, base_values[:, 0], self._output_names()

        values = difference[:, :, None] * integrated_gradient
        return values, base_values, self._output_names()

    def _forward_hidden(self, data: FloatArray) -> Tuple[FloatArray, List[FloatArray]]:
        activation = data
        hidden_outputs: List[FloatArray] = []
        for weights, intercept in zip(self.model.coefs_[:-1], self.model.intercepts_[:-1]):
            pre_activation = activation @ weights + intercept
            activation = self._activate(pre_activation)
            hidden_outputs.append(activation)
        return activation, hidden_outputs

    def _explained_output(self, data: FloatArray) -> FloatArray:
        activation, _ = self._forward_hidden(data)
        output = activation @ self.model.coefs_[-1] + self.model.intercepts_[-1]
        return np.asarray(output, dtype=float).reshape(data.shape[0], -1)

    def _input_gradient(self, data: FloatArray) -> FloatArray:
        _, hidden_outputs = self._forward_hidden(data)
        output_weights = np.asarray(self.model.coefs_[-1], dtype=float)
        jacobian = np.broadcast_to(
            output_weights.T, (data.shape[0], *output_weights.T.shape)
        ).copy()

        for layer in range(len(hidden_outputs) - 1, -1, -1):
            derivative = self._activation_derivative(hidden_outputs[layer])
            jacobian *= derivative[:, None, :]
            jacobian = np.einsum(
                "sou,iu->soi", jacobian, np.asarray(self.model.coefs_[layer])
            )
        return np.transpose(jacobian, (0, 2, 1))

    def _activate(self, values: FloatArray) -> FloatArray:
        activation = self.model.activation
        if activation == "identity":
            return values
        if activation == "relu":
            return np.maximum(values, 0.0)
        if activation == "tanh":
            return np.tanh(values)
        if activation == "logistic":
            result = np.empty_like(values)
            positive = values >= 0
            result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
            exponential = np.exp(values[~positive])
            result[~positive] = exponential / (1.0 + exponential)
            return result
        raise ValueError(f"unsupported sklearn MLP activation: {activation}")

    def _activation_derivative(self, activated: FloatArray) -> FloatArray:
        activation = self.model.activation
        if activation == "identity":
            return np.ones_like(activated)
        if activation == "relu":
            return (activated > 0.0).astype(float)
        if activation == "tanh":
            return 1.0 - activated**2
        if activation == "logistic":
            return activated * (1.0 - activated)
        raise ValueError(f"unsupported sklearn MLP activation: {activation}")

    def _output_names(self) -> Optional[Sequence[str]]:
        if isinstance(self.model, MLPClassifier):
            return [str(self.model.classes_[1])]
        n_outputs = int(self.model.n_outputs_)
        return [str(index) for index in range(n_outputs)] if n_outputs > 1 else None

