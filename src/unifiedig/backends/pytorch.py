"""Optional native automatic-gradient Integrated Gradients for PyTorch."""

from typing import Any, Dict, Literal, Optional

import numpy as np

from .._keras import validate_keras_output
from .base import BackendResult, classification_score_result


class PyTorchBackend:
    """Explain scalar or vector PyTorch outputs with native autograd."""

    input_kind = "torch"

    @classmethod
    def supports(cls, model: object) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return isinstance(model, torch.nn.Module)

    def __init__(
        self,
        model: object,
        *,
        n_steps: int = 64,
        batch_size: int = 8192,
        output_kind: Literal["auto", "regression", "classification"] = "auto",
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch support requires an optional dependency. Install it "
                "with `pip install unifiedig[torch]`."
            ) from exc

        self.model = model
        validate_keras_output(model, output_kind)
        self._torch = torch
        self.n_steps = n_steps
        self.batch_size = batch_size
        tensors = list(model.parameters()) + list(model.buffers())
        reference = next((tensor for tensor in tensors if tensor.is_floating_point()), None)
        self.device = reference.device if reference is not None else None
        self.dtype = reference.dtype if reference is not None else None
        self.output_kind = output_kind
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        self._nodes = 0.5 * (nodes + 1.0)
        self._weights = 0.5 * weights

    def explain(
        self, data: Any, baseline: Any, baseline_weights: Any
    ) -> BackendResult:
        module_states: Dict[Any, bool] = {
            module: module.training for module in self.model.modules()
        }
        self.model.eval()
        try:
            output_values = self._output(data)
            baseline_outputs = self._output(baseline)
            n_outputs = 1 if output_values.ndim == 1 else output_values.shape[1]
            if n_outputs == 1:
                mean_base_value = (baseline_weights * baseline_outputs).sum()
                base_values = mean_base_value.expand(data.shape[0]).clone()
                attributions = self._attribute_output(
                    data, baseline, baseline_weights, target=None
                )
            else:
                mean_base_value = (
                    baseline_weights[:, None] * baseline_outputs
                ).sum(dim=0)
                base_values = mean_base_value.expand(
                    data.shape[0], n_outputs
                ).clone()
                by_output = [
                    self._attribute_output(
                        data, baseline, baseline_weights, target=target
                    )
                    for target in range(n_outputs)
                ]
                attributions = self._torch.stack(by_output, dim=-1)
        finally:
            for module, training in module_states.items():
                module.training = training

        values_array = self._to_numpy(attributions)
        base_array = self._to_numpy(base_values)
        output_array = self._to_numpy(output_values)
        if n_outputs >= 2 and self.output_kind != "regression":
            return classification_score_result(
                values_array,
                base_array,
                output_array,
                [str(index) for index in range(n_outputs)],
            )
        output_names = (
            [str(index) for index in range(n_outputs)]
            if n_outputs >= 2
            else None
        )
        return BackendResult(values_array, base_array, output_array, output_names)

    def _attribute_output(
        self,
        data: Any,
        baseline: Any,
        baseline_weights: Any,
        *,
        target: Optional[int],
    ) -> Any:
        torch = self._torch
        nodes = torch.as_tensor(self._nodes, dtype=data.dtype, device=data.device)
        weights = torch.as_tensor(
            self._weights, dtype=data.dtype, device=data.device
        )
        n_samples = data.shape[0]
        baselines_per_batch = max(1, self.batch_size // n_samples)
        attributions = torch.zeros_like(data)

        for start in range(0, baseline.shape[0], baselines_per_batch):
            stop = min(start + baselines_per_batch, baseline.shape[0])
            baseline_rows = baseline[start:stop]
            row_weights = baseline_weights[start:stop]
            delta = data.unsqueeze(0) - baseline_rows.unsqueeze(1)
            integrated = torch.zeros_like(delta)
            for node, weight in zip(nodes, weights):
                path = (
                    baseline_rows.unsqueeze(1) + node * delta
                ).reshape(-1, *data.shape[1:]).detach().requires_grad_(True)
                with torch.enable_grad():
                    output = self._forward(path)
                    selected = output if target is None else output[:, target]
                    gradient = torch.autograd.grad(selected.sum(), path)[0]
                integrated += weight * gradient.reshape_as(delta)
            weight_shape = (len(baseline_rows),) + (1,) * data.ndim
            attributions += torch.sum(
                row_weights.reshape(weight_shape) * delta * integrated,
                dim=0,
            )
        return attributions

    def _forward(self, data: Any) -> Any:
        output = self.model(data)
        if output.ndim == 1:
            return output
        if output.ndim == 2 and output.shape[1] == 1:
            return output[:, 0]
        if output.ndim == 2 and output.shape[1] >= 2:
            return output
        raise ValueError(
            "PyTorch models must return one scalar or one output vector for "
            "every sample"
        )

    def _output(self, data: Any) -> Any:
        import torch

        with torch.no_grad():
            return self._forward(data)

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy()
