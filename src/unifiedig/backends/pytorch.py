"""Optional Captum-backed Integrated Gradients for PyTorch modules."""

from typing import Any, Dict

import numpy as np

from .base import BackendResult


class PyTorchBackend:
    """Explain one raw scalar PyTorch model output per sample with Captum."""

    input_kind = "torch"

    @classmethod
    def supports(cls, model: object) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return isinstance(model, torch.nn.Module)

    def __init__(self, model: object, *, n_steps: int = 64) -> None:
        try:
            import torch
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ImportError(
                "PyTorch support requires optional dependencies. Install them "
                "with `pip install unifiedig[torch]`."
            ) from exc

        self.model = model
        self.n_steps = n_steps
        tensors = list(model.parameters()) + list(model.buffers())
        reference = next((tensor for tensor in tensors if tensor.is_floating_point()), None)
        self.device = reference.device if reference is not None else None
        self.dtype = reference.dtype if reference is not None else None
        self._integrated_gradients = IntegratedGradients(self._forward_scalar)

    def explain(self, data: Any, baseline: Any) -> BackendResult:
        module_states: Dict[Any, bool] = {
            module: module.training for module in self.model.modules()
        }
        self.model.eval()
        try:
            output_values = self._output(data)
            base_values = self._output(baseline)
            attributions = self._integrated_gradients.attribute(
                data,
                baselines=baseline,
                n_steps=self.n_steps,
                method="gausslegendre",
            )
        finally:
            for module, training in module_states.items():
                module.training = training

        return BackendResult(
            self._to_numpy(attributions),
            self._to_numpy(base_values),
            self._to_numpy(output_values),
            None,
        )

    def _forward_scalar(self, data: Any) -> Any:
        output = self.model(data)
        if output.ndim == 1:
            return output
        if output.ndim == 2 and output.shape[1] == 1:
            return output[:, 0]
        raise ValueError(
            "V1 PyTorch support requires one raw scalar output per sample; "
            "for binary classification, return the logit rather than a probability"
        )

    def _output(self, data: Any) -> Any:
        import torch

        with torch.no_grad():
            return self._forward_scalar(data)

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy()
