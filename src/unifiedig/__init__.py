"""Unified Integrated Gradients public API."""

from importlib.metadata import PackageNotFoundError, version

from .explainer import Explainer
from .explanation import Explanation
from .jax import JaxModel
from .loss_explainer import LossExplainer
from .tensorflow import TensorFlowModel

__all__ = ["Explainer", "Explanation", "JaxModel", "LossExplainer", "TensorFlowModel"]

try:
    __version__ = version("unifiedig")
except PackageNotFoundError:
    __version__ = "0+unknown"
