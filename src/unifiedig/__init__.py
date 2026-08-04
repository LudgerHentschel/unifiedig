"""Unified Integrated Gradients public API."""

from importlib.metadata import PackageNotFoundError, version

from .explainer import Explainer
from .explanation import Explanation
from .jax import JaxModel
from .tensorflow import TensorFlowModel

__all__ = ["Explainer", "Explanation", "JaxModel", "TensorFlowModel"]

try:
    __version__ = version("unifiedig")
except PackageNotFoundError:
    __version__ = "0+unknown"
