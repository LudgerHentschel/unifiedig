"""Unified Integrated Gradients public API."""

from importlib.metadata import PackageNotFoundError, version

from .explainer import Explainer
from .explanation import Explanation

__all__ = ["Explainer", "Explanation"]

try:
    __version__ = version("unifiedig")
except PackageNotFoundError:
    __version__ = "0+unknown"
