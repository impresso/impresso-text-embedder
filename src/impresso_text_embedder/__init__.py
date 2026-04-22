"""impresso-text-embedder: multilingual text embeddings for Impresso content items."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("impresso-text-embedder")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
