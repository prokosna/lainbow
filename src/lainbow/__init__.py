"""Lainbow: similar-song and natural-language search for a local music library."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("lainbow")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["__version__"]
