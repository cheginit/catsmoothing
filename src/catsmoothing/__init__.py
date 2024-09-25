"""Top-level API for CatSmoothing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from catsmoothing import CatmullRom, fit_catmull_rom

try:
    __version__ = version("catsmoothing")
except PackageNotFoundError:
    __version__ = "999"

__all__ = ["CatmullRom", "fit_catmull_rom"]
