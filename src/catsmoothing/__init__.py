"""Top-level API for CatSmoothing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from catsmoothing.catsmoothing import (
    CatmullRom,
    compute_tangents,
    smooth_linestring,
    smooth_polygon,
)

try:
    __version__ = version("catsmoothing")
except PackageNotFoundError:
    __version__ = "999"

__all__ = ["CatmullRom", "smooth_linestring", "smooth_polygon", "compute_tangents"]
