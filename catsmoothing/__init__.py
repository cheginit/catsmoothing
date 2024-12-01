"""Top-level API for CatSmoothing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from catsmoothing.api import (
    CatmullRom,
    linestrings_tangent_angles,
    smooth_linestrings,
    smooth_polygon,
)

try:
    __version__ = version("catsmoothing")
except PackageNotFoundError:
    __version__ = "999"

__all__ = ["CatmullRom", "linestrings_tangent_angles", "smooth_linestrings", "smooth_polygon"]
