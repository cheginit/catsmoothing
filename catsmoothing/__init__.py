"""Top-level API for CatSmoothing."""

from __future__ import annotations

from catsmoothing._catsmoothing import __version__
from catsmoothing.api import (
    CatmullRom,
    linestrings_tangent_angles,
    smooth_linestrings,
    smooth_polygon,
)

__all__ = [
    "CatmullRom",
    "__version__",
    "linestrings_tangent_angles",
    "smooth_linestrings",
    "smooth_polygon",
]
