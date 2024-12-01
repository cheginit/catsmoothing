"""Top-level API for CatSmoothing."""

from __future__ import annotations

from catsmoothing.api import (
    CatmullRom,
    linestrings_tangent_angles,
    smooth_linestrings,
    smooth_polygon,
)

from catsmoothing._catsmoothing import __version__

__all__ = ["CatmullRom", "linestrings_tangent_angles", "smooth_linestrings", "smooth_polygon", "__version__"]
