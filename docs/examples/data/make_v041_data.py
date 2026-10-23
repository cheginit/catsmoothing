"""Record catsmoothing 0.4.1 outputs for the what's new notebook.

Run from the repository root against the released 0.4.1 wheel::

    uv run --isolated --no-project --python 3.14 --with catsmoothing==0.4.1 \
        python docs/examples/data/make_v041_data.py

0.4.1 has no wheels for Python 3.15, so any Python 3.14 interpreter works.

The notebook rebuilds every input below and checks it against the saved copy.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shapely

import catsmoothing as cs

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

OUTPUT = Path(__file__).with_name("v041.npz")


def noisy_bumps(n_lines: int, n_vertices: int, seed: int) -> NDArray[np.float64]:
    """Noisy samples of a Gaussian bump, shape ``(n_lines, n_vertices, 2)``."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 2.5, n_vertices)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal((n_lines, n_vertices))
    return np.stack([np.broadcast_to(x, y.shape), y], axis=-1)


def noisy_ring() -> NDArray[np.float64]:
    """Return a noisy circle of radius 10 with its first (seam) vertex at radius 12."""
    rng = np.random.default_rng(7)
    theta = np.linspace(0, 2 * np.pi, 61)[:-1]
    radius = 10 + 0.5 * rng.standard_normal(60)
    ring = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    ring[0] = (12.0, 0.0)
    return ring


def digest(arr: NDArray[np.float64]) -> NDArray[np.str_]:
    """SHA-256 of an array's bytes, to check a regenerated input without storing it."""
    return np.array(hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest())


def error_text(func: Callable[[], object]) -> str:
    """Name and message of what ``func`` raises; 0.4.1 panics raise a BaseException."""
    try:
        func()
    except BaseException as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}".strip()
    return "no error"


def best_times(func: Callable[[], object], repeat: int = 7) -> NDArray[np.float64]:
    """Wall times of ``repeat`` calls after one warm-up call."""
    func()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return np.array(times)


def record_curves(out: dict[str, NDArray[np.generic]]) -> None:
    """Record spacing, pre-smoothing, ring, and derivative outputs."""
    bumps = noisy_bumps(200, 100, seed=123)
    params = [cs.CatmullRom(b, alpha=0.5).uniform_distances(100) for b in bumps]
    out["bumps_digest"] = digest(bumps)
    out["spacing_params"] = np.array(params)
    out["spacing_points"] = cs.CatmullRom(bumps[0], alpha=0.5).evaluate(params[0])

    line_100 = np.array([(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)], dtype=float)
    out["line_100"] = line_100
    smoothed = cs.smooth_linestrings(shapely.LineString(line_100), distances=10)
    out["distances_points"] = shapely.get_coordinates(smoothed)

    straight = np.c_[np.linspace(0, 20, 21), np.linspace(0, 10, 21)]
    out["straight"] = straight
    spline = cs.CatmullRom(straight, alpha=0.5, gaussian_sigma=3)
    out["straight_smoothed"] = spline.evaluate(spline.grid)

    ring = noisy_ring()
    out["ring"] = ring
    spline = cs.CatmullRom(
        np.vstack([ring, ring[:1]]), alpha=0.5, bc_type="closed", gaussian_sigma=3
    )
    out["ring_smoothed"] = spline.evaluate(spline.grid)
    projected = shapely.Polygon(ring * 1000 + (500_000.0, 4_000_000.0))
    out["projected_coords"] = shapely.get_coordinates(cs.smooth_polygon(projected, n_pts=100))
    out["ring_angles"] = cs.linestrings_tangent_angles(shapely.LinearRing(ring))

    wavy = np.array([(0, 0), (1, 0.8), (2.2, 0.6), (3, 1.6), (4.4, 1.2), (5, 0)], dtype=float)
    out["wavy"] = wavy
    out["derivative_params"] = np.array([0.3, 1.5, 2.9, 4.0])
    out["derivatives"] = cs.CatmullRom(wavy, alpha=0.5).evaluate(out["derivative_params"], 1)


def record_degenerate_inputs(out: dict[str, NDArray[np.generic]]) -> None:
    """Record what 0.4.1 returns or raises for inputs it mishandles."""
    line_100, wavy = out["line_100"], out["wavy"]
    zigzag = np.array([(0, 0), (1, 1), (2, 0), (3, 1)], dtype=float)
    out["zigzag"] = zigzag
    out["offset_grid_params"] = cs.CatmullRom(zigzag, grid=[1, 2, 3, 4]).uniform_distances(5)
    out["unsorted_grid_points"] = cs.CatmullRom(zigzag, grid=[0, 2, 1, 3]).evaluate([0.5])
    short = np.c_[np.arange(5.0), 2 * np.arange(5.0)]
    out["short"] = short
    spline = cs.CatmullRom(short, alpha=0.5, gaussian_sigma=3)
    out["short_smoothed"] = spline.evaluate(spline.grid)
    out["negative_distance_error"] = np.array(
        error_text(lambda: cs.smooth_linestrings(shapely.LineString(wavy), distances=-1.0))
    )
    nan_sigma = cs.smooth_linestrings(shapely.LineString(wavy), n_pts=5, gaussian_sigmas=np.nan)
    out["nan_sigma_coords"] = shapely.get_coordinates(nan_sigma)

    repeated = shapely.LineString([(0, 0), (1, 1), (1, 1), (2, 0)])
    out["repeated_coords"] = shapely.get_coordinates(cs.smooth_linestrings(repeated, n_pts=5))
    out["two_vertex_error"] = np.array(
        error_text(lambda: cs.smooth_linestrings(shapely.LineString([(0, 0), (1, 1)]), n_pts=5))
    )
    out["n_pts_one_error"] = np.array(
        error_text(lambda: cs.smooth_linestrings(shapely.LineString(line_100), n_pts=1))
    )
    with_nan = shapely.LineString([(0, 0), (1, np.nan), (2, 0)])
    out["nan_coords"] = shapely.get_coordinates(cs.smooth_linestrings(with_nan, n_pts=5))
    out["xyz_points"] = cs.CatmullRom([(0, 0, 1), (1, 1, 2), (2, 0, 3)]).evaluate([0.5])


def record_timing(out: dict[str, NDArray[np.generic]]) -> None:
    """Record wall times for smoothing 10,000 lines and the machine they ran on."""
    lines = list(shapely.linestrings(noisy_bumps(10_000, 100, seed=2026)))
    out["timing_digest"] = digest(noisy_bumps(10_000, 100, seed=2026))
    out["smooth_times"] = best_times(lambda: cs.smooth_linestrings(lines, n_pts=[100] * len(lines)))
    out["machine"] = np.array(
        f"{platform.platform()}, {os.cpu_count()} CPUs, Python {platform.python_version()}, "
        f"numpy {np.__version__}, shapely {shapely.__version__}, catsmoothing {cs.__version__}"
    )


def main() -> None:
    if "continuity" in inspect.signature(cs.CatmullRom).parameters:
        msg = "Run this script against the 0.4.1 wheel, not a newer build."
        raise RuntimeError(msg)
    out: dict[str, NDArray[np.generic]] = {}
    record_curves(out)
    record_degenerate_inputs(out)
    record_timing(out)
    np.savez_compressed(OUTPUT, **out)


if __name__ == "__main__":
    main()
