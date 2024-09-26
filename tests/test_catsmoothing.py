from __future__ import annotations

import numpy as np
import pytest
import shapely

import catsmoothing as cs
from catsmoothing import CatmullRom


def assert_close(a: float, b: float, tol: float = 1e-3) -> None:
    assert np.allclose(a, b, atol=tol)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0, [1.5, 0.6099]),
        (0.5, [1.4570, 0.5289]),
        (1, [1.4754, 0.3844]),
    ],
)
def test_spline(alpha: float, expected: float) -> None:
    verts = [(0, 0), (0, 0.5), (1.5, 1.5), (1.6, 1.5), (3, 0.2), (3, 0)]
    n_pts = 15
    s = CatmullRom(verts, alpha=alpha, bc_types="closed")
    dots = int((s.grid[-1] - s.grid[0]) * n_pts) + 1
    distances = s.grid[0] + np.arange(dots) / n_pts
    assert_close(s.evaluate(distances).mean(axis=0), expected)


def test_poly():
    verts = [(0, 0), (0, 0.5), (1.5, 1.5), (1.6, 1.5), (3, 0.2), (3, 0)]
    poly = shapely.Polygon(verts)
    ploy_smoothed = cs.smooth_polygon(poly, n_pts=50)
    assert_close(poly.area, ploy_smoothed.area, 0.5)


def test_line_tangent():
    rng = np.random.default_rng(123)
    x = np.linspace(-3, 2.5, 50)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(50)
    line = shapely.LineString(np.c_[x, y])
    line_smoothed = cs.smooth_linestring(line, n_pts=30, gaussian_sigma=2)
    assert_close(line.length, line_smoothed.length, 2.8)

    line_smoothed = cs.smooth_linestring(line, distance=0.2, gaussian_sigma=2)
    assert_close(line.length, line_smoothed.length, 2.8)

    vertices = shapely.get_coordinates(line_smoothed)
    tangents = cs.compute_tangents(vertices)
    assert_close(tangents.mean(), 0.0072)
