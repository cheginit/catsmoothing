from __future__ import annotations

import numpy as np
import pytest
import shapely
from itertools import product

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
    s = CatmullRom(verts, alpha=alpha, bc_type="closed")
    dots = int((s.grid[-1] - s.grid[0]) * n_pts) + 1
    distances = s.grid[0] + np.arange(dots) / n_pts
    assert_close(s.evaluate(distances).mean(axis=0), expected)


def test_poly():
    verts = [(0, 0), (0, 0.5), (1.5, 1.5), (1.6, 1.5), (3, 0.2), (3, 0)]
    poly = shapely.Polygon(verts)
    ploy_smoothed = cs.smooth_polygon(poly, n_pts=50)
    assert_close(poly.area, ploy_smoothed.area, 0.7)


def test_line_tangent():
    rng = np.random.default_rng(123)
    x = np.linspace(-3, 2.5, 50)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(50)
    line = shapely.LineString(np.c_[x, y])
    line_smoothed = cs.smooth_linestrings(line, n_pts=30, gaussian_sigmas=2)
    assert_close(line.length, line_smoothed.length, 2.8)

    line_smoothed = cs.smooth_linestrings(line, distances=0.2, gaussian_sigmas=2)
    assert_close(line.length, line_smoothed.length, 2.8)

    tangents = cs.linestrings_tangent_angles(line_smoothed)
    assert_close(tangents.mean(), 0.0072)


def test_frozen_attrs():
    verts = [(0, 0), (0, 0.5), (1.5, 1.5), (1.6, 1.5), (3, 0.2), (3, 0)]
    s = CatmullRom(verts, alpha=0.5, bc_type="closed")
    with pytest.raises(AttributeError):
        s.grid = np.array([0, 1, 2])
    with pytest.raises(AttributeError):
        s.bc_type = "closed"
    with pytest.raises(AttributeError):
        s.gaussian_sigma = 2
    with pytest.raises(AttributeError):
        s.alpha = 1
    with pytest.raises(AttributeError):
        del s.grid

@pytest.fixture
def verts():
    rng = np.random.default_rng(123)
    n_pts = 2000
    x = np.linspace(-3, 2.5, n_pts)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(n_pts)
    return np.c_[x, y]


@pytest.mark.benchmark
class TestBenchmark:
    @pytest.mark.parametrize(["alpha", "order"], list(product((0, 0.5, 1), (0, 1, 2))))
    def test_eval(self, verts: list[tuple[float, float]], alpha: float, order: int):
        s = CatmullRom(verts, alpha=alpha, bc_type="closed")
        distances = np.linspace(0, 8, 10 * 2000)
        s.evaluate(distances, n=order)

    @pytest.mark.parametrize("n_pts", [50, 100, 1000])
    def test_poly(self, verts: list[tuple[float, float]], n_pts: int):
        cs.smooth_polygon(shapely.Polygon(verts), n_pts=n_pts)

    @pytest.mark.parametrize(["n_pts", "gaussian_sigma"], list(product((50, 100, 1000), (None, 2))))
    def test_line(self, verts: list[tuple[float, float]], n_pts: int, gaussian_sigma: float):
        line = shapely.LineString(verts)
        cs.smooth_linestrings(line, n_pts=n_pts, gaussian_sigmas=gaussian_sigma)

    @pytest.mark.parametrize("gaussian_sigma", (None, 2))
    def test_tangents(self, verts: list[tuple[float, float]], gaussian_sigma: float):
        line = shapely.LineString(verts)
        cs.linestrings_tangent_angles(line, gaussian_sigma)