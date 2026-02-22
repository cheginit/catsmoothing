from __future__ import annotations

from itertools import product
from typing import Literal

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


def test_repr():
    verts = [(0, 0), (1, 1), (2, 0), (3, 1)]
    s = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    r = repr(s)
    assert "CatmullRom(" in r
    assert "n_vertices=" in r
    assert "bc_type='clamped'" in r
    assert "alpha=0.5" in r
    assert "gaussian_sigma" not in r

    s2 = CatmullRom(verts, alpha=0.5, bc_type="natural", gaussian_sigma=1.0)
    r2 = repr(s2)
    assert "gaussian_sigma=1.0" in r2


def test_slots():
    verts = [(0, 0), (1, 1), (2, 0), (3, 1)]
    s = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    with pytest.raises(AttributeError):
        s.new_attr = 42  # type: ignore[attr-defined]


def test_uniform_distances():
    verts = [(0, 0), (1, 1), (2, 0), (3, 1)]
    s = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    dists = s.uniform_distances(10)
    assert len(dists) == 10
    assert dists[0] == pytest.approx(s.grid[0], abs=1e-5)
    assert dists[-1] == pytest.approx(s.grid[-1], abs=1e-5)
    # Check uniformity: spacing should be approximately equal
    spacings = np.diff(dists)
    assert np.std(spacings) < 0.2 * np.mean(spacings)


def test_derivatives():
    verts = [(0, 0), (1, 1), (2, 0), (3, 1)]
    s = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    dists = np.linspace(s.grid[0], s.grid[-1], 20)
    # Evaluate positions, first and second derivatives
    pos = s.evaluate(dists, n=0)
    deriv1 = s.evaluate(dists, n=1)
    deriv2 = s.evaluate(dists, n=2)
    assert pos.shape == (20, 2)
    assert deriv1.shape == (20, 2)
    assert deriv2.shape == (20, 2)
    # First derivative should be finite
    assert np.all(np.isfinite(deriv1))
    # Second derivative should be finite
    assert np.all(np.isfinite(deriv2))


def test_gaussian_sigma():
    verts = [(0, 0), (1, 1.5), (2, -0.5), (3, 1), (4, 0)]
    s_no_smooth = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    s_smooth = CatmullRom(verts, alpha=0.5, bc_type="clamped", gaussian_sigma=1.0)
    dists = np.linspace(s_no_smooth.grid[0], s_no_smooth.grid[-1], 50)
    pos_raw = s_no_smooth.evaluate(dists)
    pos_smooth = s_smooth.evaluate(dists)
    # Smoothed curve should differ from raw
    assert not np.allclose(pos_raw, pos_smooth)
    # Smoothed curve should have less curvature variation
    d2_raw = s_no_smooth.evaluate(dists, n=2)
    d2_smooth = s_smooth.evaluate(dists, n=2)
    assert np.linalg.norm(d2_smooth) <= np.linalg.norm(d2_raw)


def test_invalid_derivative_order():
    verts = [(0, 0), (1, 1), (2, 0), (3, 1)]
    s = CatmullRom(verts, alpha=0.5, bc_type="clamped")
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        s.evaluate(np.array([0.0, 1.0]), n=3)  # type: ignore[arg-type]


def test_invalid_bc_type():
    verts = [(0, 0), (1, 1), (2, 0)]
    with pytest.raises(ValueError, match="bc_type"):
        CatmullRom(verts, bc_type="invalid")  # type: ignore[arg-type]


def test_smooth_linestrings_multiple():
    line1 = shapely.LineString([(0, 0), (1, 1), (2, 0)])
    line2 = shapely.LineString([(0, 0), (1, -1), (2, 0)])
    result = cs.smooth_linestrings([line1, line2], n_pts=[20, 20])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, shapely.LineString) for r in result)


def test_smooth_linestrings_invalid_type():
    with pytest.raises(TypeError, match="LineString"):
        cs.smooth_linestrings(shapely.Point(0, 0))  # type: ignore[arg-type]


def test_smooth_polygon_invalid_type():
    with pytest.raises(TypeError, match="Polygon"):
        cs.smooth_polygon(shapely.LineString([(0, 0), (1, 1)]))  # type: ignore[arg-type]


def test_multipolygon():
    poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = shapely.Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])
    mp = shapely.MultiPolygon([poly1, poly2])
    result = cs.smooth_polygon(mp, n_pts=20)
    assert isinstance(result, shapely.MultiPolygon)
    assert len(result.geoms) == 2


def test_polygon_with_hole():
    exterior = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly_with_hole = shapely.Polygon(exterior.exterior.coords, [hole])
    result = cs.smooth_polygon(poly_with_hole, n_pts=30)
    assert isinstance(result, shapely.Polygon)
    assert shapely.get_num_interior_rings(result) >= 1


def test_tangent_angles_multiple_lines():
    line1 = shapely.LineString([(0, 0), (1, 1), (2, 0)])
    line2 = shapely.LineString([(0, 0), (1, -1), (2, 0)])
    result = cs.linestrings_tangent_angles([line1, line2])
    assert isinstance(result, list)
    assert len(result) == 2


def test_tangent_angles_with_gaussian():
    line = shapely.LineString([(0, 0), (1, 1), (2, 0), (3, 1)])
    angles_raw = cs.linestrings_tangent_angles(line)
    angles_smooth = cs.linestrings_tangent_angles(line, gaussian_sigmas=1.0)
    assert len(angles_raw) == len(angles_smooth)
    # Smoothed tangent angles should differ from raw
    assert not np.allclose(angles_raw, angles_smooth)


def test_tangent_angles_invalid_type():
    with pytest.raises(TypeError, match="LineString"):
        cs.linestrings_tangent_angles(shapely.Point(0, 0))  # type: ignore[arg-type]


def test_smooth_linestrings_bc_types():
    line = shapely.LineString([(0, 0), (1, 1), (2, 0), (3, 1)])
    for bc in ("natural", "clamped"):
        result = cs.smooth_linestrings(line, n_pts=20, bc_types=bc)
        assert isinstance(result, shapely.LineString)


def test_linear_ring():
    ring = shapely.LinearRing([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = cs.smooth_linestrings(ring, n_pts=20, bc_types="closed")
    assert isinstance(result, shapely.LinearRing)


@pytest.fixture
def verts():
    rng = np.random.default_rng(123)
    n_pts = 2000
    x = np.linspace(-3, 2.5, n_pts)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(n_pts)
    return np.c_[x, y]


@pytest.mark.benchmark
class TestBenchmark:
    @pytest.mark.parametrize(("alpha", "order"), list(product((0, 0.5, 1), (0, 1, 2))))
    def test_eval(self, verts: list[tuple[float, float]], alpha: float, order: Literal[0, 1, 2]):
        s = CatmullRom(verts, alpha=alpha, bc_type="closed")
        distances = np.linspace(0, 8, 10 * 2000, dtype=np.float64)
        s.evaluate(distances, n=order)

    @pytest.mark.parametrize("n_pts", [50, 100, 1000])
    def test_poly(self, verts: list[tuple[float, float]], n_pts: int):
        cs.smooth_polygon(shapely.Polygon(verts), n_pts=n_pts)

    @pytest.mark.parametrize(("n_pts", "gaussian_sigma"), list(product((50, 100, 1000), (None, 2))))
    def test_line(self, verts: list[tuple[float, float]], n_pts: int, gaussian_sigma: float):
        line = shapely.LineString(verts)
        cs.smooth_linestrings(line, n_pts=n_pts, gaussian_sigmas=gaussian_sigma)

    @pytest.mark.parametrize("gaussian_sigma", [None, 2])
    def test_tangents(self, verts: list[tuple[float, float]], gaussian_sigma: float):
        line = shapely.LineString(verts)
        cs.linestrings_tangent_angles(line, gaussian_sigma)
