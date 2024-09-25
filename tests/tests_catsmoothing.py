import catsmoothing as cs
import numpy as np
import pytest
import shapely
from catsmoothing import CatmullRom

def assert_close(a, b):
    assert np.allclose(a, b, atol=1e-3)

@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0, [1.3818, 0.4]),
        (0.5, 1.3776, 0.4555),
        (1, [1.4791 , 0.3825]),
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
    assert_close(poly.area, ploy_smoothed.area)

def test_line():
    rng = np.random.default_rng(123)
    x = np.linspace(-3, 2.5, 50)
    y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(50)
    line = shapely.LineString(np.c_[x, y])
    line_smoothed = cs.smooth_linestring(line, n_pts=30, gaussian_sigma=2)
    assert_close(line.length, line_smoothed.length)
