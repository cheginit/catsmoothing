import numpy as np
import shapely

import catsmoothing as cs
from catsmoothing import CatmullRom


def parameterized(names, params):
    def decorator(func):
        func.param_names = names
        func.params = params
        return func

    return decorator


class TimeSuite:
    def setup(self, *args, **kwargs):
        rng = np.random.default_rng(123)
        n_pts = 2000
        x = np.linspace(-3, 2.5, n_pts)
        y = np.exp(-(x**2)) + 0.1 * rng.standard_normal(n_pts)
        self.verts = np.c_[x, y]
        self.distances = np.linspace(0, 8, 10 * n_pts)
        self.poly = shapely.Polygon(self.verts)
        self.line = shapely.LineString(self.verts)

    @parameterized(["alpha", "order"], [(0, 0.5, 1), (1, 2, 3)])
    def time_eval(self, alpha, order):
        s = CatmullRom(self.verts, alpha=alpha, bc_types="closed")
        s.evaluate(self.distances, n=order)

    @parameterized(["n_pts"], [(50, 100, 1000)])
    def time_poly(self, n_pts):
        cs.smooth_polygon(self.poly, n_pts=n_pts)

    @parameterized(["n_pts", "gaussian_sigma"], [(50, 100, 1000), (None, 2)])
    def time_line(self, n_pts, gaussian_sigma):
        cs.smooth_linestring(self.line, n_pts=n_pts, gaussian_sigma=gaussian_sigma)

    @parameterized(["gaussian_sigma"], [(None, 2)])
    def time_tangents(self, gaussian_sigma):
        cs.compute_tangents(self.verts, gaussian_sigma)
