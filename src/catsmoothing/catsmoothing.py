"""Catmull-Rom spline based on [splines](https://github.com/AudioSceneDescriptionFormat/splines) library."""

from __future__ import annotations

import functools
import importlib.util
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from shapely import LineString, MultiPolygon, Polygon, get_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    BCType = Literal["natural", "closed"]
    ArrayLike = list[tuple[float, float]] | list[tuple[float, float, float]] | FloatArray
    PolygonType = TypeVar("PolygonType", Polygon, MultiPolygon)


try:
    import opt_einsum  # pyright: ignore[reportMissingImports]

    einsum = functools.partial(opt_einsum.contract)
except ImportError:
    einsum = functools.partial(np.einsum, optimize=True)

has_scipy = importlib.util.find_spec("scipy") is not None

__all__ = ["CatmullRom", "smooth_linestring", "smooth_polygon", "compute_tangents"]


def _check_grid(
    grid: list[float] | FloatArray | None, alpha: float | None, vertices: FloatArray
) -> FloatArray:
    """Check grid values and return them."""
    if grid is not None and alpha is not None:
        raise TypeError("Only one of {grid, alpha} is allowed")

    if grid is alpha is None:
        return np.arange(len(vertices), dtype="f8")

    if alpha is not None:
        return np.insert(
            np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1) ** alpha), 0, 0
        )

    if len(vertices) != len(grid):  # pyright: ignore[reportArgumentType]
        raise ValueError("As many grid times as vertices are needed")
    return np.asarray(grid)


def _check_bc_types(
    bc_types: BCType | tuple[float, float], vertices: FloatArray, grid: FloatArray
) -> tuple[
    BCType | float, BCType | float, zip[tuple[float, float, float]], zip[tuple[float, float, float]]
]:
    """Check boundary conditions and return them."""
    if isinstance(bc_types, tuple | list) and len(bc_types) == 2:
        start, end = bc_types
    elif bc_types == "closed":
        vertices = np.concatenate([vertices, vertices[1:2]])
        grid = np.concatenate([grid, [grid[-1] + grid[1] - grid[0]]])
        start = end = bc_types
    elif bc_types == "natural":
        start = end = bc_types
    else:
        raise TypeError(
            "bc_types must be a string (closed or natural) or a pair of floats (start, end)"
        )

    v_triples = zip(vertices, vertices[1:], vertices[2:], strict=False)
    g_triples = zip(grid, grid[1:], grid[2:], strict=False)
    return start, end, v_triples, g_triples


def _calculate_tangent(
    points: tuple[float, float, float], times: tuple[float, float, float]
) -> float:
    """Calculate the tangent at a point."""
    x_1, x0, x1 = points
    t_1, t0, t1 = times
    delta_1 = t0 - t_1
    delta0 = t1 - t0
    v_1 = (x0 - x_1) / delta_1
    v0 = (x1 - x0) / delta0
    return (delta0 * v_1 + delta_1 * v0) / (delta0 + delta_1)


def _end_tangent(
    bc_type: Literal["natural"] | float,
    vertices: FloatArray,
    times: FloatArray,
    other_tangent: float,
) -> float:
    """Calculate tangent at the end points."""
    if bc_type == "natural":
        x0, x1 = vertices
        t0, t1 = times
        delta = t1 - t0
        return 3 * (x1 - x0) / (2 * delta) - other_tangent * 0.5
    return bc_type


def _check_tangents(
    vertices: FloatArray, grid: FloatArray, bc_types: BCType | tuple[float, float]
) -> FloatArray:
    """Check tangents and return them."""
    start, end, v_triples, g_triples = _check_bc_types(bc_types, vertices, grid)
    # Compute tangents and then duplicate them since incoming and outgoing are the same
    tangents = [
        tangent
        for p, t in zip(v_triples, g_triples, strict=False)
        for tangent in [_calculate_tangent(p, t)] * 2
    ]
    if len(tangents) < 2:
        # straight line
        if not (len(vertices) == len(grid) == 2):
            raise ValueError("Exactly 2 vertices are needed for a straight line")
        vertices = np.asarray(vertices)
        tangents = [(vertices[1] - vertices[0]) / (grid[1] - grid[0])] * 2
    if start == end == "closed":
        # Move last (outgoing) tangent to the beginning
        tangents = tangents[-1:] + tangents[:-1]
    else:
        tangents.insert(0, _end_tangent(start, vertices[:2], grid[:2], tangents[0]))  # pyright: ignore[reportArgumentType]
        tangents.append(_end_tangent(end, vertices[-2:], grid[-2:], tangents[-1]))  # pyright: ignore[reportArgumentType]

    if len(tangents) != 2 * (len(vertices) - 1):
        raise ValueError("Exactly 2 tangents per segment are needed")
    return np.asarray(tangents)


class CatmullRom:
    """Catmull-Rom spline based on Catmull and Rom 1974 [1].

    Parameters
    ----------
    vertices : array_like
        Sequence of xy or xyz vertices.
    grid : array_like, optional
        Sequence of parameter values. Must be strictly increasing.
        If not specified, a uniform grid is used (0, 1, 2, 3, ...).
    alpha : float, optional
        Catmull-Rom parameter. If specified, ``grid`` is ignored.
    bc_types : {'closed', 'natural', (start_tangent, end_tangent)}, optional
        Start/end conditions. If 'closed', the first vertex is re-used as
        last vertex and an additional ``grid`` value has to be specified.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian kernel. If specified, applies Gaussian
        smoothing to the vertices before fitting the curve. Default is None (no smoothing).

    Examples
    --------
    >>> verts = [(0., 0.), (1., 1.), (2., 0.), (3., 1.)]
    >>> s = CatmullRom(verts, alpha=0.5)
    >>> grid = np.linspace(0, s.grid[-1], 5)
    >>> s.evaluate(grid)
    array([[0. , 0. ],
           [0.78226261, 0.93404706],
           [1.57197579, 0.39074521],
           [2.3194772 , 0.20959155],
           [3. , 1.5 ]])

    References
    ----------
    [1] E. Catmull and R. Rom, "A Class of Local Interpolating Splines,"
        in Computer Aided Geometric Design, 1974.
        [DOI: 10.1016/B978-0-12-079050-0.50020-5](https://doi.org/10.1016/B978-0-12-079050-0.50020-5)
    """

    def __init__(
        self,
        vertices: list[tuple[float, float]] | list[tuple[float, float, float]] | FloatArray,
        grid: list[float] | FloatArray | None = None,
        alpha: float | None = None,
        bc_types: BCType | tuple[float, float] = "natural",
        gaussian_sigma: float | None = None,
    ) -> None:
        self.vertices = np.asarray(vertices)
        self.gaussian_sigma = gaussian_sigma
        if len(self.vertices) < 2:
            raise ValueError("At least two vertices are required")
        self.bc_types = bc_types
        if self.bc_types == "closed":
            self.vertices = np.concatenate([self.vertices, self.vertices[:1]])

        if gaussian_sigma is not None:
            if not has_scipy:
                raise ImportError("`scipy` is required for Gaussian smoothing")

            from scipy.ndimage import gaussian_filter1d

            original_start = self.vertices[0].copy()
            original_end = self.vertices[-1].copy()
            self.vertices = gaussian_filter1d(self.vertices, sigma=gaussian_sigma, axis=0)
            self.vertices[0] = original_start
            self.vertices[-1] = original_end
        self.alpha = alpha
        self.grid = _check_grid(grid, self.alpha, self.vertices)

        tangents = _check_tangents(self.vertices, self.grid, self.bc_types)

        matrix = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
        x0 = self.vertices[:-1]
        x1 = self.vertices[1:]
        t0 = self.grid[:-1]
        t1 = self.grid[1:]
        v0 = tangents[::2]
        v1 = tangents[1::2]
        dt = t1 - t0
        segment_data = np.stack([x0, x1, dt[:, np.newaxis] * v0, dt[:, np.newaxis] * v1], axis=-1)
        self.segments = einsum("ij,klj->kil", matrix, segment_data)
        self._is_frozen = True

    def evaluate(self, distances: list[tuple[float, float]] | FloatArray, n: int = 0) -> FloatArray:
        """Get value (or n-th derivative) at given parameter value(s).

        Parameters
        ----------
        distances : array_like
            Distance(s) along the curve to evaluate.
        n : int, optional
            Order of derivative, by default 0, i.e., the value itself.

        Returns
        -------
        array_like
            Value (or n-th derivative) at given parameter value(s).
        """
        distances = np.atleast_1d(distances)
        values = np.empty((distances.shape[0], self.vertices.shape[1]))

        idxs = np.searchsorted(self.grid, distances, side="right") - 1
        idxs = np.clip(idxs, 0, len(self.grid) - 2)

        d0, d1 = self.grid[idxs], self.grid[idxs + 1]
        norm = d1 - d0
        t_normalized = (distances - d0) / norm

        seg_slice = np.s_[: -n or None]
        coefficients = np.array([self.segments[i][seg_slice] for i in idxs])
        powers = np.arange(coefficients.shape[1])[::-1]
        weights = np.prod([powers + 1 + i for i in range(n)]) / np.power(norm[:, np.newaxis], n)
        values = einsum(
            "ij,ijk->ik",
            np.power(t_normalized[:, np.newaxis], powers) * weights,
            coefficients,
        )
        return values

    def __setattr__(self, key: str, value: Any):
        """Prevent modification of attributes after initialization."""
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot modify attribute '{key}'. The object is frozen. Reinstantiate the class."
            )
        object.__setattr__(self, key, value)

    def __delattr__(self, item: str):
        """Prevent modification of attributes after initialization."""
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot delete attribute '{item}'. The object is frozen. Reinstantiate the class."
            )
        object.__delattr__(self, item)


def _uniform_distances(
    catmull: CatmullRom, n_pts: int, tolerance: float, max_iterations: int
) -> FloatArray:
    """Compute uniform distances for given number of points."""
    dis_arr = np.linspace(0, catmull.grid[-1], n_pts)
    for _ in range(max_iterations):
        points = catmull.evaluate(dis_arr)
        diff = np.diff(points, axis=0)
        arc_lengths = np.concatenate(([0], np.cumsum(np.linalg.norm(diff, axis=1))))
        total_length = arc_lengths[-1]
        uniform_lengths = np.linspace(0, total_length, n_pts)

        error = np.abs(arc_lengths - uniform_lengths)
        if np.max(error) < tolerance:
            break

        dis_arr = np.interp(uniform_lengths, arc_lengths, dis_arr)
    return dis_arr


def smooth_linestring(
    line: LineString,
    distance: float | None = None,
    n_pts: int | None = None,
    gaussian_sigma: float | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> LineString:
    """Smooth a LineString using Centripetal Catmull-Rom splines with uniform spacing.

    Parameters
    ----------
    line : shapely.LineString or numpy.ndarray
        Line to be fitted as either a LineString (Z) or a 2/3D ``numpy`` array.
    distance : float, optional
        Distance between two consecutive points, by default None. You must
        specify either ``distance`` or ``n_pts``.
    n_pts : int, optional
        Number of points to be generated, by default None. You must specify
        either ``distance`` or ``n_pts``.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian kernel, by default ``None``,
        i.e., no smoothing. Note that if specified, ``scipy`` is required.
    tolerance : float, optional
        Tolerance for uniform spacing, by default 1e-6.
    max_iterations : int, optional
        Maximum number of iterations for uniform spacing, by default 100.

    Returns
    -------
    shapely.LineString or numpy.ndarray
        Fitted CatmullRom curve as either a LineString (Z) or a 2/3D ``numpy`` array.
    """
    if not isinstance(line, LineString):
        raise TypeError("`line` must be a shapely.LineString")

    vertices = get_coordinates(line, include_z=line.has_z)
    catmull = CatmullRom(vertices, alpha=0.5, gaussian_sigma=gaussian_sigma)

    if distance is not None and n_pts is None:
        length = np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
        n_pts = int(np.ceil(length / distance))
    elif n_pts is None:
        raise ValueError("You must specify either `distance` or `n_pts`")

    dis_arr = _uniform_distances(catmull, n_pts, tolerance, max_iterations)
    return LineString(catmull.evaluate(dis_arr))


def compute_tangents(
    vertices: ArrayLike,
    gaussian_sigma: float | None = None,
) -> FloatArray:
    """Compute tangent angles for a line.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertices of a line to compute tangent angles for.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian kernel, by default ``None``,
        i.e., no smoothing. Note that if specified, ``scipy`` is required.

    Returns
    -------
    np.ndarray
        Signed tangent angles in radians.
    """
    vertices = np.asarray(vertices)
    if vertices.shape[1] not in (2, 3):
        raise TypeError("`vertices` must be 2 or 3D array.")

    catmull = CatmullRom(vertices, alpha=0.5, gaussian_sigma=gaussian_sigma)
    tangents = catmull.evaluate(catmull.grid, n=1)
    return np.arctan2(tangents[:, 1], tangents[:, 0])


def _poly_smooth(
    polygon: Polygon,
    distance: float | None = None,
    n_pts: int | None = None,
    gaussian_sigma: float | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Polygon:
    """Smooth a Polygon using Catmull-Rom splines."""
    exterior = smooth_linestring(
        polygon.exterior, distance, n_pts, gaussian_sigma, tolerance, max_iterations
    )
    interiors = [
        smooth_linestring(interior, distance, n_pts, gaussian_sigma, tolerance, max_iterations)
        for interior in polygon.interiors
    ]
    return Polygon(exterior, interiors)


def smooth_polygon(
    polygon: PolygonType,
    distance: float | None = None,
    n_pts: int | None = None,
    gaussian_sigma: float | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> PolygonType:
    """Smooth a (Multi)Polygon using Centripetal Catmull-Rom splines with uniform spacing.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        (Multi)Polygon to be smoothed.
    distance : float, optional
        Distance between two consecutive points, by default None. You must
        specify either ``distance`` or ``n_pts``.
    n_pts : int, optional
        Number of points to be generated, by default None. You must specify
        either ``distance`` or ``n_pts``.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian kernel, by default ``None``,
        i.e., no smoothing. Note that if specified, ``scipy`` is required.
    tolerance : float, optional
        Tolerance for uniform spacing, by default 1e-6.
    max_iterations : int, optional
        Maximum number of iterations for uniform spacing, by default 100.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        Smoothed (Multi)Polygon.
    """
    if not isinstance(polygon, Polygon | MultiPolygon):
        raise TypeError("`polygon` must be a shapely.Polygon or shapely.MultiPolygon")

    if isinstance(polygon, Polygon):
        return _poly_smooth(polygon, distance, n_pts, gaussian_sigma, tolerance, max_iterations)

    return MultiPolygon(
        [
            _poly_smooth(poly, distance, n_pts, gaussian_sigma, tolerance, max_iterations)
            for poly in polygon.geoms
        ]
    )
