"""Catmull-Rom spline based on [splines](https://github.com/AudioSceneDescriptionFormat/splines) library."""

# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import shapely

from catsmoothing._catsmoothing import CatmullRom as _CatmullRomRS
from catsmoothing._catsmoothing import linestrings_tangent_angles as _tangent_angles_rs
from catsmoothing._catsmoothing import smooth_linestrings as _smooth_rs

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely import LinearRing, LineString, MultiPolygon, Polygon

    FloatArray = NDArray[np.float64]
    BCType = Literal["natural", "closed", "clamped"]
    PolygonType = TypeVar("PolygonType", Polygon, MultiPolygon)

__all__ = ["CatmullRom", "linestrings_tangent_angles", "smooth_linestrings", "smooth_polygon"]


def _tolist(arr: Any, dtype: str) -> list[Any]:
    """Convert array to list."""
    return np.asarray(arr, dtype=dtype).tolist()


class CatmullRom:
    """Catmull-Rom spline generator.

    Parameters
    ----------
    vertices : array_like
        Sequence of (x, y) or (x, y, z) vertices.
    grid : array_like, optional
        Sequence of parameter values. Must be strictly increasing.
        If not specified, a uniform grid is used (0, 1, 2, 3, ...).
    alpha : float, optional
        Catmull-Rom parameter. If specified, ``grid`` is ignored.
    bc_types : {"closed", "natural", "clamped"}, optional
        Start/end conditions. If ``"closed"``, the first vertex is re-used as
        last vertex and an additional ``grid`` value has to be specified.
        If ``"clamped"``, endpoint tangents are set to ensure the spline passes
        through the start and end points without deviation.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian kernel. If specified, applies Gaussian
        smoothing to the vertices before fitting the curve. Default is None (no smoothing).

    Examples
    --------
    >>> verts = [(0., 0.), (1., 1.), (2., 0.), (3., 1.)]
    >>> s = CatmullRom(verts, alpha=0.5, bc_types="clamped")
    >>> grid = np.linspace(0, s.grid[-1], 5)
    >>> s.evaluate(grid)
    array([[0.        , 0.        ],
           [0.78125   , 0.9375    ],
           [1.5625    , 0.375     ],
           [2.34375   , 0.21875   ],
           [3.        , 1.        ]])

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
        bc_type: BCType = "natural",
        gaussian_sigma: float | None = None,
    ) -> None:
        vertices = _tolist(vertices, "f8")
        grid = _tolist(grid, "f8") if grid is not None else None
        self.alpha = float(alpha) if alpha is not None else None
        self.gaussian_sigma = float(gaussian_sigma) if gaussian_sigma is not None else None
        self.bc_type = bc_type
        if self.bc_type not in ("natural", "closed", "clamped"):
            raise ValueError("`bc_types` must be 'natural', 'closed', 'clamped'.")
        self._spline = _CatmullRomRS(vertices, grid, self.alpha, self.bc_type, self.gaussian_sigma)
        self.grid = np.asarray(self._spline.grid)
        self._is_frozen = True

    def evaluate(self, distances: list[float] | FloatArray, n: int = 0) -> FloatArray:
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
        if n not in (0, 1, 2):
            raise ValueError("`n` must be 0, 1, or 2 since Catmull-Rom spline is cubic.")
        results = self._spline.evaluate(_tolist(np.atleast_1d(distances), "f8"), int(n))
        return np.asarray(results)

    def uniform_distances(
        self, n_pts: int, tolerance: float = 1e-6, max_iterations: int = 100
    ) -> FloatArray:
        """Get uniformly spaced parameter values.

        Parameters
        ----------
        n_pts : int
            Number of points to generate.
        tolerance : float, optional
            Tolerance for uniform spacing, by default 1e-6.
        max_iterations : int, optional
            Maximum number of iterations for uniform spacing, by default 100.

        Returns
        -------
        array_like
            Uniformly spaced parameter values.
        """
        results = self._spline.uniform_distances(n_pts, tolerance, max_iterations)
        return np.asarray(results)

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


@overload
def smooth_linestrings(
    lines: LineString,
    distances: float | None = None,
    n_pts: int | None = None,
    gaussian_sigmas: float | None = None,
    bc_types: BCType | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> LineString: ...


@overload
def smooth_linestrings(  # pyright: ignore[reportOverlappingOverload]
    lines: LinearRing,
    distances: float | None = None,
    n_pts: int | None = None,
    gaussian_sigmas: float | None = None,
    bc_types: BCType | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> LinearRing: ...


@overload
def smooth_linestrings(
    lines: list[LinearRing],
    distances: list[float] | None = None,
    n_pts: list[int] | None = None,
    gaussian_sigmas: list[float] | None = None,
    bc_types: list[BCType] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[LinearRing]: ...


@overload
def smooth_linestrings(
    lines: list[LineString],
    distances: list[float] | None = None,
    n_pts: list[int] | None = None,
    gaussian_sigmas: list[float] | None = None,
    bc_types: list[BCType] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[LineString]: ...


def smooth_linestrings(
    lines: LineString | LinearRing | list[LineString] | list[LinearRing],
    distances: float | list[float] | None = None,
    n_pts: int | list[int] | None = None,
    gaussian_sigmas: float | list[float] | None = None,
    bc_types: BCType | list[BCType] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> LineString | LinearRing | list[LineString] | list[LinearRing]:
    """Smooth LineStrings using Centripetal Catmull-Rom splines and uniform spacing.

    Parameters
    ----------
    lines : list of shapely.LineString
        LineStrings to be smoothed.
    distances : list of float, optional
        Distance between two consecutive points, by default None. You must
        specify either ``distances`` or ``n_pts``.
    n_pts : list of int, optional
        Number of points to be generated, by default None. You must specify
        either ``distances`` or ``n_pts``.
    gaussian_sigmas : list of float, optional
        Standard deviation for Gaussian kernel, by default None,
        i.e., no smoothing.
    bc_types : list of {"closed", "natural", "clamped"}, optional
        Start/end conditions for each LineString. If ``"closed"``, the first
        vertex is re-used as last vertex and an additional ``distances`` value
        has to be specified. If ``"clamped"``, endpoint tangents are set to
        ensure the spline passes through the start and end points without deviation.
    tolerance : float, optional
        Tolerance for uniform spacing, by default 1e-6.
    max_iterations : int, optional
        Maximum number of iterations for uniform spacing, by default 100.

    Returns
    -------
    shapely.LineString or numpy.ndarray of shapely.LineString
        Fitted CatmullRom curve as either a LineString.
    """
    lines = np.atleast_1d(lines)  # pyright: ignore[reportCallIssue,reportArgumentType]
    if np.any(~np.isin(shapely.get_type_id(lines), [1, 2])):
        raise TypeError("`lines` must be a list shapely.LineString")

    smoothed = _smooth_rs(
        [shapely.get_coordinates(line).tolist() for line in lines],  # pyright: ignore[reportGeneralTypeIssues]
        _tolist(np.atleast_1d(distances), "f8") if distances is not None else None,
        _tolist(np.atleast_1d(n_pts), "i8") if n_pts is not None else None,
        _tolist(np.atleast_1d(gaussian_sigmas), "f8") if gaussian_sigmas is not None else None,
        _tolist(np.atleast_1d(bc_types), "U") if bc_types is not None else None,
        tolerance,
        max_iterations,
    )
    geom_type = shapely.get_type_id(lines[0])  # pyright: ignore[reportIndexIssue]
    if len(smoothed) == 1:
        return (
            shapely.LineString(smoothed[0]) if geom_type == 1 else shapely.LinearRing(smoothed[0])
        )
    return shapely.linestrings(smoothed) if geom_type == 1 else shapely.linearrings(smoothed)


@overload
def linestrings_tangent_angles(
    lines: list[LineString] | list[LinearRing],
    gaussian_sigmas: list[float] | None = None,
) -> list[FloatArray]: ...


@overload
def linestrings_tangent_angles(
    lines: LineString | LinearRing,
    gaussian_sigmas: float | None = None,
) -> FloatArray: ...


def linestrings_tangent_angles(
    lines: LineString | LinearRing | list[LineString] | list[LinearRing],
    gaussian_sigmas: float | list[float] | None = None,
) -> FloatArray | list[FloatArray]:
    """Compute tangent angles for a line.

    Parameters
    ----------
    lines : list of shapely.LineString
        LineStrings to be smoothed.
    gaussian_sigmas : list of float, optional
        Standard deviation for Gaussian kernel, by default None,
        i.e., no smoothing.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Signed tangent angles in radians.
    """
    lines = np.atleast_1d(lines)  # pyright: ignore[reportCallIssue,reportArgumentType]
    if np.any(~np.isin(shapely.get_type_id(lines), [1, 2])):
        raise TypeError("`lines` must be a list shapely.LineString")
    angles = _tangent_angles_rs(
        [shapely.get_coordinates(line).tolist() for line in lines],  # pyright: ignore[reportGeneralTypeIssues]
        _tolist(np.atleast_1d(gaussian_sigmas), "f8") if gaussian_sigmas is not None else None,
    )
    if len(angles) == 1:
        return np.asarray(angles[0])
    return [np.asarray(angle) for angle in angles]


def _poly_smooth(
    polygon: Polygon,
    distance: float | None = None,
    n_pts: int | None = None,
    gaussian_sigma: float | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Polygon:
    """Smooth a Polygon using Catmull-Rom splines."""
    exterior = smooth_linestrings(
        polygon.exterior, distance, n_pts, gaussian_sigma, "closed", tolerance, max_iterations
    )
    n_holes = shapely.get_num_interior_rings(polygon)
    if n_holes == 0:
        return shapely.Polygon(exterior)
    interiors = smooth_linestrings(  # pyright: ignore[reportCallIssue]
        polygon.interiors,  # pyright: ignore[reportArgumentType]
        [distance] * n_holes if distance is not None else None,
        [n_pts] * n_holes if n_pts is not None else None,
        [gaussian_sigma] * n_holes if gaussian_sigma is not None else None,
        ["closed"] * n_holes,  # pyright: ignore[reportArgumentType]
        tolerance,
        max_iterations,
    )
    return shapely.Polygon(exterior, interiors)


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
    if not isinstance(polygon, shapely.Polygon | shapely.MultiPolygon):
        raise TypeError("`polygon` must be a shapely.Polygon or shapely.MultiPolygon")

    if isinstance(polygon, shapely.Polygon):
        return _poly_smooth(polygon, distance, n_pts, gaussian_sigma, tolerance, max_iterations)

    return shapely.MultiPolygon(
        [
            _poly_smooth(poly, distance, n_pts, gaussian_sigma, tolerance, max_iterations)
            for poly in polygon.geoms
        ]
    )
