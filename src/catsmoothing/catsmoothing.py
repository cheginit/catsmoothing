"""Catmull--Rom spline based on `splines <https://github.com/AudioSceneDescriptionFormat/splines>`__."""

from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import shapely
from shapely import LineString


if TYPE_CHECKING:
    FloatArray = npt.NDArray[np.float64]
    BCType = Literal["natural", "closed"]
    LineType = TypeVar("LineType", LineString, FloatArray)


__all__ = ["CatmullRom", "fit_catmull_rom"]


def _check_grid(
    grid: list[float] | FloatArray | None, alpha: float | None, vertices: FloatArray
) -> FloatArray:
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
    if bc_type == "natural":
        x0, x1 = vertices
        t0, t1 = times
        delta = t1 - t0
        return 3 * (x1 - x0) / (2 * delta) - other_tangent * 0.5
    return bc_type


def _check_tangents(
    vertices: FloatArray, grid: FloatArray, bc_types: BCType | tuple[float, float]
) -> FloatArray:
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
    """Catmull--Rom spline based on :cite:t:`catmull1974splines`.

    Parameters
    ----------
    vertices : array_like
        Sequence of xy or xyz vertices.
    grid : array_like, optional
        Sequence of parameter values. Must be strictly increasing.
        If not specified, a uniform grid is used (0, 1, 2, 3, ...).
    alpha : float, optional
        Catmull--Rom parameter. If specified, ``grid`` is ignored.
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
    array([[0.        , 0.        ],
        [0.78226261, 0.93404706],
        [1.57197579, 0.39074521],
        [2.3194772 , 0.20959155],
        [3.        , 1.5       ]])
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
        if bc_types == "closed":
            self.vertices = np.concatenate([self.vertices, self.vertices[:1]])

        if gaussian_sigma is not None:
            try:
                from scipy.ndimage import gaussian_filter1d
            except ImportError:
                raise ImportError("`scipy` is required for Gaussian smoothing")
            original_start = self.vertices[0].copy()
            original_end = self.vertices[-1].copy()
            self.vertices = gaussian_filter1d(self.vertices, sigma=gaussian_sigma, axis=0)
            self.vertices[0] = original_start
            self.vertices[-1] = original_end
        self.grid = _check_grid(grid, alpha, self.vertices)

        tangents = _check_tangents(self.vertices, self.grid, bc_types)

        matrix = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
        x0 = self.vertices[:-1]
        x1 = self.vertices[1:]
        t0 = self.grid[:-1]
        t1 = self.grid[1:]
        v0 = tangents[::2]
        v1 = tangents[1::2]
        dt = t1 - t0
        segment_data = np.stack([x0, x1, dt[:, np.newaxis] * v0, dt[:, np.newaxis] * v1], axis=-1)
        self.segments = np.einsum("ij,klj->kil", matrix, segment_data, optimize=True)

    def evaluate(self, distances: list[tuple[float, float]] | FloatArray, n: int = 0) -> FloatArray:
        """Get value (or n-th derivative) at given parameter value(s).

        Parameters
        ----------
        times : array_like
            Parameter value(s) to evaluate the spline at.
        n : int, optional
            Order of derivative, by default 0.

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
        values = np.einsum(
            "ij,ijk->ik", np.power(t_normalized[:, np.newaxis], powers) * weights, coefficients, optimize=True
        )
        return values


def _uniform_distances(
    catmull: CatmullRom, n_pts: int, tolerance: float, max_iterations: int
) -> FloatArray:
    """Compute uniform distances for given number of points."""
    # Initial guess
    dis_arr = np.linspace(0, catmull.grid[-1], n_pts)
    for _ in range(max_iterations):
        points = catmull.evaluate(dis_arr)
        diff = np.diff(points, axis=0)
        arc_lengths = np.concatenate(([0], np.cumsum(np.linalg.norm(diff, axis=1))))
        total_length = arc_lengths[-1]
        uniform_lengths = np.linspace(0, total_length, n_pts)

        # Compute error
        error = np.abs(arc_lengths - uniform_lengths)
        if np.max(error) < tolerance:
            break

        # Update t values
        dis_arr = np.interp(uniform_lengths, arc_lengths, dis_arr)
    return dis_arr


@overload
def fit_catmull_rom(
    line: LineType,
    distance: float | None = ...,
    n_pts: int | None = ...,
    gaussian_sigma: float | None = ...,
    return_tangent: Literal[False] = False,
) -> LineType: ...


@overload
def fit_catmull_rom(
    line: LineType,
    distance: float | None = ...,
    n_pts: int | None = ...,
    gaussian_sigma: float | None = ...,
    return_tangent: Literal[True] = True,
) -> tuple[LineType, FloatArray]: ...


def fit_catmull_rom(
    line: LineType,
    distance: float | None = None,
    n_pts: int | None = None,
    gaussian_sigma: float | None = None,
    return_tangent: bool = False,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> LineType | tuple[LineType, FloatArray]:
    """Fit a CatmullRom curve and compute its tangent angle with uniform spacing.

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
        Standard deviation for Gaussian kernel, by default None, i.e., no smoothing.
    return_tangent : bool, optional
        Whether to return tangent angle, by default False.
    tolerance : float, optional
        Tolerance for uniform spacing, by default 1e-6.
    max_iterations : int, optional
        Maximum number of iterations for uniform spacing, by default 100.

    Returns
    -------
    line : shapely.LineString or numpy.ndarray
        Fitted CatmullRom curve as either a LineString (Z) or a 2/3D ``numpy`` array.
    phi : np.ndarray
        Signed tangent angles in radians, if ``return_tangent`` is True.
    """
    if not isinstance(line, LineString | np.ndarray | list | tuple):
        raise TypeError("line must be either shapely.LineString or numpy.ndarray")

    if isinstance(line, LineString):
        if line.has_z:
            vertices = shapely.get_coordinates(line, include_z=True)
        else:
            vertices = shapely.get_coordinates(line)
    else:
        vertices = np.array(line)
        if vertices.shape[1] not in (2, 3):
            raise ValueError("Input `line` must be 2D or 3D")

    catmull = CatmullRom(vertices, alpha=0.5, gaussian_sigma=gaussian_sigma)

    if distance is not None and n_pts is None:
        length = np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
        n_pts = int(np.ceil(length / distance))
    elif n_pts is None:
        raise ValueError("You must specify either `distance` or `n_pts`")

    dis_arr = _uniform_distances(catmull, n_pts, tolerance, max_iterations)
    points = catmull.evaluate(dis_arr)

    line_fitted = LineString(points) if isinstance(line, LineString) else points
    if not return_tangent:
        return line_fitted  # pyright: ignore[reportReturnType]

    # Compute tangent angles
    tangents = catmull.evaluate(dis_arr, n=1)
    phi = np.arctan2(tangents[:, 1], tangents[:, 0])

    return line_fitted, phi  # pyright: ignore[reportReturnType]
