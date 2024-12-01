from __future__ import annotations
from typing import Literal

__version__: str

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
        vertices: list[tuple[float, float]],
        grid: list[float] | None = None,
        alpha: float | None = None,
        bc_type: Literal["natural", "closed", "clamped"] = "natural",
        gaussian_sigma: float | None = None,
    ) -> None: ...


def linestrings_tangent_angles(
    lines: list[list[tuple[float, float]]],
    gaussian_sigmas: list[float] | None = None,
) -> list[list[float]]: ...


def smooth_linestrings(
    lines: list[list[tuple[float, float]]],
    distances: list[float] | None = None,
    n_pts: list[int] | None = None,
    gaussian_sigmas: list[float] | None = None,
    bc_types: list[Literal["natural", "closed", "clamped"]] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[list[tuple[float, float]]]: ...
