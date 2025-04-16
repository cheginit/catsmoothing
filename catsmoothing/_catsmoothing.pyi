from __future__ import annotations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np

    FloatArray = NDArray[np.float64]
    IntArray = NDArray[np.int64]
    StrArray = NDArray[np.str_]


__version__: str

class CatmullRom:
    def __init__(
        self,
        vertices: FloatArray,
        grid: FloatArray | None = None,
        alpha: float | None = None,
        bc_type: Literal["natural", "closed", "clamped"] = "natural",
        gaussian_sigma: float | None = None,
    ) -> None: ...
    def evaluate(self, distances: FloatArray, n: Literal[0, 1, 2] = 0) -> FloatArray: ...
    def uniform_distances(self, n_pts: int, tolerance: float = 1e-6, max_iterations: int = 100) -> FloatArray: ...


def linestrings_tangent_angles(
    lines: list[FloatArray],
    gaussian_sigmas: FloatArray | None = None,
) -> list[FloatArray]: ...


def smooth_linestrings(
    lines: list[FloatArray],
    distances: FloatArray | None = None,
    n_pts: IntArray | None = None,
    gaussian_sigmas: FloatArray | None = None,
    bc_types: StrArray | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[FloatArray]: ...
