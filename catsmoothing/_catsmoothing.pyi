from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

__version__: str

class CatmullRom:
    @property
    def grid(self) -> FloatArray: ...
    def __init__(
        self,
        vertices: FloatArray,
        grid: FloatArray | None = None,
        alpha: float | None = None,
        bc_type: Literal["natural", "closed", "clamped"] = "natural",
        gaussian_sigma: float | None = None,
        sigma_units: Literal["vertices", "length"] = "vertices",
        taubin_iterations: int | None = None,
        continuity: Literal["c1", "g2", "c2"] = "c1",
    ) -> None: ...
    def evaluate(self, distances: FloatArray, n: Literal[0, 1, 2] = 0) -> FloatArray: ...
    def uniform_distances(self, n_pts: int) -> FloatArray: ...

def linestrings_tangent_angles(
    coords: FloatArray,
    offsets: IntArray,
    gaussian_sigmas: FloatArray | None = None,
    sigma_units: Literal["vertices", "length"] = "vertices",
    taubin_iterations: int | None = None,
    closed: NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, IntArray]: ...

def smooth_linestrings(
    coords: FloatArray,
    offsets: IntArray,
    distances: FloatArray | None = None,
    n_pts: IntArray | None = None,
    gaussian_sigmas: FloatArray | None = None,
    bc_types: list[str] | None = None,
    sigma_units: Literal["vertices", "length"] = "vertices",
    taubin_iterations: int | None = None,
    continuity: Literal["c1", "g2", "c2"] = "c1",
) -> tuple[FloatArray, IntArray]: ...
