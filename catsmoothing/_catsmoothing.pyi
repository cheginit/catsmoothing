from __future__ import annotations
from typing import Literal

__version__: str

class CatmullRom:
    def __init__(
        self,
        vertices: list[tuple[float, float]],
        grid: list[float] | None = None,
        alpha: float | None = None,
        bc_type: Literal["natural", "closed", "clamped"] = "natural",
        gaussian_sigma: float | None = None,
    ) -> None: ...
    def evaluate(self, distances: list[float], n: int = 0) -> list[list[float]]: ...
    def uniform_distances(self, n_pts: int, tolerance: float = 1e-6, max_iterations: int = 100) -> list[list[float]]: ...


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
