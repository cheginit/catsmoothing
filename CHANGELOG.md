# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased] - 20XX-XX-XX

### Added

### Changed

### Fixed

## [0.4.0] - 2026-02-21

This release adds Python 3.14 support and brings significant improvements to both the
Rust core and the Python API. Key highlights include a corrected second derivative
formula, proper handling of mixed `LineString`/`LinearRing` inputs, and better error
handling by returning `Result` instead of panicking. On the Rust side, two crate
dependencies (`lazy_static` and `num_cpus`) have been removed in favor of standard
library alternatives, and all major dependencies have been bumped. The Python API gains
`__repr__` and `__slots__` on `CatmullRom`, along with cleaner data passing to Rust by
using plain lists instead of `numpy` array intermediaries.

### Added

- Python 3.14 support.
- `__repr__` method on `CatmullRom` class for better debugging output.
- `__slots__` on `CatmullRom` class for reduced memory usage and prevention of dynamic
    attribute creation.
- Length validation in `linestrings_tangent_angles` to raise an error when
    `gaussian_sigmas` length does not match the number of input lines.
- `grid` property in `_catsmoothing.pyi` type stubs.

### Fixed

- Fixed second derivative formula in spline evaluation (was constant `6.0` instead of
    correct `6.0 * t`).
- In `smooth_linestrings`, if the input was a mix of `LineString` and `LinearRing` the
    output was either a `LineString` or a `LinearRing` depending on the type of the
    first input. Now, the output correctly is a mix of `LineString` and `LinearRing`
    depending on the type of each element in the input.
- Fixed `_catsmoothing.pyi` type stubs to match the actual Rust API signatures.
- Removed outdated `scipy` requirement mention from `smooth_polygon` docstring.

### Changed

- `evaluate()` and `uniform_distances()` now return `Result` instead of panicking on
    invalid input, propagating errors to Python as `ValueError`.
- Replaced `lazy_static` with `std::sync::LazyLock` and `num_cpus` with
    `std::thread::available_parallelism()`, removing two crate dependencies.
- Enforce valid polygon generation in `smooth_polygon` using `shapely.make_valid`
    function.
- `Vec2D` now derives `Copy` for reduced heap allocations.
- Improved spline evaluation by using `partition_point` for binary search and ordered
    tangent construction instead of `insert(0, ...)`.
- Pass `bc_types` as a plain `list[str]` to Rust instead of converting through a `numpy`
    string array.
- Pass `gaussian_sigmas` as a plain `list` in `linestrings_tangent_angles` instead of
    converting through a `numpy` array.
- Bumped Rust dependencies: `ndarray` 0.16 to 0.17, `pyo3` 0.24 to 0.28, `numpy` (crate)
    to 0.28, `rand` to 0.10, `rand_distr` to 0.6.

## [0.3.1] - 2025-04-16

### Fixed

- Fix the threading issue when initializing Rayon.

## [0.3.0] - 2025-04-16

### Changed

- Refactor the Rust code base to use `ndarray` instead of `nalgebra` for matrix
    operations since the Python API uses `numpy`. This change improves the performance
    of the code as it avoids unnecessary conversions between `nalgebra` and `numpy`
    arrays.
- Added specialized `Vec2D` structure to replace generic `Array1<f64>` for 2D vectors
- Improved memory management with preallocation and reduced temporary allocations
- Optimized mathematical operations with direct 2D vector calculations
- Added `lazy_static` Hermitian matrix to avoid recreation
- Eliminated redundant vector storage and minimized cloning
- Improved Gaussian smoothing for better cache utilization
- Fixed thread safety issues for proper parallel execution

## [0.2.2] - 2024-12-06

### Changed

- Improved the code performance by creating a global thread pool that is shared across
    all functions that run in parallel. This avoids creating a new thread pool for each
    function call, which was causing a performance overhead.
- Improved the performance of the `gaussian_smoothing` function by using a more
    efficient algorithm to compute the convolution of the input data with a Gaussian
    kernel.

## [0.2.1] - 2024-12-02

### Fixed

- Fix the `smooth_linestrings` function to correctly handle returning multiple
    `LineString` geometries when a list of `LineString` geometries is passed as input.

## [0.2.0] - 2024-12-01

This is a major release with breaking changes. The codebase has been written from
scratch in Rust and the Python bindings have been updated to reflect the changes in the
underlying implementation.

### Changed

- The `bc_types` argument of `CatmullRom` class has been renamed to `bc_type` and custom
    boundary conditions have been removed. The available boundary conditions are
    `'natural'`, `'closed'`, and `'clamped'`.
- The `smooth_linstring` function has been renamed to `smooth_linestrings`. This
    function now handles a list of `LineString` geometries instead of a single geometry.
    The code runs in parallel in Rust and is significantly faster than the previous
    implementation. If a single `LineString` geometry is passed a single `LineString`
    geometry is returned, otherwise a list of `LineString` geometries is returned.
- The `compute_tangents` has been renamed to `linestrings_tangent_angles`. This function
    now handles a list of `LineString` geometries and returns the tangent angles at each
    vertex of the input geometries. The code runs in parallel in Rust and is
    significantly faster than the previous. If a single `LineString` geometry is passed
    a single array of tangent angles is returned, otherwise a list of arrays is
    returned.

## [0.1.1] - 2024-09-25

### Added

- Add a new boundary condition for the `CatmullRom` class called `'clamped'`, which
    allows anchoring the first and last points of the curve to the first and last
    control points. This is particularly useful for smoothing `MultiLineString`
    geometries where the first and last points of each line segment must remain
    connected to the previous and next line segment.

### Changed

- Freeze attributes of `CatmullRom` since they are not supposed to be changed, once the
    class is instantiated. If needed, a new instance should be created.

## [0.1.0] - 2024-08-31

- Initial release.
