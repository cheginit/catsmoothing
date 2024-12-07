# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

## [0.2.2]

### Changed

- Improved the code performance by creating a global thread pool that is
    shared across all functions that run in parallel. This avoids creating
    a new thread pool for each function call, which was causing a performance
    overhead.
- Improved the performance of the `gaussian_smoothing` function by using
    a more efficient algorithm to compute the convolution of the input
    data with a Gaussian kernel.

## [0.2.1]

### Fixed

- Fix the `smooth_linestrings` function to correctly handle returning
    multiple `LineString` geometries when a list of `LineString` geometries
    is passed as input.

## [0.2.0] - 2024-12-01

This is a major release with breaking changes. The codebase has been
written from scratch in Rust and the Python bindings have been updated
to reflect the changes in the underlying implementation.

### Changed

- The `bc_types` argument of `CatmullRom` class has been renamed to
    `bc_type` and custom boundary conditions have been removed. The
    available boundary conditions are `'natural'`, `'closed'`, and
    `'clamped'`.
- The `smooth_linstring` function has been renamed to `smooth_linestrings`.
    This function now handles a list of `LineString` geometries instead of a
    single geometry. The code runs in parallel in Rust and is significantly
    faster than the previous implementation. If a single `LineString` geometry is passed a single `LineString` geometry is returned, otherwise a list of `LineString` geometries is returned.
- The `compute_tangents` has been renamed to `linestrings_tangent_angles`.
    This function now handles a list of `LineString` geometries and returns
    the tangent angles at each vertex of the input geometries. The code runs
    in parallel in Rust and is significantly faster than the previous. If a single `LineString` geometry is passed a single array of tangent angles is returned, otherwise a list of arrays is returned.

## [0.1.1] - 2024-09-25

### Added

- Add a new boundary condition for the `CatmullRom` class called `'clamped'`,
    which allows anchoring the first and last points of the curve to the first
    and last control points. This is particularly useful for smoothing
    `MultiLineString` geometries where the first and last points of each
    line segment must remain connected to the previous and next line segment.

### Changed

- Freeze attributes of `CatmullRom` since they are not supposed to be changed,
    once the class is instantiated. If needed, a new instance should be created.

## [0.1.0] - 2024-08-31

- Initial release.
