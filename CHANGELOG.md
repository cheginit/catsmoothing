# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[Unreleased\]

### Added

- Add a new boundary condition for the `CatmullRom` class called `'clamped'`,
    which allows anchoring the first and last points of the curve to the first
    and last control points. This is particularly useful for smoothing
    `MultiLineString` geometries where the first and last points of each
    line segment must remain connected to the previous and next line segment.

### Changed

- Freeze attributes of `CatmullRom` since they are not supposed to be changed,
    once the class is instantiated. If needed, a new instance should be created.

## \[0.1.0\] - 2024-08-31

- Initial release.
