//! Catmull-Rom spline smoothing of 2D polylines and polygon rings: the Rust core of the
//! `catsmoothing` Python package.

mod error;
mod linestring;
mod presmooth;
#[cfg(feature = "python")]
mod python;
mod splines;
mod utils;

pub use error::SplineError;
pub use linestring::{
    line_tangents, lines_tangents, smooth_linestring, smooth_linestrings, LineToSmooth, Spacing,
};
pub use splines::{BoundaryCondition, CatmullRom, Presmoothing, SigmaUnits, SplineOptions};
