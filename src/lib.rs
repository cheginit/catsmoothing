mod error;
mod linestring;
#[cfg(feature = "python")]
mod python;
mod splines;
mod utils;

pub use error::SplineError;
pub use linestring::{line_tangents, lines_tangents, smooth_linestring, smooth_linestrings};
pub use splines::{BoundaryCondition, CatmullRom};
