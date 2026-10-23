use crate::error::SplineError;
use crate::splines::{BoundaryCondition, CatmullRom, Presmoothing, SplineOptions};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Spacing of the points a smoothed line is resampled at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spacing {
    /// Exactly this many points, evenly spaced by arc length.
    Count(usize),
    /// The fewest evenly spaced points that are no farther apart than this along the curve.
    MaxDistance(f64),
}

/// A line to smooth, how to fit it, and how to resample it.
#[derive(Debug, Clone, PartialEq)]
pub struct LineToSmooth {
    pub vertices: Vec<[f64; 2]>,
    pub spacing: Spacing,
    pub options: SplineOptions,
}

/// Tangent angles in radians (`atan2(dy, dx)`) at each vertex of a centripetal spline
/// through `line`, with natural ends, or periodic when `closed`.
///
/// # Errors
///
/// Fails on the same inputs as [`CatmullRom::new`].
pub fn line_tangents(
    line: Vec<[f64; 2]>,
    presmoothing: Option<Presmoothing>,
    closed: bool,
) -> Result<Vec<f64>, SplineError> {
    let bc_type = if closed {
        BoundaryCondition::Closed
    } else {
        BoundaryCondition::Natural
    };
    let options = SplineOptions {
        alpha: Some(0.5),
        bc_type,
        presmoothing,
        ..SplineOptions::default()
    };
    let spline = CatmullRom::new(line, options)?;
    let tangent_vectors = spline.evaluate(spline.grid(), 1)?;
    Ok(tangent_vectors
        .iter()
        .map(|[dx, dy]| f64::atan2(*dy, *dx))
        .collect())
}

/// [`line_tangents`] for each line, in parallel.
///
/// # Errors
///
/// Fails when the input lengths differ or any line fails.
pub fn lines_tangents(
    lines: Vec<Vec<[f64; 2]>>,
    presmoothing: Vec<Option<Presmoothing>>,
    closed: Vec<bool>,
) -> Result<Vec<Vec<f64>>, SplineError> {
    if lines.len() != presmoothing.len() || lines.len() != closed.len() {
        return Err(SplineError::MismatchedInputLengths);
    }
    lines
        .into_par_iter()
        .zip(presmoothing)
        .zip(closed)
        .map(|((vertices, presmoothing), closed)| line_tangents(vertices, presmoothing, closed))
        .collect()
}

/// Fits a spline through `vertices` and resamples it evenly by arc length.
///
/// # Errors
///
/// Fails for a count below 2, a distance that is not finite and positive, or any input
/// [`CatmullRom::new`] rejects.
pub fn smooth_linestring(
    vertices: Vec<[f64; 2]>,
    spacing: Spacing,
    options: SplineOptions,
) -> Result<Vec<[f64; 2]>, SplineError> {
    match spacing {
        Spacing::Count(n) if n < 2 => return Err(SplineError::InvalidPointCount),
        Spacing::MaxDistance(d) if !(d.is_finite() && d > 0.0) => {
            return Err(SplineError::InvalidDistance)
        }
        _ => {}
    }
    let spline = CatmullRom::new(vertices, options)?;
    let n_pts = match spacing {
        Spacing::Count(n) => n,
        // ceil(L / d) intervals, one more point.
        Spacing::MaxDistance(d) => ((spline.length() / d).ceil() as usize).saturating_add(1),
    };
    let distances = spline.uniform_distances(n_pts)?;
    spline.evaluate(&distances, 0)
}

/// [`smooth_linestring`] for each line, in parallel.
///
/// # Errors
///
/// Fails when any line fails.
pub fn smooth_linestrings(lines: Vec<LineToSmooth>) -> Result<Vec<Vec<[f64; 2]>>, SplineError> {
    lines
        .into_par_iter()
        .map(|line| smooth_linestring(line.vertices, line.spacing, line.options))
        .collect()
}
