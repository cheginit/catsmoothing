use crate::error::SplineError;
use crate::splines::{BoundaryCondition, CatmullRom};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Tangent angles in radians (`atan2(dy, dx)`) at each vertex of a centripetal spline
/// through `line`.
///
/// # Errors
///
/// Fails on the same inputs as [`CatmullRom::new`].
pub fn line_tangents(
    line: Vec<[f64; 2]>,
    gaussian_sigma: Option<f64>,
) -> Result<Vec<f64>, SplineError> {
    let spline = CatmullRom::new(
        line,
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        gaussian_sigma,
    )?;

    let tangent_vectors = spline.evaluate(spline.grid(), 1)?;

    let tangent_angles: Vec<f64> = tangent_vectors
        .iter()
        .map(|[dx, dy]| f64::atan2(*dy, *dx))
        .collect();
    Ok(tangent_angles)
}

/// [`line_tangents`] for each line, in parallel.
///
/// # Errors
///
/// Fails when the input lengths differ or any line fails.
pub fn lines_tangents(
    lines: Vec<Vec<[f64; 2]>>,
    gaussian_sigmas: Vec<Option<f64>>,
) -> Result<Vec<Vec<f64>>, SplineError> {
    if lines.len() != gaussian_sigmas.len() {
        return Err(SplineError::MismatchedInputLengths);
    }
    lines
        .into_par_iter()
        .zip(gaussian_sigmas)
        .map(|(vertices, gaussian_sigma)| line_tangents(vertices, gaussian_sigma))
        .collect()
}

pub fn smooth_linestring(
    vertices: Vec<[f64; 2]>,
    distance: Option<f64>,
    n_pts: Option<usize>,
    gaussian_sigma: Option<f64>,
    bc_type: Option<BoundaryCondition>,
) -> Result<Vec<[f64; 2]>, SplineError> {
    if n_pts.is_none() && distance.is_none() {
        return Err(SplineError::InvalidSmoothingParameters);
    }
    if distance.is_some_and(|d| !(d.is_finite() && d > 0.0)) {
        return Err(SplineError::InvalidDistance);
    }
    if n_pts.is_some_and(|n| n < 2) {
        return Err(SplineError::InvalidPointCount);
    }
    let spline = CatmullRom::new(
        vertices,
        None,
        Some(0.5),
        bc_type.unwrap_or(BoundaryCondition::Clamped),
        gaussian_sigma,
    )?;
    let n_pts = match (n_pts, distance) {
        (Some(n_pts), _) => n_pts,
        // Even spacing no wider than `distance`: ceil(L / d) intervals, one more point.
        (None, Some(distance)) => ((spline.length() / distance).ceil() as usize).saturating_add(1),
        (None, None) => unreachable!("checked above"),
    };
    let distances = spline.uniform_distances(n_pts)?;
    let points = spline.evaluate(&distances, 0)?;
    Ok(points)
}

pub fn smooth_linestrings(
    lines: Vec<Vec<[f64; 2]>>,
    distances: Vec<Option<f64>>,
    n_pts_list: Vec<Option<usize>>,
    gaussian_sigmas: Vec<Option<f64>>,
    bc_types: Vec<Option<BoundaryCondition>>,
) -> Result<Vec<Vec<[f64; 2]>>, SplineError> {
    if lines.len() != distances.len()
        || lines.len() != n_pts_list.len()
        || lines.len() != gaussian_sigmas.len()
        || lines.len() != bc_types.len()
    {
        return Err(SplineError::MismatchedInputLengths);
    }
    lines
        .into_par_iter()
        .zip(distances)
        .zip(n_pts_list)
        .zip(gaussian_sigmas)
        .zip(bc_types)
        .map(
            |((((vertices, distance), n_pts), gaussian_sigma), bc_type)| {
                smooth_linestring(vertices, distance, n_pts, gaussian_sigma, bc_type)
            },
        )
        .collect()
}
