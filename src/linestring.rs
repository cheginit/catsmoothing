use crate::error::SplineError;
use crate::spline::{BoundaryCondition, CatmullRomRust};
use rayon::prelude::*;

pub fn line_tangents(
    line: Vec<[f64; 2]>,
    gaussian_sigma: Option<f64>,
) -> Result<Vec<f64>, SplineError> {
    let spline = CatmullRomRust::new(
        line,
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        gaussian_sigma,
    )?;

    // Get tangent vectors by evaluating first derivative (n=1)
    let tangent_vectors = spline.evaluate(&spline.grid, 1);

    // Convert tangent vectors to angles using atan2
    let tangent_angles: Vec<f64> = tangent_vectors
        .iter()
        .map(|[dx, dy]| f64::atan2(*dy, *dx)) // Changed order to match Python's np.arctan2(dy, dx)
        .collect();
    Ok(tangent_angles)
}

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
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<Vec<[f64; 2]>, SplineError> {
    let spline = CatmullRomRust::new(
        vertices,
        None,
        Some(0.5),
        bc_type.unwrap_or(BoundaryCondition::Clamped),
        gaussian_sigma,
    )?;
    if n_pts.is_none() && distance.is_none() {
        return Err(SplineError::InvalidSmoothingParameters);
    }

    let mut n_pts_ = n_pts.unwrap_or(0);
    if n_pts_ == 0 {
        let total_length = spline.grid[spline.grid.len() - 1];
        n_pts_ = (total_length / distance.unwrap()).ceil() as usize;
    }
    let distances = spline.uniform_distances(
        n_pts_,
        tolerance.unwrap_or(1e-6),
        max_iterations.unwrap_or(100),
    );

    let points = spline.evaluate(&distances, 0);

    Ok(points)
}

pub fn smooth_linestrings(
    lines: Vec<Vec<[f64; 2]>>,
    distances: Vec<Option<f64>>,
    n_pts_list: Vec<Option<usize>>,
    gaussian_sigmas: Vec<Option<f64>>,
    bc_types: Vec<Option<BoundaryCondition>>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
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
                smooth_linestring(
                    vertices,
                    distance,
                    n_pts,
                    gaussian_sigma,
                    bc_type,
                    tolerance,
                    max_iterations,
                )
            },
        )
        .collect()
}
