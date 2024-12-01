use crate::linestring;
use crate::{BoundaryCondition, CatmullRomRust, SplineError};
use pyo3::prelude::*;

/// Convert a Python error to a displayable string
impl From<SplineError> for PyErr {
    fn from(err: SplineError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

#[pyfunction(signature = (vertices, gaussian_sigma=None))]
fn linestrings_tangent_angles(
    vertices: Vec<Vec<[f64; 2]>>,
    gaussian_sigma: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<Vec<f64>>> {
    let tangents = linestring::lines_tangents(
        vertices.clone(),
        gaussian_sigma.unwrap_or_else(|| vec![None; vertices.len()]),
    )?;
    Ok(tangents)
}

#[pyfunction(signature = (lines, distances = None, n_pts_list = None, gaussian_sigmas = None, bc_types = None, tolerance = None, max_iterations = None))]
fn smooth_linestrings(
    lines: Vec<Vec<[f64; 2]>>,
    distances: Option<Vec<Option<f64>>>,
    n_pts_list: Option<Vec<Option<usize>>>,
    gaussian_sigmas: Option<Vec<Option<f64>>>,
    bc_types: Option<Vec<Option<String>>>, // Accept strings
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Vec<Vec<[f64; 2]>>> {
    // Convert strings to BoundaryCondition
    let bc_types_converted: Vec<Option<BoundaryCondition>> = bc_types
        .unwrap_or_else(|| vec![None; lines.len()])
        .into_iter()
        .map(|bc| {
            // Handle Option<Result> properly
            match bc {
                Some(b) => Some(match b.to_lowercase().as_str() {
                    "natural" => Ok(BoundaryCondition::Natural),
                    "closed" => Ok(BoundaryCondition::Closed),
                    "clamped" => Ok(BoundaryCondition::Clamped),
                    _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid value for bc_type. Use 'natural', 'closed', or 'clamped'.",
                    )),
                })
                .transpose(), // Converts Option<Result<T, E>> to Result<Option<T>, E>
                None => Ok(None),
            }
        })
        .collect::<Result<Vec<Option<BoundaryCondition>>, PyErr>>()?;

    let lines_len = lines.len();
    let smoothed = linestring::smooth_linestrings(
        lines,
        distances.unwrap_or_else(|| vec![None; lines_len]),
        n_pts_list.unwrap_or_else(|| vec![None; lines_len]),
        gaussian_sigmas.unwrap_or_else(|| vec![None; lines_len]),
        bc_types_converted,
        tolerance,
        max_iterations,
    )?;
    Ok(smoothed)
}

#[pyclass]
struct CatmullRom {
    inner: CatmullRomRust,
}

#[pymethods]
impl CatmullRom {
    #[new]
    #[pyo3(signature = (vertices, grid=None, alpha=None, bc_type="natural", gaussian_sigma=None))]
    fn new(
        vertices: Vec<[f64; 2]>,
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        bc_type: &str,
        gaussian_sigma: Option<f64>,
    ) -> PyResult<Self> {
        let boundary_condition = match bc_type.to_lowercase().as_str() {
            "natural" => BoundaryCondition::Natural,
            "closed" => BoundaryCondition::Closed,
            "clamped" => BoundaryCondition::Clamped,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid value for bc_type. Use 'natural', 'closed', or 'clamped'.",
                ));
            }
        };

        let spline =
            CatmullRomRust::new(vertices, grid, alpha, boundary_condition, gaussian_sigma)?;
        Ok(CatmullRom { inner: spline })
    }

    #[getter]
    fn grid(&self) -> Vec<f64> {
        self.inner.grid.clone()
    }

    #[pyo3(signature = (distances, n))]
    fn evaluate(&self, distances: Vec<f64>, n: usize) -> Vec<[f64; 2]> {
        self.inner.evaluate(&distances, n)
    }

    fn uniform_distances(&self, n_pts: usize, tolerance: f64, max_iterations: usize) -> Vec<f64> {
        self.inner
            .uniform_distances(n_pts, tolerance, max_iterations)
    }
}

#[pymodule]
fn _catsmoothing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linestrings_tangent_angles, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_linestrings, m)?)?;
    m.add_class::<CatmullRom>()?;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
