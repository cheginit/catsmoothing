use crate::{linestring, BoundaryCondition, CatmullRom, SplineError};
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Convert SplineError into PyErr.
impl From<SplineError> for PyErr {
    fn from(err: SplineError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

// Helper: convert a 2D ndarray view (shape (n,2)) into a Vec<[f64; 2]>.
fn array2_to_vec(line: ArrayView2<f64>) -> Vec<[f64; 2]> {
    line.outer_iter().map(|row| [row[0], row[1]]).collect()
}

// Helper: convert Vec<[f64; 2]> points to flattened Vec<f64> for ndarray
fn points_to_flat(points: Vec<[f64; 2]>) -> (usize, Vec<f64>) {
    let n = points.len();
    let flat: Vec<f64> = points.into_iter().flat_map(|pt| [pt[0], pt[1]]).collect();
    (n, flat)
}

// Helper: validate array length matches expected count, returning PyErr if not
fn validate_array_length(arr_len: usize, expected: usize, name: &str) -> PyResult<()> {
    if arr_len != expected {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length of {} must match number of lines",
            name
        )));
    }
    Ok(())
}

/// Python function: linestrings_tangent_angles
///
/// Parameters:
///   - vertices: list of NumPy arrays (each shape (n,2))
///   - gaussian_sigmas: optional list of Option<f64> (one per linestring)
///
/// Returns:
///   - list of 1D NumPy arrays of tangent angles.
#[pyfunction(signature = (vertices, gaussian_sigmas=None))]
fn linestrings_tangent_angles(
    py: Python,
    vertices: Vec<PyReadonlyArray2<f64>>,
    gaussian_sigmas: Option<Vec<Option<f64>>>,
) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    let n_lines = vertices.len();
    let gs = gaussian_sigmas.unwrap_or_else(|| vec![None; n_lines]);
    validate_array_length(gs.len(), n_lines, "gaussian_sigmas")?;

    let mut results = Vec::with_capacity(n_lines);
    for (line_arr, sigma) in vertices.into_iter().zip(gs.into_iter()) {
        let line_vec = array2_to_vec(line_arr.as_array());

        let angles = linestring::line_tangents(line_vec, sigma).map_err(PyErr::from)?;

        results.push(Array1::from(angles).into_pyarray(py).to_owned().into());
    }

    Ok(results)
}

/// Python function: smooth_linestrings
///
/// Parameters:
///   - lines: list of NumPy arrays (each shape (n,2))
///   - distances: optional 1D NumPy array (one per linestring)
///   - n_pts: optional 1D NumPy array of usize (one per linestring)
///   - gaussian_sigmas: optional 1D NumPy array (one per linestring)
///   - bc_types: optional Vec<String> (one per linestring; "natural", "closed", or "clamped")
///   - tolerance: float (default 1e-6)
///   - max_iterations: int (default 100)
///
/// Returns:
///   - list of NumPy arrays (each shape (m,2))
#[pyfunction(
    signature = (
        lines,
        distances=None,
        n_pts=None,
        gaussian_sigmas=None,
        bc_types=None,
        tolerance=1e-6,
        max_iterations=100
    )
)]
fn smooth_linestrings(
    py: Python,
    lines: Vec<PyReadonlyArray2<f64>>,
    distances: Option<PyReadonlyArray1<f64>>,
    n_pts: Option<PyReadonlyArray1<i64>>,
    gaussian_sigmas: Option<PyReadonlyArray1<f64>>,
    bc_types: Option<Vec<String>>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    let n_lines = lines.len();
    let lines_vec: Vec<Vec<[f64; 2]>> = lines
        .into_iter()
        .map(|line_arr| array2_to_vec(line_arr.as_array()))
        .collect();

    let ds: Vec<Option<f64>> = if let Some(arr) = distances {
        let array_view = arr.as_array();
        let slice = array_view.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid distances array")
        })?;
        validate_array_length(slice.len(), n_lines, "distances")?;
        slice.iter().map(|&x| Some(x)).collect()
    } else {
        vec![None; n_lines]
    };

    let n_pts_vec: Vec<Option<usize>> = if let Some(arr) = n_pts {
        let array_view = arr.as_array();
        let slice = array_view.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid n_pts array")
        })?;
        validate_array_length(slice.len(), n_lines, "n_pts")?;
        slice.iter().map(|&x| Some(x as usize)).collect()
    } else {
        vec![None; n_lines]
    };

    let gs: Vec<Option<f64>> = if let Some(arr) = gaussian_sigmas {
        let array_view = arr.as_array();
        let slice = array_view.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid gaussian_sigmas array")
        })?;
        validate_array_length(slice.len(), n_lines, "gaussian_sigmas")?;
        slice.iter().map(|&x| Some(x)).collect()
    } else {
        vec![None; n_lines]
    };

    let bc_conv: Vec<Option<BoundaryCondition>> = if let Some(bcs) = bc_types {
        validate_array_length(bcs.len(), n_lines, "bc_types")?;
        bcs.into_iter()
            .map(|s| match s.to_lowercase().as_str() {
                "natural" => Ok(Some(BoundaryCondition::Natural)),
                "closed" => Ok(Some(BoundaryCondition::Closed)),
                "clamped" => Ok(Some(BoundaryCondition::Clamped)),
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid bc_type. Use 'natural', 'closed', or 'clamped'.",
                )),
            })
            .collect::<Result<_, _>>()?
    } else {
        vec![None; n_lines]
    };

    let smoothed_vec = linestring::smooth_linestrings(
        lines_vec,
        ds,
        n_pts_vec,
        gs,
        bc_conv,
        Some(tolerance),
        Some(max_iterations),
    )
    .map_err(PyErr::from)?;

    let result: Vec<Py<PyArray2<f64>>> = smoothed_vec
        .into_iter()
        .map(|line| {
            let (n, flat) = points_to_flat(line);
            Array2::from_shape_vec((n, 2), flat)
                .unwrap()
                .into_pyarray(py)
                .to_owned()
                .into()
        })
        .collect();

    Ok(result)
}

/// Python class: CatmullRom
///
/// API:
///   CatmullRom(vertices: NDArray[float64], grid: NDArray[float64] | None = None, alpha: float | None = None, bc_type: Literal["natural", "closed", "clamped"] = "natural", gaussian_sigma: float | None = None)
///   Methods:
///     evaluate(distances: NDArray[float64], n: 0|1|2 = 0) -> NDArray[float64]
///     uniform_distances(n_pts: int, tolerance: float = 1e-6, max_iterations: int = 100) -> NDArray[float64]
#[pyclass(name = "CatmullRom")]
struct CatmullRomWrapper {
    inner: CatmullRom,
}

#[pymethods]
impl CatmullRomWrapper {
    #[new]
    #[pyo3(signature = (vertices, grid=None, alpha=None, bc_type="natural", gaussian_sigma=None))]
    fn new(
        vertices: PyReadonlyArray2<f64>,
        grid: Option<PyReadonlyArray1<f64>>,
        alpha: Option<f64>,
        bc_type: &str,
        gaussian_sigma: Option<f64>,
    ) -> PyResult<Self> {
        let verts: Vec<[f64; 2]> = array2_to_vec(vertices.as_array());
        let grid_opt: Option<Vec<f64>> = grid.map(|g| g.as_array().to_vec());
        let bc = match bc_type.to_lowercase().as_str() {
            "natural" => BoundaryCondition::Natural,
            "closed" => BoundaryCondition::Closed,
            "clamped" => BoundaryCondition::Clamped,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid bc_type. Use 'natural', 'closed', or 'clamped'.",
                ))
            }
        };
        let spline =
            CatmullRom::new(verts, grid_opt, alpha, bc, gaussian_sigma).map_err(PyErr::from)?;
        Ok(CatmullRomWrapper { inner: spline })
    }

    #[getter]
    fn grid(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.grid.clone().into_pyarray(py).to_owned().into()
    }

    #[pyo3(signature = (distances, n))]
    fn evaluate(
        &self,
        py: Python,
        distances: PyReadonlyArray1<f64>,
        n: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let array_view = distances.as_array();
        let slice = array_view.as_slice().unwrap();

        let pts = self.inner.evaluate(slice, n).map_err(PyErr::from)?;
        let (n_pts, flat) = points_to_flat(pts);

        // Convert to Python array
        Ok(Array2::from_shape_vec((n_pts, 2), flat)
            .unwrap()
            .into_pyarray(py)
            .to_owned()
            .into())
    }

    fn uniform_distances(
        &self,
        py: Python,
        n_pts: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let distances = self
            .inner
            .uniform_distances(n_pts, tolerance, max_iterations)
            .map_err(PyErr::from)?;
        Ok(distances.into_pyarray(py).to_owned().into())
    }
}

#[pymodule]
fn _catsmoothing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linestrings_tangent_angles, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_linestrings, m)?)?;
    m.add_class::<CatmullRomWrapper>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
