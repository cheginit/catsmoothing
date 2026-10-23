use crate::{
    linestring, BoundaryCondition, CatmullRom, LineToSmooth, Presmoothing, SigmaUnits, Spacing,
    SplineError, SplineOptions,
};
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::borrow::Cow;

impl From<SplineError> for PyErr {
    fn from(err: SplineError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// Helper: convert a 2D ndarray view (shape (n,2)) into a Vec<[f64; 2]>.
fn array2_to_vec(line: ArrayView2<f64>) -> PyResult<Vec<[f64; 2]>> {
    if line.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "Vertices must have shape (n, 2), got (n, {})",
            line.ncols()
        )));
    }
    Ok(line.outer_iter().map(|row| [row[0], row[1]]).collect())
}

// Helper: one optional value per line from an optional array of any memory layout.
fn per_line<T: numpy::Element + Copy>(
    arr: Option<PyReadonlyArray1<T>>,
    n_lines: usize,
    name: &str,
) -> PyResult<Vec<Option<T>>> {
    let Some(arr) = arr else {
        return Ok(vec![None; n_lines]);
    };
    let view = arr.as_array();
    validate_array_length(view.len(), n_lines, name)?;
    Ok(view.iter().map(|&x| Some(x)).collect())
}

// Helper: reinterpret points as an (n, 2) array, reusing their allocation.
fn points_to_pyarray(py: Python<'_>, points: Vec<[f64; 2]>) -> Py<PyArray2<f64>> {
    let n = points.len();
    Array2::from_shape_vec((n, 2), points.into_flattened())
        .expect("n points flatten to exactly 2n values")
        .into_pyarray(py)
        .unbind()
}

// Helper: an optional Gaussian sigma as a presmoothing setting.
fn gaussian_presmoothing(sigma: Option<f64>, units: SigmaUnits) -> Option<Presmoothing> {
    sigma.map(|sigma| Presmoothing::Gaussian { sigma, units })
}

// Helper: validate array length matches expected count, returning PyErr if not
fn validate_array_length(arr_len: usize, expected: usize, name: &str) -> PyResult<()> {
    if arr_len != expected {
        return Err(PyValueError::new_err(format!(
            "Length of {name} must match number of lines"
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
#[pyfunction(signature = (vertices, gaussian_sigmas=None, sigma_units="vertices"))]
fn linestrings_tangent_angles(
    py: Python,
    vertices: Vec<PyReadonlyArray2<f64>>,
    gaussian_sigmas: Option<Vec<Option<f64>>>,
    sigma_units: &str,
) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    let units: SigmaUnits = sigma_units.parse()?;
    let n_lines = vertices.len();
    let gs = gaussian_sigmas.unwrap_or_else(|| vec![None; n_lines]);
    validate_array_length(gs.len(), n_lines, "gaussian_sigmas")?;

    let lines_vec: Vec<Vec<[f64; 2]>> = vertices
        .iter()
        .map(|line_arr| array2_to_vec(line_arr.as_array()))
        .collect::<PyResult<_>>()?;
    let angles = py
        .detach(|| {
            let presmoothing = gs
                .into_iter()
                .map(|s| gaussian_presmoothing(s, units))
                .collect();
            linestring::lines_tangents(lines_vec, presmoothing)
        })
        .map_err(PyErr::from)?;

    Ok(angles
        .into_iter()
        .map(|a| a.into_pyarray(py).unbind())
        .collect())
}

/// Python function: smooth_linestrings
///
/// Parameters:
///   - lines: list of NumPy arrays (each shape (n,2))
///   - distances: optional 1D NumPy array (one per linestring)
///   - n_pts: optional 1D NumPy array of usize (one per linestring)
///   - gaussian_sigmas: optional 1D NumPy array (one per linestring)
///   - bc_types: optional Vec<String> (one per linestring; "natural", "closed", or "clamped")
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
        sigma_units="vertices"
    )
)]
fn smooth_linestrings(
    py: Python,
    lines: Vec<PyReadonlyArray2<f64>>,
    distances: Option<PyReadonlyArray1<f64>>,
    n_pts: Option<PyReadonlyArray1<i64>>,
    gaussian_sigmas: Option<PyReadonlyArray1<f64>>,
    bc_types: Option<Vec<String>>,
    sigma_units: &str,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    let units: SigmaUnits = sigma_units.parse()?;
    let n_lines = lines.len();
    let lines_vec: Vec<Vec<[f64; 2]>> = lines
        .iter()
        .map(|line_arr| array2_to_vec(line_arr.as_array()))
        .collect::<PyResult<_>>()?;
    let ds = per_line(distances, n_lines, "distances")?;
    let n_pts_vec = per_line(n_pts, n_lines, "n_pts")?
        .into_iter()
        .map(|n| {
            n.map(usize::try_from)
                .transpose()
                .map_err(|_| PyErr::from(SplineError::InvalidPointCount))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let gs = per_line(gaussian_sigmas, n_lines, "gaussian_sigmas")?;

    let bc_conv: Vec<Option<BoundaryCondition>> = if let Some(bcs) = bc_types {
        validate_array_length(bcs.len(), n_lines, "bc_types")?;
        bcs.iter()
            .map(|s| s.parse().map(Some))
            .collect::<Result<_, SplineError>>()?
    } else {
        vec![None; n_lines]
    };

    let mut jobs = Vec::with_capacity(n_lines);
    for (i, vertices) in lines_vec.into_iter().enumerate() {
        let spacing = match (n_pts_vec[i], ds[i]) {
            (Some(n), _) => Spacing::Count(n),
            (None, Some(d)) => Spacing::MaxDistance(d),
            (None, None) => return Err(SplineError::InvalidSmoothingParameters.into()),
        };
        let options = SplineOptions {
            alpha: Some(0.5),
            bc_type: bc_conv[i].unwrap_or(BoundaryCondition::Clamped),
            presmoothing: gaussian_presmoothing(gs[i], units),
            ..SplineOptions::default()
        };
        jobs.push(LineToSmooth {
            vertices,
            spacing,
            options,
        });
    }

    let smoothed_vec = py
        .detach(|| linestring::smooth_linestrings(jobs))
        .map_err(PyErr::from)?;

    Ok(smoothed_vec
        .into_iter()
        .map(|line| points_to_pyarray(py, line))
        .collect())
}

/// Python class: CatmullRom
///
/// API:
///   CatmullRom(vertices: NDArray[float64], grid: NDArray[float64] | None = None, alpha: float | None = None, bc_type: Literal["natural", "closed", "clamped"] = "natural", gaussian_sigma: float | None = None)
///   Methods:
///     evaluate(distances: NDArray[float64], n: 0|1|2 = 0) -> NDArray[float64]
///     uniform_distances(n_pts: int) -> NDArray[float64]
#[pyclass(name = "CatmullRom")]
struct CatmullRomWrapper {
    inner: CatmullRom,
}

#[pymethods]
impl CatmullRomWrapper {
    #[new]
    #[pyo3(signature = (
        vertices,
        grid=None,
        alpha=None,
        bc_type="natural",
        gaussian_sigma=None,
        sigma_units="vertices"
    ))]
    fn new(
        py: Python,
        vertices: PyReadonlyArray2<f64>,
        grid: Option<PyReadonlyArray1<f64>>,
        alpha: Option<f64>,
        bc_type: &str,
        gaussian_sigma: Option<f64>,
        sigma_units: &str,
    ) -> PyResult<Self> {
        let units: SigmaUnits = sigma_units.parse()?;
        let verts: Vec<[f64; 2]> = array2_to_vec(vertices.as_array())?;
        let grid_opt: Option<Vec<f64>> = grid.map(|g| g.as_array().to_vec());
        let bc: BoundaryCondition = bc_type.parse()?;
        let spline = py
            .detach(|| {
                let options = SplineOptions {
                    grid: grid_opt,
                    alpha,
                    bc_type: bc,
                    presmoothing: gaussian_presmoothing(gaussian_sigma, units),
                };
                CatmullRom::new(verts, options)
            })
            .map_err(PyErr::from)?;
        Ok(CatmullRomWrapper { inner: spline })
    }

    #[getter]
    fn grid(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.grid().to_vec().into_pyarray(py).unbind()
    }

    #[pyo3(signature = (distances, n))]
    fn evaluate(
        &self,
        py: Python,
        distances: PyReadonlyArray1<f64>,
        n: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let view = distances.as_array();
        let distances: Cow<[f64]> = view
            .as_slice()
            .map_or_else(|| Cow::Owned(view.to_vec()), Cow::Borrowed);

        let pts = py
            .detach(|| self.inner.evaluate(&distances, n))
            .map_err(PyErr::from)?;
        Ok(points_to_pyarray(py, pts))
    }

    fn uniform_distances(&self, py: Python, n_pts: usize) -> PyResult<Py<PyArray1<f64>>> {
        let distances = py
            .detach(|| self.inner.uniform_distances(n_pts))
            .map_err(PyErr::from)?;
        Ok(distances.into_pyarray(py).unbind())
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
