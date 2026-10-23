// Binding signatures mirror the Python API's keyword arguments.
#![allow(clippy::too_many_arguments)]

use crate::{
    linestring, BoundaryCondition, CatmullRom, Continuity, LineToSmooth, Presmoothing, SigmaUnits,
    Spacing, SplineError, SplineOptions,
};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
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

// Results of all lines stacked into one array, with the number of rows per line.
type Stacked<T> = (Py<T>, Py<PyArray1<i64>>);

// Helper: split stacked (n, 2) coordinates into lines at `offsets`, which rise from 0 to n.
fn split_lines(coords: ArrayView2<f64>, offsets: ArrayView1<i64>) -> PyResult<Vec<Vec<[f64; 2]>>> {
    let points = array2_to_vec(coords)?;
    let bounds = offsets
        .iter()
        .map(|&o| usize::try_from(o))
        .collect::<Result<Vec<usize>, _>>()
        .ok()
        .filter(|b| {
            b.first() == Some(&0)
                && b.last() == Some(&points.len())
                && b.windows(2).all(|w| w[0] <= w[1])
        })
        .ok_or_else(|| PyValueError::new_err("offsets must rise from 0 to the coordinate count"))?;
    Ok(bounds
        .windows(2)
        .map(|w| points[w[0]..w[1]].to_vec())
        .collect())
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

// Helper: a batch-wide Taubin setting, which excludes Gaussian sigmas.
fn taubin_setting(taubin_iterations: Option<i64>, any_sigma: bool) -> PyResult<Option<usize>> {
    let Some(n) = taubin_iterations else {
        return Ok(None);
    };
    if any_sigma {
        return Err(SplineError::ConflictingPresmoothing.into());
    }
    match usize::try_from(n) {
        Ok(n) if n >= 1 => Ok(Some(n)),
        _ => Err(SplineError::InvalidTaubinIterations.into()),
    }
}

// Helper: one line's presmoothing, Taubin when set, otherwise its optional Gaussian sigma.
fn presmoothing_for(
    sigma: Option<f64>,
    units: SigmaUnits,
    taubin: Option<usize>,
) -> Option<Presmoothing> {
    match taubin {
        Some(iterations) => Some(Presmoothing::Taubin { iterations }),
        None => sigma.map(|sigma| Presmoothing::Gaussian { sigma, units }),
    }
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
///   - coords: all lines' coordinates stacked, shape (n, 2)
///   - offsets: start of each line in `coords`, plus the total count (n_lines + 1 entries)
///   - gaussian_sigmas: optional 1D NumPy array (one per linestring)
///   - closed: optional 1D boolean array; closed lines get periodic splines
///
/// Returns:
///   - tangent angles of all lines stacked, and the count per line.
#[pyfunction(signature = (
    coords,
    offsets,
    gaussian_sigmas=None,
    sigma_units="vertices",
    taubin_iterations=None,
    closed=None
))]
fn linestrings_tangent_angles(
    py: Python,
    coords: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray1<i64>,
    gaussian_sigmas: Option<PyReadonlyArray1<f64>>,
    sigma_units: &str,
    taubin_iterations: Option<i64>,
    closed: Option<PyReadonlyArray1<bool>>,
) -> PyResult<Stacked<PyArray1<f64>>> {
    let units: SigmaUnits = sigma_units.parse()?;
    let line_vertices = split_lines(coords.as_array(), offsets.as_array())?;
    let n_lines = line_vertices.len();
    let sigmas = per_line(gaussian_sigmas, n_lines, "gaussian_sigmas")?;
    let taubin = taubin_setting(taubin_iterations, sigmas.iter().any(Option::is_some))?;
    let closed: Vec<bool> = per_line(closed, n_lines, "closed")?
        .into_iter()
        .map(|c| c.unwrap_or(false))
        .collect();

    let angles = py
        .detach(|| {
            let presmoothing = sigmas
                .into_iter()
                .map(|s| presmoothing_for(s, units, taubin))
                .collect();
            linestring::lines_tangents(line_vertices, presmoothing, closed)
        })
        .map_err(PyErr::from)?;

    let counts: Vec<i64> = angles.iter().map(|a| a.len() as i64).collect();
    let flat: Vec<f64> = angles.into_iter().flatten().collect();
    Ok((
        flat.into_pyarray(py).unbind(),
        counts.into_pyarray(py).unbind(),
    ))
}

/// Python function: smooth_linestrings
///
/// Parameters:
///   - coords: all lines' coordinates stacked, shape (n, 2)
///   - offsets: start of each line in `coords`, plus the total count (n_lines + 1 entries)
///   - distances: optional 1D NumPy array (one per linestring)
///   - n_pts: optional 1D NumPy array of usize (one per linestring)
///   - gaussian_sigmas: optional 1D NumPy array (one per linestring)
///   - bc_types: optional Vec<String> (one per linestring; "natural", "closed", or "clamped")
///
/// Returns:
///   - smoothed coordinates of all lines stacked, shape (m, 2), and the count per line.
#[pyfunction(
    signature = (
        coords,
        offsets,
        distances=None,
        n_pts=None,
        gaussian_sigmas=None,
        bc_types=None,
        sigma_units="vertices",
        taubin_iterations=None,
        continuity="c1"
    )
)]
fn smooth_linestrings(
    py: Python,
    coords: PyReadonlyArray2<f64>,
    offsets: PyReadonlyArray1<i64>,
    distances: Option<PyReadonlyArray1<f64>>,
    n_pts: Option<PyReadonlyArray1<i64>>,
    gaussian_sigmas: Option<PyReadonlyArray1<f64>>,
    bc_types: Option<Vec<String>>,
    sigma_units: &str,
    taubin_iterations: Option<i64>,
    continuity: &str,
) -> PyResult<Stacked<PyArray2<f64>>> {
    let units: SigmaUnits = sigma_units.parse()?;
    let continuity: Continuity = continuity.parse()?;
    let line_vertices = split_lines(coords.as_array(), offsets.as_array())?;
    let n_lines = line_vertices.len();
    let max_distances = per_line(distances, n_lines, "distances")?;
    let point_counts = per_line(n_pts, n_lines, "n_pts")?
        .into_iter()
        .map(|n| {
            n.map(usize::try_from)
                .transpose()
                .map_err(|_| PyErr::from(SplineError::InvalidPointCount))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let sigmas = per_line(gaussian_sigmas, n_lines, "gaussian_sigmas")?;
    let taubin = taubin_setting(taubin_iterations, sigmas.iter().any(Option::is_some))?;

    let bc_per_line: Vec<Option<BoundaryCondition>> = if let Some(bcs) = bc_types {
        validate_array_length(bcs.len(), n_lines, "bc_types")?;
        bcs.iter()
            .map(|s| s.parse().map(Some))
            .collect::<Result<_, SplineError>>()?
    } else {
        vec![None; n_lines]
    };

    let mut jobs = Vec::with_capacity(n_lines);
    for (i, vertices) in line_vertices.into_iter().enumerate() {
        let spacing = match (point_counts[i], max_distances[i]) {
            (Some(n), _) => Spacing::Count(n),
            (None, Some(d)) => Spacing::MaxDistance(d),
            (None, None) => return Err(SplineError::InvalidSmoothingParameters.into()),
        };
        let options = SplineOptions {
            alpha: Some(0.5),
            bc_type: bc_per_line[i].unwrap_or(BoundaryCondition::Clamped),
            presmoothing: presmoothing_for(sigmas[i], units, taubin),
            continuity,
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

    let counts: Vec<i64> = smoothed_vec.iter().map(|l| l.len() as i64).collect();
    let flat: Vec<[f64; 2]> = smoothed_vec.into_iter().flatten().collect();
    Ok((
        points_to_pyarray(py, flat),
        counts.into_pyarray(py).unbind(),
    ))
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
        sigma_units="vertices",
        taubin_iterations=None,
        continuity="c1"
    ))]
    fn new(
        py: Python,
        vertices: PyReadonlyArray2<f64>,
        grid: Option<PyReadonlyArray1<f64>>,
        alpha: Option<f64>,
        bc_type: &str,
        gaussian_sigma: Option<f64>,
        sigma_units: &str,
        taubin_iterations: Option<i64>,
        continuity: &str,
    ) -> PyResult<Self> {
        let units: SigmaUnits = sigma_units.parse()?;
        let continuity: Continuity = continuity.parse()?;
        let taubin = taubin_setting(taubin_iterations, gaussian_sigma.is_some())?;
        let vertex_list: Vec<[f64; 2]> = array2_to_vec(vertices.as_array())?;
        let grid_values: Option<Vec<f64>> = grid.map(|g| g.as_array().to_vec());
        let bc: BoundaryCondition = bc_type.parse()?;
        let spline = py
            .detach(|| {
                let options = SplineOptions {
                    grid: grid_values,
                    alpha,
                    bc_type: bc,
                    presmoothing: presmoothing_for(gaussian_sigma, units, taubin),
                    continuity,
                };
                CatmullRom::new(vertex_list, options)
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
