use crate::error::SplineError;
use crate::utils::{are_points_close, interpolate};
use ndarray::parallel::prelude::*;
use ndarray::{array, Array1, Array2};
use num_cpus;
use rayon::ThreadPoolBuilder;
use std::sync::Once;

// Rayon initialization (same as nalgebra version)
pub fn init_rayon() {
    static INIT_RAYON: Once = Once::new();
    INIT_RAYON.call_once(|| {
        let _ = ThreadPoolBuilder::new()
            .num_threads(num_cpus::get_physical())
            .build_global();
        // Ignore errors - pool might already be initialized
    });
}

// Specialized 2D vector type to replace generic Array1
#[derive(Debug, Clone, PartialEq)]
pub struct Vec2D {
    x: f64,
    y: f64,
}

impl Vec2D {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn from_slice(slice: &[f64]) -> Self {
        assert!(slice.len() >= 2);
        Self {
            x: slice[0],
            y: slice[1],
        }
    }

    #[inline]
    pub fn from_array(arr: &Array1<f64>) -> Self {
        assert!(arr.len() >= 2);
        Self {
            x: arr[0],
            y: arr[1],
        }
    }

    #[inline]
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![self.x, self.y])
    }

    #[inline]
    pub fn sub(&self, other: &Vec2D) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    #[inline]
    pub fn add(&self, other: &Vec2D) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    Natural,
    Closed,
    Clamped,
}

// Pre-computed matrix for spline calculations
lazy_static::lazy_static! {
    static ref HERMITE_MATRIX: Array2<f64> = array![
        [2.0, -2.0, 1.0, 1.0],
        [-3.0, 3.0, -2.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ];
}

#[derive(Debug)]
pub struct CatmullRom {
    pub vertices: Vec<Vec2D>,
    pub grid: Vec<f64>,
    pub segments: Vec<Array2<f64>>,
    pub alpha: Option<f64>,
    pub bc_type: BoundaryCondition,
}

impl CatmullRom {
    pub fn new(
        vertices: Vec<[f64; 2]>,
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        bc_type: BoundaryCondition,
        gaussian_sigma: Option<f64>,
    ) -> Result<Self, SplineError> {
        if vertices.len() < 2 {
            return Err(SplineError::TooFewVertices);
        }

        // Initialize rayon at the beginning
        init_rayon();

        let vertices = if matches!(bc_type, BoundaryCondition::Closed) {
            let mut v = vertices;
            if let (Some(first), Some(last)) = (v.first(), v.last()) {
                if are_points_close(first, last) {
                    v.pop();
                }
            }
            if let Some(first) = v.first().copied() {
                v.push(first);
            }
            v
        } else {
            vertices
        };

        // Use specialized Vec2D instead of Array1
        let vertices_vec: Vec<Vec2D> = if let Some(sigma) = gaussian_sigma {
            Self::gaussian_smoothing(&vertices, sigma)
        } else {
            vertices.iter().map(|&v| Vec2D::new(v[0], v[1])).collect()
        };

        let grid = Self::get_grid(grid, alpha, &vertices_vec)?;
        let tangents = Self::get_tangents(&vertices_vec, &grid, &bc_type)?;
        let segments = Self::get_segments(&vertices_vec, &grid, &tangents)?;

        Ok(Self {
            vertices: vertices_vec,
            grid,
            segments,
            alpha,
            bc_type,
        })
    }

    fn gaussian_smoothing(vertices: &[[f64; 2]], sigma: f64) -> Vec<Vec2D> {
        let vec_vertices: Vec<Vec2D> = vertices.iter().map(|&v| Vec2D::new(v[0], v[1])).collect();

        let n_verts = vec_vertices.len();
        if n_verts <= 2 || sigma <= 1e-6 {
            return vec_vertices;
        }

        // Truncate kernel at 4 sigma for good approximation
        let truncate = 4.0;
        let kernel_size = (2 * (truncate * sigma).ceil() as usize) + 1;
        let center = kernel_size / 2;
        let inv_sigma2 = 1.0 / (2.0 * sigma * sigma);

        // Pre-compute kernel for efficiency
        let mut kernel = Vec::with_capacity(kernel_size);
        let mut kernel_sum = 0.0;
        for i in 0..kernel_size {
            let x = i as f64 - center as f64;
            let val = (-x * x * inv_sigma2).exp();
            kernel.push(val);
            kernel_sum += val;
        }

        // Normalize kernel
        let inv_kernel_sum = 1.0 / kernel_sum;
        for k in &mut kernel {
            *k *= inv_kernel_sum;
        }

        // Pre-allocate result vector
        let mut smoothed_vertices = Vec::with_capacity(n_verts);
        smoothed_vertices.push(vec_vertices[0].clone());

        // Only smooth interior points, preserve endpoints
        for i in 1..n_verts - 1 {
            let mut smoothed_x = 0.0;
            let mut smoothed_y = 0.0;

            for j in 0..kernel_size {
                // Mirror boundary handling for convolution
                let idx = if (i as isize + j as isize - center as isize) < 0 {
                    (center - i - j) % n_verts
                } else if i + j - center >= n_verts {
                    (2 * (n_verts - 1) - (i + j - center)) % n_verts
                } else {
                    i + j - center
                };

                smoothed_x += vec_vertices[idx].x * kernel[j];
                smoothed_y += vec_vertices[idx].y * kernel[j];
            }

            smoothed_vertices.push(Vec2D::new(smoothed_x, smoothed_y));
        }

        smoothed_vertices.push(vec_vertices[n_verts - 1].clone());
        smoothed_vertices
    }

    fn get_grid(
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        vertices: &Vec<Vec2D>,
    ) -> Result<Vec<f64>, SplineError> {
        if grid.is_none() && alpha.is_none() {
            // Use efficient iterator for uniform grid
            return Ok((0..vertices.len()).map(|i| i as f64).collect());
        } else if let Some(alpha) = alpha {
            // Pre-allocate for efficiency
            let n = vertices.len();
            let mut norms = Vec::with_capacity(n - 1);
            let mut grid_diffs = Vec::with_capacity(n - 1);
            let mut grid = Vec::with_capacity(n);

            // Calculate differences and norms
            for i in 0..n - 1 {
                let diff = vertices[i + 1].sub(&vertices[i]);
                norms.push(diff.norm());
            }

            // Calculate grid differences
            for &norm in &norms {
                grid_diffs.push(norm.powf(alpha));
            }

            // Construct cumulative grid
            grid.push(0.0);
            let mut cumsum = 0.0;
            for diff in grid_diffs {
                cumsum += diff;
                grid.push(cumsum);
            }

            Ok(grid)
        } else if let Some(grid) = grid {
            if grid.len() != vertices.len() {
                return Err(SplineError::GridLengthMismatch);
            }
            Ok(grid)
        } else {
            Err(SplineError::InvalidBoundaryCondition)
        }
    }

    fn get_tangents(
        vertices: &[Vec2D],
        grid: &[f64],
        bc_type: &BoundaryCondition,
    ) -> Result<Vec<Vec2D>, SplineError> {
        let n_verts = vertices.len();

        // Early error check
        if n_verts < 2 {
            return Err(SplineError::TooFewVertices);
        }

        // Determine working set of vertices and grid points
        let n = if *bc_type == BoundaryCondition::Closed {
            n_verts + 1
        } else {
            n_verts
        };

        // Pre-allocate tangents with the right capacity
        let expected_tangents = 2 * (n_verts - 1);
        let mut tangents = Vec::with_capacity(expected_tangents);

        // Special case for only two points
        if n_verts == 2 {
            let delta = grid[1] - grid[0];
            let tangent = vertices[1].sub(&vertices[0]).scale(1.0 / delta);
            // Push it twice
            tangents.push(tangent.clone());
            tangents.push(tangent);
        } else {
            // Index mapping function to handle the closed case efficiently
            let v_idx = |i: usize| -> usize {
                if *bc_type == BoundaryCondition::Closed && i == n - 1 {
                    1 // For closed curves, wrap back to second point
                } else {
                    i % n_verts // Normal indexing for all other cases
                }
            };

            let g_idx = |i: usize| -> f64 {
                if *bc_type == BoundaryCondition::Closed && i == n - 1 {
                    // For closed curves, the last grid point is computed
                    grid[grid.len() - 1] + grid[1] - grid[0]
                } else {
                    grid[i % grid.len()]
                }
            };

            // Process internal tangents (more efficiently)
            for i in 0..n - 2 {
                let p_1 = &vertices[v_idx(i)];
                let p0 = &vertices[v_idx(i + 1)];
                let p1 = &vertices[v_idx(i + 2)];

                let t_1 = g_idx(i);
                let t0 = g_idx(i + 1);
                let t1 = g_idx(i + 2);

                let delta_1 = t0 - t_1;
                let delta0 = t1 - t0;
                let inv_delta_1 = 1.0 / delta_1;
                let inv_delta0 = 1.0 / delta0;
                let delta_sum_inv = 1.0 / (delta0 + delta_1);

                // Calculate tangent directly with Vec2D operations
                let v_1 = p0.sub(p_1).scale(inv_delta_1);
                let v0 = p1.sub(p0).scale(inv_delta0);
                let tangent = v_1
                    .scale(delta0)
                    .add(&v0.scale(delta_1))
                    .scale(delta_sum_inv);

                // Push tangent twice
                tangents.push(tangent.clone());
                tangents.push(tangent);
            }
        }

        // Handle boundary conditions
        if *bc_type == BoundaryCondition::Closed {
            tangents.rotate_right(1);
        } else {
            // Add boundary tangents for non-closed curves
            let start_tangent = Self::get_bc_tangent(
                bc_type,
                &vertices[0],
                &vertices[1],
                grid[0],
                grid[1],
                &tangents[0],
            )?;

            let end_tangent = Self::get_bc_tangent(
                bc_type,
                &vertices[n_verts - 2],
                &vertices[n_verts - 1],
                grid[n_verts - 2],
                grid[n_verts - 1],
                &tangents[tangents.len() - 1],
            )?;

            // Insert at beginning and push at end
            tangents.insert(0, start_tangent);
            tangents.push(end_tangent);
        }

        // Final validation
        if tangents.len() != expected_tangents {
            return Err(SplineError::TangentError);
        }

        Ok(tangents)
    }

    fn get_bc_tangent(
        bc_type: &BoundaryCondition,
        p0: &Vec2D,
        p1: &Vec2D,
        t0: f64,
        t1: f64,
        other_tangent: &Vec2D,
    ) -> Result<Vec2D, SplineError> {
        match bc_type {
            BoundaryCondition::Natural => {
                let delta = t1 - t0;
                let inv_delta = 1.0 / (2.0 * delta);
                Ok(p1
                    .sub(p0)
                    .scale(3.0 * inv_delta)
                    .add(&other_tangent.scale(-0.5)))
            }
            BoundaryCondition::Clamped => {
                let delta = t1 - t0;
                Ok(p1.sub(p0).scale(1.0 / delta))
            }
            _ => Err(SplineError::InvalidBoundaryCondition),
        }
    }

    fn get_segments(
        vertices: &Vec<Vec2D>,
        grid: &[f64],
        tangents: &Vec<Vec2D>,
    ) -> Result<Vec<Array2<f64>>, SplineError> {
        let n_segments = vertices.len() - 1;

        // Use rayon's par_iter directly instead of into_par_iter for better performance
        let segments: Vec<Array2<f64>> = (0..n_segments)
            .into_par_iter()
            .map(|i| {
                let x0 = &vertices[i];
                let x1 = &vertices[i + 1];
                let v0 = &tangents[2 * i];
                let v1 = &tangents[2 * i + 1];
                let dt = grid[i + 1] - grid[i];

                // Use Array2 directly with fixed size for efficiency
                let mut seg_data = Array2::zeros((4, 2));

                // Fill in values directly
                seg_data[[0, 0]] = x0.x;
                seg_data[[0, 1]] = x0.y;
                seg_data[[1, 0]] = x1.x;
                seg_data[[1, 1]] = x1.y;
                seg_data[[2, 0]] = dt * v0.x;
                seg_data[[2, 1]] = dt * v0.y;
                seg_data[[3, 0]] = dt * v1.x;
                seg_data[[3, 1]] = dt * v1.y;

                // Use reference to HERMITE_MATRIX for dot product
                HERMITE_MATRIX.dot(&seg_data)
            })
            .collect();

        Ok(segments)
    }

    pub fn evaluate(&self, distances: &[f64], n: usize) -> Vec<[f64; 2]> {
        if distances.is_empty() {
            return Vec::new();
        }

        let grid_max_idx = self.grid.len().saturating_sub(2);

        // Pre-compute t-vector functions for different derivative orders
        let t_vector_fn = match n {
            0 => |t: f64| -> [f64; 4] {
                let t2 = t * t;
                let t3 = t2 * t;
                [t3, t2, t, 1.0]
            },
            1 => |t: f64| -> [f64; 4] {
                let t2 = t * t;
                [3.0 * t2, 2.0 * t, 1.0, 0.0]
            },
            2 => |_: f64| -> [f64; 4] { [6.0, 2.0, 0.0, 0.0] },
            _ => panic!("Unsupported derivative order"),
        };

        // Use par_iter for parallel execution with thread-safe references
        distances
            .par_iter()
            .map(|&distance| {
                // Find appropriate spline segment with boundary handling
                let idx = if distance <= self.grid[0] {
                    0
                } else if distance >= self.grid[self.grid.len() - 1] {
                    grid_max_idx
                } else {
                    // Binary search for segment containing the distance
                    match self.grid.binary_search_by(|&x| {
                        if x <= distance {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }) {
                        Ok(i) | Err(i) => i.saturating_sub(1).min(grid_max_idx),
                    }
                };

                let d0 = self.grid[idx];
                let d1 = self.grid[idx + 1];
                let t_normalized = (distance - d0) / (d1 - d0);

                // Get t vector for current parameterization
                let t_vec = t_vector_fn(t_normalized);

                // Get coefficient matrix for current segment
                let coeff = &self.segments[idx];

                // Calculate point coordinates
                let mut x = 0.0;
                let mut y = 0.0;

                for i in 0..4 {
                    x += t_vec[i] * coeff[[i, 0]];
                    y += t_vec[i] * coeff[[i, 1]];
                }

                [x, y]
            })
            .collect()
    }

    pub fn uniform_distances(
        &self,
        n_pts: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Vec<f64> {
        if n_pts < 2 {
            return vec![self.grid[0]];
        }

        // Initial uniform parameterization
        let grid_end = self.grid[self.grid.len() - 1];
        let inv_n_pts_minus_1 = 1.0 / ((n_pts - 1) as f64);

        // Pre-allocate with capacity
        let mut dis_arr = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            dis_arr.push(grid_end * (i as f64) * inv_n_pts_minus_1);
        }

        // Pre-allocate these vectors to reuse across iterations
        let mut arc_lengths = Vec::with_capacity(n_pts);
        let mut uniform_lengths = Vec::with_capacity(n_pts);

        // Reparameterize to achieve uniform arc length
        for _ in 0..max_iterations {
            let points = self.evaluate(&dis_arr, 0);

            arc_lengths.clear();
            arc_lengths.push(0.0);

            let mut cumsum = 0.0;
            for i in 1..n_pts {
                let dx = points[i][0] - points[i - 1][0];
                let dy = points[i][1] - points[i - 1][1];
                cumsum += (dx * dx + dy * dy).sqrt();
                arc_lengths.push(cumsum);
            }

            let total_length = cumsum;
            if total_length < tolerance {
                return dis_arr;
            }

            uniform_lengths.clear();
            for i in 0..n_pts {
                uniform_lengths.push(total_length * (i as f64) * inv_n_pts_minus_1);
            }

            // Check convergence
            let mut max_error: f64 = 0.0;
            for i in 0..n_pts {
                let error = (arc_lengths[i] - uniform_lengths[i]).abs();
                max_error = max_error.max(error);
            }

            if max_error < tolerance {
                break;
            }

            // Use optimized interpolation
            dis_arr = interpolate(&uniform_lengths, &arc_lengths, &dis_arr);
        }

        dis_arr
    }
}
