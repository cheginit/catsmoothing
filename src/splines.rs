use crate::error::SplineError;
use crate::utils::{are_points_close, interpolate};
use ndarray::parallel::prelude::*;
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    Natural,
    Closed,
    Clamped,
}

#[derive(Debug)]
pub struct CatmullRom {
    pub vertices: Vec<Array1<f64>>,
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

        let vertices_vec: Vec<Array1<f64>> = if let Some(sigma) = gaussian_sigma {
            Self::gaussian_smoothing(&vertices, sigma)
        } else {
            vertices
                .iter()
                .map(|&v| Array1::from(vec![v[0], v[1]]))
                .collect()
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

    fn gaussian_smoothing(vertices: &[[f64; 2]], sigma: f64) -> Vec<Array1<f64>> {
        let vec_vertices: Vec<Array1<f64>> = vertices
            .iter()
            .map(|&v| Array1::from(vec![v[0], v[1]]))
            .collect();

        let n_verts = vec_vertices.len();
        if n_verts <= 2 || sigma <= 1e-6 {
            return vec_vertices;
        }

        // Truncate kernel at 4 sigma for good approximation
        let truncate = 4.0;
        let kernel_size = (2 * (truncate * sigma).ceil() as usize) + 1;
        let center = kernel_size / 2;
        let inv_sigma2 = 1.0 / (2.0 * sigma * sigma);

        let mut kernel = Vec::with_capacity(kernel_size);
        let mut kernel_sum = 0.0;
        for i in 0..kernel_size {
            let x = i as f64 - center as f64;
            let val = (-x * x * inv_sigma2).exp();
            kernel.push(val);
            kernel_sum += val;
        }

        let inv_kernel_sum = 1.0 / kernel_sum;
        for k in &mut kernel {
            *k *= inv_kernel_sum;
        }

        let mut smoothed_vertices = Vec::with_capacity(n_verts);
        smoothed_vertices.push(vec_vertices[0].clone());

        for i in 1..n_verts - 1 {
            let mut smoothed = Array1::<f64>::zeros(2);

            for j in 0..kernel_size {
                // Mirror boundary handling for convolution
                let idx = if (i as isize + j as isize - center as isize) < 0 {
                    (center - i - j) % n_verts
                } else if i + j - center >= n_verts {
                    (2 * (n_verts - 1) - (i + j - center)) % n_verts
                } else {
                    i + j - center
                };

                for d in 0..2 {
                    smoothed[d] += vec_vertices[idx][d] * kernel[j];
                }
            }

            smoothed_vertices.push(smoothed);
        }

        smoothed_vertices.push(vec_vertices[n_verts - 1].clone());

        smoothed_vertices
    }

    fn get_grid(
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        vertices: &Vec<Array1<f64>>,
    ) -> Result<Vec<f64>, SplineError> {
        if grid.is_none() && alpha.is_none() {
            Ok((0..vertices.len()).map(|i| i as f64).collect())
        } else if let Some(alpha) = alpha {
            let diffs: Vec<Array1<f64>> = vertices
                .windows(2)
                .map(|pair| &pair[1] - &pair[0])
                .collect();
            let norms: Vec<f64> = diffs.iter().map(|diff| diff.dot(diff).sqrt()).collect();
            let grid_diffs: Vec<f64> = norms.iter().map(|&norm| norm.powf(alpha)).collect();
            let mut cumsum = 0.0;
            let mut grid = vec![0.0];
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
        vertices: &[Array1<f64>],
        grid: &[f64],
        bc_type: &BoundaryCondition,
    ) -> Result<Vec<Array1<f64>>, SplineError> {
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
            // Create this tangent just once
            let mut tangent = Array1::zeros(2);
            for i in 0..2 {
                tangent[i] = (vertices[1][i] - vertices[0][i]) / delta;
            }
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
                let delta_sum_inv = 1.0 / (delta0 + delta_1);

                // Create the tangent directly
                let mut tangent = Array1::zeros(2);
                for j in 0..2 {
                    let v_1 = (p0[j] - p_1[j]) / delta_1;
                    let v0 = (p1[j] - p0[j]) / delta0;
                    tangent[j] = (v_1 * delta0 + v0 * delta_1) * delta_sum_inv;
                }

                // Push tangent twice (more efficient than clone + push)
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
        p0: &Array1<f64>,
        p1: &Array1<f64>,
        t0: f64,
        t1: f64,
        other_tangent: &Array1<f64>,
    ) -> Result<Array1<f64>, SplineError> {
        match bc_type {
            BoundaryCondition::Natural => {
                let delta = t1 - t0;
                Ok(((p1.clone() - p0.clone()) * 3.0 / (2.0 * delta)) - other_tangent * 0.5)
            }
            BoundaryCondition::Clamped => {
                let delta = t1 - t0;
                Ok((p1.clone() - p0.clone()) / delta)
            }
            _ => Err(SplineError::InvalidBoundaryCondition),
        }
    }

    fn get_segments(
        vertices: &Vec<Array1<f64>>,
        grid: &[f64],
        tangents: &Vec<Array1<f64>>,
    ) -> Result<Vec<Array2<f64>>, SplineError> {
        let n_segments = vertices.len() - 1;
        let matrix = array![
            [2.0, -2.0, 1.0, 1.0],
            [-3.0, 3.0, -2.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ];
        let segments: Vec<Array2<f64>> = (0..n_segments)
            .into_par_iter()
            .map(|i| {
                let x0 = &vertices[i];
                let x1 = &vertices[i + 1];
                let v0 = &tangents[2 * i];
                let v1 = &tangents[2 * i + 1];
                let dt = grid[i + 1] - grid[i];
                let seg_data = array![
                    [x0[0], x0[1]],
                    [x1[0], x1[1]],
                    [dt * v0[0], dt * v0[1]],
                    [dt * v1[0], dt * v1[1]]
                ];
                matrix.dot(&seg_data)
            })
            .collect();
        Ok(segments)
    }

    pub fn evaluate(&self, distances: &[f64], n: usize) -> Vec<[f64; 2]> {
        if distances.is_empty() {
            return Vec::new();
        }

        let grid_max_idx = self.grid.len().saturating_sub(2);

        // Choose basis function based on the derivative order
        let t_vector_fn = match n {
            0 => |t: f64| {
                let t2 = t * t;
                array![t2 * t, t2, t, 1.0]
            },
            1 => |t: f64| {
                let t2 = t * t;
                array![3.0 * t2, 2.0 * t, 1.0, 0.0]
            },
            2 => |_: f64| array![6.0, 2.0, 0.0, 0.0],
            _ => panic!("Unsupported derivative order"),
        };

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
                    let mut left = 0;
                    let mut right = self.grid.len() - 1;

                    while left + 1 < right {
                        let mid = (left + right) / 2;
                        if self.grid[mid] <= distance {
                            left = mid;
                        } else {
                            right = mid;
                        }
                    }
                    left
                };

                let d0 = self.grid[idx];
                let d1 = self.grid[idx + 1];
                let t_normalized = (distance - d0) / (d1 - d0);

                let t_vector = t_vector_fn(t_normalized);
                let coeff = &self.segments[idx];
                let point = coeff.t().dot(&t_vector);

                [point[0], point[1]]
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

        let mut dis_arr = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            dis_arr.push(grid_end * (i as f64) * inv_n_pts_minus_1);
        }

        // Reuse these vectors across iterations
        let mut arc_lengths = Vec::with_capacity(n_pts);
        let mut uniform_lengths = Vec::with_capacity(n_pts);

        // Reparameterize to achieve uniform arc length
        for iter in 0..max_iterations {
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

            // Early exit for small curves
            if iter > 0 && cumsum < tolerance {
                return dis_arr;
            }

            let total_length = cumsum;

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

            dis_arr = interpolate(&uniform_lengths, &arc_lengths, &dis_arr);
        }

        dis_arr
    }
}
