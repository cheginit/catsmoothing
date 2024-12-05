use crate::error::SplineError;
use crate::utils::{are_points_close, interpolate};
use nalgebra::{Matrix4, Matrix4x2, Vector2, Vector4};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    Natural,
    Closed,
    Clamped,
}

#[derive(Debug)]
pub struct CatmullRomRust {
    pub vertices: Vec<Vector2<f64>>,
    pub grid: Vec<f64>,
    pub segments: Vec<Matrix4x2<f64>>,
    pub alpha: Option<f64>,
    pub bc_type: BoundaryCondition,
}

impl CatmullRomRust {
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

        let vertices_vec = if let Some(sigma) = gaussian_sigma {
            Self::gaussian_smoothing(&vertices, sigma)
        } else {
            vertices.iter().map(|&v| Vector2::new(v[0], v[1])).collect()
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

    fn gaussian_smoothing(vertices: &[[f64; 2]], sigma: f64) -> Vec<Vector2<f64>> {
        let vec_vertices: Vec<Vector2<f64>> =
            vertices.iter().map(|&v| Vector2::new(v[0], v[1])).collect();

        // Set truncate value to 4 for general use
        let truncate = 4.0;
        let kernel_size = (2 * (truncate * sigma).ceil() as usize) + 1; // Ensure odd size
        let mut kernel = vec![0.0; kernel_size];
        let center = kernel_size / 2;

        // Create Gaussian kernel
        let sigma2 = sigma * sigma;
        for i in 0..kernel_size {
            let x = i as f64 - center as f64;
            kernel[i] = (-x * x / (2.0 * sigma2)).exp();
        }

        // Normalize kernel
        let kernel_sum: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= kernel_sum);

        // Apply convolution
        let original_vertices = vec_vertices.clone();
        let smoothed_vertices: Vec<Vector2<f64>> = (0..vec_vertices.len())
            .into_par_iter()
            .map(|i| {
                // Preserve boundary points
                if i == 0 || i == vec_vertices.len() - 1 {
                    return original_vertices[i];
                }

                let mut smoothed = Vector2::zeros();
                for j in 0..kernel_size {
                    let offset = j as isize - center as isize;
                    let idx = if i as isize + offset < 0 {
                        -(i as isize + offset) // Mirror
                    } else if (i as isize + offset) >= vec_vertices.len() as isize {
                        2 * (vec_vertices.len() as isize - 1) - (i as isize + offset)
                    // Mirror
                    } else {
                        i as isize + offset
                    };

                    smoothed += original_vertices[idx as usize] * kernel[j];
                }
                smoothed
            })
            .collect();

        smoothed_vertices
    }

    fn get_grid(
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        vertices: &Vec<Vector2<f64>>,
    ) -> Result<Vec<f64>, SplineError> {
        if grid.is_none() && alpha.is_none() {
            Ok((0..vertices.len()).map(|i| i as f64).collect())
        } else if let Some(alpha) = alpha {
            let diffs: Vec<Vector2<f64>> =
                vertices.windows(2).map(|pair| pair[1] - pair[0]).collect();

            let norms: Vec<f64> = diffs.iter().map(|diff| diff.norm()).collect();

            let grid_diffs: Vec<f64> = norms.iter().map(|&norm| norm.powf(alpha)).collect();

            // Generate grid using simpler cumsum
            let mut cumsum = 0.0;
            let mut grid = vec![0.0];
            grid_diffs.iter().for_each(|&diff| {
                cumsum += diff;
                grid.push(cumsum);
            });

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
        vertices: &[Vector2<f64>],
        grid: &[f64],
        bc_type: &BoundaryCondition,
    ) -> Result<Vec<Vector2<f64>>, SplineError> {
        let mut tangents: Vec<Vector2<f64>> = Vec::new();

        // Create extended slices for the closed case
        let vertices_ext: Vec<_>;
        let grid_ext: Vec<_>;
        let (vert_slice, grid_slice) = if *bc_type == BoundaryCondition::Closed {
            vertices_ext = {
                let mut v = vertices.to_vec();
                v.push(vertices[1].clone());
                v
            };
            grid_ext = {
                let mut g = grid.to_vec();
                let new_grid_value = grid[grid.len() - 1] + grid[1] - grid[0];
                g.push(new_grid_value);
                g
            };
            (&vertices_ext[..], &grid_ext[..])
        } else {
            (vertices, grid)
        };

        let n = vert_slice.len();
        if n == 2 {
            // Handle straight line case
            let tangent = (vert_slice[1] - vert_slice[0]) / (grid_slice[1] - grid_slice[0]);
            tangents.push(tangent);
            tangents.push(tangent);
        } else {
            // Calculate internal tangents
            for i in 0..n - 2 {
                let p_1 = &vert_slice[i];
                let p0 = &vert_slice[i + 1];
                let p1 = &vert_slice[i + 2];
                let t_1 = grid_slice[i];
                let t0 = grid_slice[i + 1];
                let t1 = grid_slice[i + 2];
                let delta_1 = t0 - t_1;
                let delta0 = t1 - t0;
                let v_1 = (p0 - p_1) / delta_1;
                let v0 = (p1 - p0) / delta0;
                let tangent = (delta0 * v_1 + delta_1 * v0) / (delta0 + delta_1);
                tangents.push(tangent);
                tangents.push(tangent);
            }
        }

        if *bc_type == BoundaryCondition::Closed {
            tangents.rotate_right(1);
        } else {
            let start_tangent = Self::get_bc_tangent(
                bc_type,
                &vert_slice[0],
                &vert_slice[1],
                grid_slice[0],
                grid_slice[1],
                &tangents[0],
            )?;
            let end_tangent = Self::get_bc_tangent(
                bc_type,
                &vert_slice[vert_slice.len() - 2],
                &vert_slice[vert_slice.len() - 1],
                grid_slice[grid_slice.len() - 2],
                grid_slice[grid_slice.len() - 1],
                &tangents[tangents.len() - 1],
            )?;
            tangents.insert(0, start_tangent);
            tangents.push(end_tangent);
        }

        if tangents.len() != 2 * (vertices.len() - 1) {
            return Err(SplineError::TangentError);
        }

        Ok(tangents)
    }

    fn get_bc_tangent(
        bc_type: &BoundaryCondition,
        p0: &Vector2<f64>,
        p1: &Vector2<f64>,
        t0: f64,
        t1: f64,
        other_tangent: &Vector2<f64>,
    ) -> Result<Vector2<f64>, SplineError> {
        match bc_type {
            BoundaryCondition::Natural => {
                let delta = t1 - t0;
                let tangent = 3.0 * (p1 - p0) / (2.0 * delta) - other_tangent * 0.5;
                Ok(tangent)
            }
            BoundaryCondition::Clamped => {
                let delta = t1 - t0;
                let tangent = (p1 - p0) / delta;
                Ok(tangent)
            }
            _ => Err(SplineError::InvalidBoundaryCondition),
        }
    }

    fn get_segments(
        vertices: &Vec<Vector2<f64>>,
        grid: &[f64],
        tangents: &[Vector2<f64>],
    ) -> Result<Vec<Matrix4x2<f64>>, SplineError> {
        let n_segments = vertices.len() - 1;
        let matrix = Matrix4::new(
            2.0, -2.0, 1.0, 1.0, -3.0, 3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        );

        let segments: Vec<Matrix4x2<f64>> = (0..n_segments)
            .into_par_iter()
            .map(|i| {
                let x0 = vertices[i];
                let x1 = vertices[i + 1];
                let v0 = tangents[2 * i];
                let v1 = tangents[2 * i + 1];
                let dt = grid[i + 1] - grid[i];

                let segment_data = Matrix4x2::new(
                    x0.x,
                    x0.y,
                    x1.x,
                    x1.y,
                    dt * v0.x,
                    dt * v0.y,
                    dt * v1.x,
                    dt * v1.y,
                );

                matrix * segment_data
            })
            .collect();

        Ok(segments)
    }

    pub fn evaluate(&self, distances: &[f64], n: usize) -> Vec<[f64; 2]> {
        distances
            .par_iter()
            .map(|&distance| {
                let idx = match self.grid.binary_search_by(|&x| {
                    if x <= distance {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                }) {
                    Ok(i) | Err(i) => i.saturating_sub(1),
                };
                let idx = idx.min(self.grid.len() - 2);
                let d0 = self.grid[idx];
                let d1 = self.grid[idx + 1];
                let norm = d1 - d0;
                let t_normalized = (distance - d0) / norm;

                // Define the t_vector for evaluation
                let t_vector = if n == 0 {
                    Vector4::new(
                        t_normalized.powi(3),
                        t_normalized.powi(2),
                        t_normalized,
                        1.0,
                    )
                } else if n == 1 {
                    Vector4::new(3.0 * t_normalized.powi(2), 2.0 * t_normalized, 1.0, 0.0)
                } else if n == 2 {
                    Vector4::new(6.0 * t_normalized, 2.0, 0.0, 0.0)
                } else {
                    panic!("Unsupported derivative order");
                };

                // Extract the coefficients for the current segment
                let coefficients = &self.segments[idx];

                // Compute the point by multiplying the t_vector with coefficients
                let point = coefficients.transpose() * t_vector;

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
        // Initial distances array - linearly spaced
        let mut dis_arr = {
            let grid_end = self.grid[self.grid.len() - 1];
            let step = grid_end / ((n_pts - 1) as f64);
            (0..n_pts).map(|i| i as f64 * step).collect::<Vec<f64>>()
        };

        for _ in 0..max_iterations {
            let points = self.evaluate(&dis_arr, 0);

            let diff: Vec<[f64; 2]> = points
                .windows(2)
                .map(|window| [window[1][0] - window[0][0], window[1][1] - window[0][1]])
                .collect();

            let mut arc_lengths = vec![0.0];
            let mut cumsum = 0.0;
            for d in diff.iter() {
                cumsum += (d[0] * d[0] + d[1] * d[1]).sqrt();
                arc_lengths.push(cumsum);
            }

            let total_length = *arc_lengths.last().unwrap();

            let uniform_lengths: Vec<f64> = (0..n_pts)
                .map(|i| total_length * (i as f64) / ((n_pts - 1) as f64))
                .collect();

            let max_error = arc_lengths
                .iter()
                .zip(uniform_lengths.iter())
                .map(|(a, u)| (a - u).abs())
                .fold(0.0f64, |max, x| max.max(x));

            if max_error < tolerance {
                break;
            }

            // Interpolate new distances
            dis_arr = interpolate(&uniform_lengths, &arc_lengths, &dis_arr);
        }

        dis_arr
    }
}
