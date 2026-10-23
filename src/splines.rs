use crate::error::SplineError;
use crate::utils::{are_points_close, interpolate};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::ops::{Add, Mul, Sub};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Vec2D {
    x: f64,
    y: f64,
}

impl Vec2D {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
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

impl Add for Vec2D {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Vec2D {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<f64> for Vec2D {
    type Output = Self;

    #[inline]
    fn mul(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }
}

/// Reflects `k` into `0..n` about both ends without repeating the edge sample, for any
/// kernel reach (`... c b | a b c d | c b a ...`). Requires `n >= 2`.
fn mirror_index(k: isize, n: usize) -> usize {
    let period = 2 * (n as isize - 1);
    let m = k.rem_euclid(period);
    (if m >= n as isize { period - m } else { m }) as usize
}

/// End conditions of a spline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Zero second derivative at both ends.
    Natural,
    /// Periodic: the curve returns to its first vertex with matching tangents.
    Closed,
    /// End tangents follow the first and last chords.
    Clamped,
}

impl FromStr for BoundaryCondition {
    type Err = SplineError;

    /// Parses `"natural"`, `"closed"`, or `"clamped"`, ignoring case.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "natural" => Ok(Self::Natural),
            "closed" => Ok(Self::Closed),
            "clamped" => Ok(Self::Clamped),
            _ => Err(SplineError::InvalidBoundaryCondition),
        }
    }
}

// Rayon dispatch costs more than evaluating a few thousand points serially.
const PARALLEL_MIN_POINTS: usize = 4096;

/// Cubic `a t^3 + b t^2 + c t + d` over the segment parameter `t` in `[0, 1]`.
#[derive(Debug, Clone, Copy)]
struct Segment {
    a: Vec2D,
    b: Vec2D,
    c: Vec2D,
    d: Vec2D,
}

/// A cubic Catmull-Rom spline through 2D vertices, one Hermite segment per vertex pair.
#[derive(Debug)]
pub struct CatmullRom {
    grid: Vec<f64>,
    segments: Vec<Segment>,
}

impl CatmullRom {
    /// Fits a spline through `vertices`.
    ///
    /// `alpha` builds the grid from chord lengths raised to `alpha` (`0.5` is centripetal,
    /// `0` uniform, `1` chordal) and takes precedence over `grid`; with neither, the grid is
    /// `0, 1, 2, ...`. `gaussian_sigma` smooths the vertices first, keeping the end vertices.
    /// Closed curves append the first vertex unless the last vertex already matches it.
    /// Without an explicit `grid`, repeated consecutive vertices are dropped.
    ///
    /// # Errors
    ///
    /// Fails for fewer than two vertices, nonfinite inputs, a grid that is not strictly
    /// increasing or does not match the vertex count, or a negative `gaussian_sigma`.
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
        if vertices.iter().flatten().any(|c| !c.is_finite()) {
            return Err(SplineError::NonFiniteVertices);
        }
        if alpha.is_some_and(|a| !a.is_finite()) {
            return Err(SplineError::InvalidAlpha);
        }
        if gaussian_sigma.is_some_and(|s| !(s.is_finite() && s >= 0.0)) {
            return Err(SplineError::InvalidGaussianSigma);
        }

        // Repeated vertices would give zero grid steps; an explicit grid keeps them.
        let mut vertices = vertices;
        if grid.is_none() {
            vertices.dedup();
            if vertices.len() < 2 {
                return Err(SplineError::TooFewVertices);
            }
        }

        let vertices = if bc_type == BoundaryCondition::Closed {
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

        let vertices_vec: Vec<Vec2D> = if let Some(sigma) = gaussian_sigma {
            Self::gaussian_smoothing(&vertices, sigma)
        } else {
            vertices.iter().map(|&v| Vec2D::new(v[0], v[1])).collect()
        };

        let grid = Self::get_grid(grid, alpha, &vertices_vec)?;
        let tangents = Self::get_tangents(&vertices_vec, &grid, bc_type);
        let segments = Self::get_segments(&vertices_vec, &grid, &tangents);

        Ok(Self { grid, segments })
    }

    /// Parameter value at each vertex.
    pub fn grid(&self) -> &[f64] {
        &self.grid
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
        smoothed_vertices.push(vec_vertices[0]);

        // Only smooth interior points, preserve endpoints
        for i in 1..n_verts - 1 {
            let mut smoothed_x = 0.0;
            let mut smoothed_y = 0.0;

            for j in 0..kernel_size {
                let idx = mirror_index(i as isize + j as isize - center as isize, n_verts);
                smoothed_x += vec_vertices[idx].x * kernel[j];
                smoothed_y += vec_vertices[idx].y * kernel[j];
            }

            smoothed_vertices.push(Vec2D::new(smoothed_x, smoothed_y));
        }

        smoothed_vertices.push(vec_vertices[n_verts - 1]);
        smoothed_vertices
    }

    /// Builds the knot grid. `alpha` takes precedence over an explicit `grid`; with neither,
    /// knots are uniform.
    fn get_grid(
        grid: Option<Vec<f64>>,
        alpha: Option<f64>,
        vertices: &[Vec2D],
    ) -> Result<Vec<f64>, SplineError> {
        match (alpha, grid) {
            (Some(alpha), _) => {
                let mut grid = Vec::with_capacity(vertices.len());
                let mut total = 0.0;
                grid.push(total);
                for pair in vertices.windows(2) {
                    total += (pair[1] - pair[0]).norm().powf(alpha);
                    grid.push(total);
                }
                Ok(grid)
            }
            (None, Some(grid)) => {
                if grid.len() != vertices.len() {
                    return Err(SplineError::GridLengthMismatch);
                }
                if !(grid.iter().all(|g| g.is_finite()) && grid.windows(2).all(|w| w[0] < w[1])) {
                    return Err(SplineError::InvalidGrid);
                }
                Ok(grid)
            }
            (None, None) => Ok((0..vertices.len()).map(|i| i as f64).collect()),
        }
    }

    /// Returns one tangent per vertex; segment `i` runs from tangent `i` to tangent `i + 1`.
    /// Closed curves repeat the first vertex at the end, so their first and last tangents match.
    fn get_tangents(vertices: &[Vec2D], grid: &[f64], bc_type: BoundaryCondition) -> Vec<Vec2D> {
        let n_verts = vertices.len();
        let mut tangents = vec![Vec2D::new(0.0, 0.0); n_verts];
        for k in 1..n_verts - 1 {
            tangents[k] = Self::interior_tangent(
                &vertices[k - 1],
                &vertices[k],
                &vertices[k + 1],
                grid[k] - grid[k - 1],
                grid[k + 1] - grid[k],
            );
        }

        let last = n_verts - 1;
        match bc_type {
            BoundaryCondition::Closed => {
                let seam = Self::interior_tangent(
                    &vertices[last - 1],
                    &vertices[0],
                    &vertices[1],
                    grid[last] - grid[last - 1],
                    grid[1] - grid[0],
                );
                tangents[0] = seam;
                tangents[last] = seam;
            }
            BoundaryCondition::Natural | BoundaryCondition::Clamped => {
                // A two-vertex line has no interior tangent; the chord slope stands in.
                let chord = (vertices[1] - vertices[0]) * (1.0 / (grid[1] - grid[0]));
                let (after_start, before_end) = if n_verts > 2 {
                    (tangents[1], tangents[last - 1])
                } else {
                    (chord, chord)
                };
                tangents[0] = Self::end_tangent(
                    bc_type,
                    &vertices[0],
                    &vertices[1],
                    grid[1] - grid[0],
                    &after_start,
                );
                tangents[last] = Self::end_tangent(
                    bc_type,
                    &vertices[last - 1],
                    &vertices[last],
                    grid[last] - grid[last - 1],
                    &before_end,
                );
            }
        }
        tangents
    }

    fn interior_tangent(
        p_prev: &Vec2D,
        p: &Vec2D,
        p_next: &Vec2D,
        delta_prev: f64,
        delta_next: f64,
    ) -> Vec2D {
        let v_prev = (*p - *p_prev) * (1.0 / delta_prev);
        let v_next = (*p_next - *p) * (1.0 / delta_next);
        (v_prev * delta_next + v_next * delta_prev) * (1.0 / (delta_next + delta_prev))
    }

    fn end_tangent(
        bc_type: BoundaryCondition,
        p0: &Vec2D,
        p1: &Vec2D,
        delta: f64,
        neighbor_tangent: &Vec2D,
    ) -> Vec2D {
        if bc_type == BoundaryCondition::Natural {
            let inv_delta = 1.0 / (2.0 * delta);
            (*p1 - *p0) * (3.0 * inv_delta) + *neighbor_tangent * -0.5
        } else {
            (*p1 - *p0) * (1.0 / delta)
        }
    }

    /// Converts each Hermite segment (end points and grid-scaled tangents) to power-basis
    /// coefficients. The Hermite basis matrix is folded into the four sums.
    fn get_segments(vertices: &[Vec2D], grid: &[f64], tangents: &[Vec2D]) -> Vec<Segment> {
        (0..vertices.len() - 1)
            .map(|i| {
                let (x0, x1) = (vertices[i], vertices[i + 1]);
                let dt = grid[i + 1] - grid[i];
                let (v0, v1) = (tangents[i] * dt, tangents[i + 1] * dt);
                Segment {
                    a: x0 * 2.0 + x1 * -2.0 + v0 + v1,
                    b: x0 * -3.0 + x1 * 3.0 + v0 * -2.0 + v1 * -1.0,
                    c: v0,
                    d: x0,
                }
            })
            .collect()
    }

    /// Evaluates the curve, or its `n`-th derivative with respect to the grid parameter, at
    /// each distance. Distances outside the grid extend the first or last segment.
    ///
    /// # Errors
    ///
    /// Fails when `n` is greater than 2.
    pub fn evaluate(&self, distances: &[f64], n: usize) -> Result<Vec<[f64; 2]>, SplineError> {
        if n > 2 {
            return Err(SplineError::UnsupportedDerivativeOrder);
        }
        let point = |&distance: &f64| self.evaluate_one(distance, n);
        if distances.len() < PARALLEL_MIN_POINTS {
            Ok(distances.iter().map(point).collect())
        } else {
            Ok(distances.par_iter().map(point).collect())
        }
    }

    fn evaluate_one(&self, distance: f64, n: usize) -> [f64; 2] {
        // Points outside the grid extrapolate from the first or last segment.
        let idx = self
            .grid
            .partition_point(|&g| g <= distance)
            .saturating_sub(1)
            .min(self.segments.len() - 1);
        let d0 = self.grid[idx];
        let dt = self.grid[idx + 1] - d0;
        let t = (distance - d0) / dt;
        let Segment { a, b, c, d } = self.segments[idx];
        // Derivatives are taken with respect to the grid, so undo the segment normalization.
        let p = match n {
            0 => ((a * t + b) * t + c) * t + d,
            1 => ((a * (3.0 * t) + b * 2.0) * t + c) * (1.0 / dt),
            _ => (a * (6.0 * t) + b * 2.0) * (1.0 / (dt * dt)),
        };
        [p.x, p.y]
    }

    /// Returns `n_pts` grid parameters whose evaluated points are evenly spaced along the
    /// curve, or the first grid value when `n_pts` is below 2.
    ///
    /// # Errors
    ///
    /// Fails when `tolerance` is not positive.
    pub fn uniform_distances(
        &self,
        n_pts: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Vec<f64>, SplineError> {
        if n_pts < 2 {
            return Ok(vec![self.grid[0]]);
        }
        if tolerance.is_nan() || tolerance <= 0.0 {
            return Err(SplineError::InvalidTolerance);
        }

        // Initial uniform parameterization
        let grid_start = self.grid[0];
        let grid_span = self.grid[self.grid.len() - 1] - grid_start;
        let inv_n_pts_minus_1 = 1.0 / ((n_pts - 1) as f64);

        // Pre-allocate with capacity
        let mut dis_arr = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            dis_arr.push(grid_start + grid_span * (i as f64) * inv_n_pts_minus_1);
        }

        // Pre-allocate these vectors to reuse across iterations
        let mut arc_lengths = Vec::with_capacity(n_pts);
        let mut uniform_lengths = Vec::with_capacity(n_pts);

        // Reparameterize to achieve uniform arc length
        for _ in 0..max_iterations {
            let points = self.evaluate(&dis_arr, 0)?;

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
                return Ok(dis_arr);
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

        Ok(dis_arr)
    }
}
