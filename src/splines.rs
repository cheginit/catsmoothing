use crate::error::SplineError;
use crate::presmooth;
use crate::utils::ends_coincide;
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

// Positive nodes and their weights of the 16-point Gauss-Legendre rule on [-1, 1]. Exact for
// polynomials up to degree 31; about 1e-5 relative error on noisy centripetal segments.
const GAUSS_LEGENDRE_NODES: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GAUSS_LEGENDRE_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_6,
    0.169_156_519_395_002_5,
    0.149_595_988_816_576_7,
    0.124_628_971_255_533_9,
    0.095_158_511_682_492_8,
    0.062_253_523_938_647_9,
    0.027_152_459_411_754_1,
];

impl Segment {
    fn velocity(&self, t: f64) -> Vec2D {
        (self.a * (3.0 * t) + self.b * 2.0) * t + self.c
    }

    /// Arc length over `[0, t]`.
    fn length_to(&self, t: f64) -> f64 {
        let half = 0.5 * t;
        let mut sum = 0.0;
        for (x, w) in GAUSS_LEGENDRE_NODES.iter().zip(&GAUSS_LEGENDRE_WEIGHTS) {
            let left = self.velocity(half * (1.0 - x)).norm();
            let right = self.velocity(half * (1.0 + x)).norm();
            sum += w * (left + right);
        }
        half * sum
    }

    /// Segment parameter where the arc length from the start reaches `target`, given the
    /// segment's full `length`. Newton steps stay inside a shrinking bisection bracket, which
    /// also covers points where the speed vanishes.
    fn param_at_length(&self, target: f64, length: f64) -> f64 {
        if length <= 0.0 {
            return 0.0;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        let mut t = target / length;
        for _ in 0..64 {
            let residual = self.length_to(t) - target;
            if residual.abs() <= 1e-12 * length {
                break;
            }
            if residual > 0.0 {
                hi = t;
            } else {
                lo = t;
            }
            let newton = t - residual / self.velocity(t).norm();
            t = if newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
        }
        t
    }
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
    /// `0, 1, 2, ...`. `gaussian_sigma` (in vertices) smooths the vertices first: open lines
    /// keep their end vertices and reflect through them, closed rings wrap around.
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
            if ends_coincide(&v) {
                v.pop();
            }
            if let Some(first) = v.first().copied() {
                v.push(first);
            }
            v
        } else {
            vertices
        };

        let vertices_vec: Vec<Vec2D> = vertices.iter().map(|&v| Vec2D::new(v[0], v[1])).collect();
        let vertices_vec = match gaussian_sigma {
            Some(sigma) => {
                presmooth::gaussian(&vertices_vec, sigma, bc_type == BoundaryCondition::Closed)
            }
            None => vertices_vec,
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

    /// Total arc length of the curve.
    pub fn length(&self) -> f64 {
        self.segments.iter().map(|s| s.length_to(1.0)).sum()
    }

    /// Returns `n_pts` grid parameters whose evaluated points are evenly spaced by arc length,
    /// starting and ending at the grid ends, or the first grid value when `n_pts` is below 2.
    ///
    /// # Errors
    ///
    /// Fails when the output cannot be allocated.
    pub fn uniform_distances(&self, n_pts: usize) -> Result<Vec<f64>, SplineError> {
        if n_pts < 2 {
            return Ok(vec![self.grid[0]]);
        }
        let mut distances = Vec::new();
        distances
            .try_reserve_exact(n_pts)
            .map_err(|_| SplineError::TooManyPoints)?;

        let lengths: Vec<f64> = self.segments.iter().map(|s| s.length_to(1.0)).collect();
        let total: f64 = lengths.iter().sum();
        // Targets increase, so one forward sweep finds every target's segment.
        let (mut idx, mut start) = (0, 0.0);
        for k in 0..n_pts {
            let target = total * k as f64 / (n_pts - 1) as f64;
            while idx + 1 < lengths.len() && start + lengths[idx] < target {
                start += lengths[idx];
                idx += 1;
            }
            let local = (target - start).clamp(0.0, lengths[idx]);
            let t = self.segments[idx].param_at_length(local, lengths[idx]);
            distances.push(self.grid[idx] + t * (self.grid[idx + 1] - self.grid[idx]));
        }
        distances[0] = self.grid[0];
        distances[n_pts - 1] = self.grid[self.grid.len() - 1];
        Ok(distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_legendre_is_exact_to_degree_31() {
        for k in 0..32 {
            let mut sum = 0.0;
            for (x, w) in GAUSS_LEGENDRE_NODES.iter().zip(&GAUSS_LEGENDRE_WEIGHTS) {
                sum += w * (0.5 * (1.0 - x)).powi(k) + w * (0.5 * (1.0 + x)).powi(k);
            }
            let exact = 1.0 / f64::from(k + 1);
            assert!(
                (0.5 * sum - exact).abs() < 1e-14,
                "degree {k}: {} vs {exact}",
                0.5 * sum
            );
        }
    }
}
