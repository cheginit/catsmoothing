//! Vertex smoothing applied before a spline is fitted.

use crate::splines::Vec2D;

/// Normalized Gaussian weights for offsets `-radius..=radius`, truncated at 4 sigma.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma).ceil() as usize;
    let mut kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    for w in &mut kernel {
        *w /= total;
    }
    kernel
}

/// Vertex `k` of an open line extended past both ends by point reflection
/// (`x[-k] = 2 x[0] - x[k]`, and likewise at the far end), which keeps straight runs straight.
/// Repeated reflections cover kernels wider than the line.
fn point_reflected(vertices: &[Vec2D], k: isize) -> Vec2D {
    let last = vertices.len() - 1;
    let period = 2 * last as isize;
    let (turns, m) = (k.div_euclid(period), k.rem_euclid(period) as usize);
    let base = if m <= last {
        vertices[m]
    } else {
        vertices[last] * 2.0 - vertices[2 * last - m]
    };
    base + (vertices[last] - vertices[0]) * (2.0 * turns as f64)
}

/// Gaussian smoothing with `sigma` measured in vertices. Open lines keep their end vertices;
/// closed rings (last vertex repeating the first) wrap around, so the seam is smoothed too.
pub(crate) fn gaussian(vertices: &[Vec2D], sigma: f64, closed: bool) -> Vec<Vec2D> {
    let n_verts = vertices.len();
    if n_verts <= 2 || sigma <= 1e-6 {
        return vertices.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as isize;

    if closed {
        let ring = &vertices[..n_verts - 1];
        let n_ring = ring.len() as isize;
        let mut smoothed: Vec<Vec2D> = (0..n_ring)
            .map(|i| {
                let mut sum = Vec2D::new(0.0, 0.0);
                for (j, &w) in kernel.iter().enumerate() {
                    let idx = (i + j as isize - radius).rem_euclid(n_ring) as usize;
                    sum = sum + ring[idx] * w;
                }
                sum
            })
            .collect();
        smoothed.push(smoothed[0]);
        return smoothed;
    }

    let mut smoothed = Vec::with_capacity(n_verts);
    smoothed.push(vertices[0]);
    for i in 1..n_verts as isize - 1 {
        let mut sum = Vec2D::new(0.0, 0.0);
        for (j, &w) in kernel.iter().enumerate() {
            sum = sum + point_reflected(vertices, i + j as isize - radius) * w;
        }
        smoothed.push(sum);
    }
    smoothed.push(vertices[n_verts - 1]);
    smoothed
}
