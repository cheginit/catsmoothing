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

/// Gaussian smoothing with `sigma` in coordinate length along the line. Each vertex is
/// weighted by the length of line it stands for (half of its two neighboring segments), which
/// approximates a convolution along the line, so uneven vertex spacing does not bias the
/// result. Ends and rings are handled as in [`gaussian`].
pub(crate) fn gaussian_by_length(vertices: &[Vec2D], sigma: f64, closed: bool) -> Vec<Vec2D> {
    let n_verts = vertices.len();
    let mut positions = Vec::with_capacity(n_verts);
    let mut total = 0.0;
    positions.push(total);
    for pair in vertices.windows(2) {
        total += (pair[1] - pair[0]).norm();
        positions.push(total);
    }
    if n_verts <= 2 || sigma <= 0.0 || total <= 0.0 {
        return vertices.to_vec();
    }
    let reach = 4.0 * sigma;
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    // Position along the line and point of sample `k` on the extended line.
    let sample = |k: isize| -> (f64, Vec2D) {
        if closed {
            let n_ring = (n_verts - 1) as isize;
            let (turns, m) = (k.div_euclid(n_ring), k.rem_euclid(n_ring) as usize);
            (positions[m] + total * turns as f64, vertices[m])
        } else {
            let last = n_verts - 1;
            let period = 2 * last as isize;
            let (turns, m) = (k.div_euclid(period), k.rem_euclid(period) as usize);
            let (s, v) = if m <= last {
                (positions[m], vertices[m])
            } else {
                (
                    2.0 * total - positions[2 * last - m],
                    vertices[last] * 2.0 - vertices[2 * last - m],
                )
            };
            let drift = (vertices[last] - vertices[0]) * (2.0 * turns as f64);
            (s + 2.0 * total * turns as f64, v + drift)
        }
    };
    let smooth = |i: isize| -> Vec2D {
        let center = sample(i).0;
        let (mut sum, mut weight) = (Vec2D::new(0.0, 0.0), 0.0);
        for step in [1, -1] {
            let mut k = if step > 0 { i } else { i - 1 };
            loop {
                let (s, v) = sample(k);
                if (s - center).abs() > reach {
                    break;
                }
                let span = 0.5 * (sample(k + 1).0 - sample(k - 1).0);
                let w = (-(s - center) * (s - center) * inv_two_sigma2).exp() * span;
                sum = sum + v * w;
                weight += w;
                k += step;
            }
        }
        sum * (1.0 / weight)
    };

    if closed {
        let mut smoothed: Vec<Vec2D> = (0..n_verts as isize - 1).map(smooth).collect();
        smoothed.push(smoothed[0]);
        return smoothed;
    }
    let mut smoothed = Vec::with_capacity(n_verts);
    smoothed.push(vertices[0]);
    smoothed.extend((1..n_verts as isize - 1).map(smooth));
    smoothed.push(vertices[n_verts - 1]);
    smoothed
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(xs: &[f64]) -> Vec<Vec2D> {
        xs.iter().map(|&x| Vec2D::new(x, x.sin())).collect()
    }

    #[test]
    fn test_length_kernel_ignores_vertex_density() {
        // The same curve, once evenly sampled and once with four times as many vertices
        // on its second half. Shared vertices should smooth to nearly the same place.
        let coarse: Vec<f64> = (0..=40).map(|i| 0.25 * f64::from(i)).collect();
        let mut dense: Vec<f64> = (0..=20).map(|i| 0.25 * f64::from(i)).collect();
        dense.extend((1..=80).map(|i| 5.0 + 0.0625 * f64::from(i)));
        let (coarse_pts, dense_pts) = (sine(&coarse), sine(&dense));
        let a = gaussian_by_length(&coarse_pts, 0.5, false);
        let b = gaussian_by_length(&dense_pts, 0.5, false);
        let by_vertex = gaussian(&dense_pts, 2.0, false);
        let coarse_by_vertex = gaussian(&coarse_pts, 2.0, false);
        let mut worst_length: f64 = 0.0;
        let mut worst_vertex: f64 = 0.0;
        for (i, &x) in coarse.iter().enumerate() {
            let j = dense.iter().position(|&d| (d - x).abs() < 1e-12).unwrap();
            worst_length = worst_length.max((a[i] - b[j]).norm());
            worst_vertex = worst_vertex.max((coarse_by_vertex[i] - by_vertex[j]).norm());
        }
        assert!(worst_length < 0.01, "length kernel moved by {worst_length}");
        assert!(
            worst_vertex > 0.1,
            "vertex kernel should depend on density: {worst_vertex}"
        );
    }

    #[test]
    fn test_length_kernel_shrinks_circle_by_known_factor() {
        // A periodic Gaussian of width sigma shrinks a circle of radius r by exp(-sigma^2 / 2r^2).
        let n = 720;
        let mut ring: Vec<Vec2D> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * f64::from(i) / f64::from(n);
                Vec2D::new(a.cos(), a.sin())
            })
            .collect();
        ring.push(ring[0]);
        let sigma = 0.3;
        let expected = (-sigma * sigma / 2.0_f64).exp();
        for p in gaussian_by_length(&ring, sigma, true) {
            assert!(
                (p.norm() - expected).abs() < 1e-4,
                "{} vs {expected}",
                p.norm()
            );
        }
    }
}
