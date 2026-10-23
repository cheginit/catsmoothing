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

/// The result of a Gaussian at least one period of the extended line wide, where the wrapped
/// kernel is flat to about 3e-9 (its first Fourier term is `exp(-2 pi^2)`): open lines become
/// evenly spaced points on the chord at `fractions` of the way, and closed rings collapse to
/// their centroid weighted by `spans`. Returning it directly bounds the work for any sigma.
fn flat_kernel_limit(
    vertices: &[Vec2D],
    fractions: &[f64],
    spans: &[f64],
    closed: bool,
) -> Vec<Vec2D> {
    let n_verts = vertices.len();
    if closed {
        let (mut sum, mut weight) = (Vec2D::new(0.0, 0.0), 0.0);
        for (v, &w) in vertices[..n_verts - 1].iter().zip(spans) {
            sum = sum + *v * w;
            weight += w;
        }
        return vec![sum * (1.0 / weight); n_verts];
    }
    let (first, last) = (vertices[0], vertices[n_verts - 1]);
    let mut limit: Vec<Vec2D> = fractions
        .iter()
        .map(|&f| first + (last - first) * f)
        .collect();
    limit[n_verts - 1] = last;
    limit
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
    let period = if closed { total } else { 2.0 * total };
    if sigma >= period {
        let fractions: Vec<f64> = positions.iter().map(|s| s / total).collect();
        let spans: Vec<f64> = (0..n_verts)
            .map(|k| {
                let before = if k > 0 {
                    positions[k] - positions[k - 1]
                } else {
                    total - positions[n_verts - 2]
                };
                let after = if k + 1 < n_verts {
                    positions[k + 1] - positions[k]
                } else {
                    positions[1]
                };
                0.5 * (before + after)
            })
            .collect();
        return flat_kernel_limit(vertices, &fractions, &spans, closed);
    }
    let reach = 4.0 * sigma;

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
        let (center, own) = sample(i);
        let (mut sum, mut weight) = (Vec2D::new(0.0, 0.0), 0.0);
        for step in [1, -1] {
            let mut k = if step > 0 { i } else { i - 1 };
            loop {
                let (s, v) = sample(k);
                if (s - center).abs() > reach {
                    break;
                }
                let span = 0.5 * (sample(k + 1).0 - sample(k - 1).0);
                // Normalizing first keeps tiny sigmas from forming 0 * inf.
                let z = (s - center) / sigma;
                let w = (-0.5 * z * z).exp() * span;
                sum = sum + v * w;
                weight += w;
                k += step;
            }
        }
        if weight > 0.0 {
            sum * (1.0 / weight)
        } else {
            own
        }
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
    let period = if closed {
        n_verts - 1
    } else {
        2 * (n_verts - 1)
    };
    if sigma >= period as f64 {
        let fractions: Vec<f64> = (0..n_verts)
            .map(|i| i as f64 / (n_verts - 1) as f64)
            .collect();
        return flat_kernel_limit(vertices, &fractions, &vec![1.0; n_verts], closed);
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

/// Taubin lambda/mu smoothing (Taubin 1995). Each iteration moves every vertex toward the
/// midpoint of its neighbors by `LAMBDA`, then away from it by `|MU|`, which damps jitter
/// without the shrinkage of repeated averaging. Open lines keep their end vertices; closed
/// rings wrap around.
pub(crate) fn taubin(vertices: &[Vec2D], iterations: usize, closed: bool) -> Vec<Vec2D> {
    const LAMBDA: f64 = 0.5;
    const MU: f64 = -0.53;
    let n_verts = vertices.len();
    let mut current = vertices.to_vec();
    if n_verts <= 2 {
        return current;
    }
    let mut next = current.clone();
    for _ in 0..iterations {
        for factor in [LAMBDA, MU] {
            if closed {
                let n_ring = n_verts - 1;
                for i in 0..n_ring {
                    let left = current[(i + n_ring - 1) % n_ring];
                    let right = current[(i + 1) % n_ring];
                    next[i] = current[i] + ((left + right) * 0.5 - current[i]) * factor;
                }
                next[n_ring] = next[0];
            } else {
                for i in 1..n_verts - 1 {
                    let midpoint = (current[i - 1] + current[i + 1]) * 0.5;
                    next[i] = current[i] + (midpoint - current[i]) * factor;
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taubin_damps_jitter_without_shrinking() {
        // Alternating jitter is the highest frequency, which the first lambda step removes at
        // every interior vertex. The pinned, jittered end vertices then leak inward one vertex
        // per half step, so only vertices more than ten steps from either end are checked.
        let line: Vec<Vec2D> = (0..41)
            .map(|i| Vec2D::new(f64::from(i), if i % 2 == 0 { 0.1 } else { -0.1 }))
            .collect();
        let smoothed = taubin(&line, 5, false);
        let worst = smoothed[11..30]
            .iter()
            .map(|p| p.y.abs())
            .fold(0.0, f64::max);
        assert!(worst < 0.01, "jitter left: {worst}");

        // A smooth ring keeps its radius, where Gaussian averaging would shrink it.
        let n = 64;
        let mut ring: Vec<Vec2D> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * f64::from(i) / f64::from(n);
                Vec2D::new(a.cos(), a.sin())
            })
            .collect();
        ring.push(ring[0]);
        for p in taubin(&ring, 10, true) {
            assert!((p.norm() - 1.0).abs() < 0.005, "radius {}", p.norm());
        }
        let shrunk = gaussian(&ring, 2.0, true)[0].norm();
        assert!(shrunk < 0.99, "Gaussian reference should shrink: {shrunk}");
    }

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
    fn test_length_kernel_with_tiny_sigma_keeps_vertices() {
        let line = vec![
            Vec2D::new(0.0, 0.0),
            Vec2D::new(1.0, 1.0),
            Vec2D::new(0.0, 0.0),
        ];
        assert_eq!(gaussian_by_length(&line, 1e-200, false), line);
    }

    #[test]
    fn test_huge_sigma_reaches_the_flat_limit_quickly() {
        let line = vec![
            Vec2D::new(0.0, 0.0),
            Vec2D::new(1.0, 2.0),
            Vec2D::new(3.0, 1.0),
            Vec2D::new(4.0, 4.0),
        ];
        let chord = |f: f64| Vec2D::new(4.0 * f, 4.0 * f);
        let by_vertex = gaussian(&line, 1e9, false);
        for (i, p) in by_vertex.iter().enumerate() {
            assert!((*p - chord(i as f64 / 3.0)).norm() < 1e-12);
        }
        let by_length = gaussian_by_length(&line, 1e9, false);
        let (short, long) = (5.0_f64.sqrt(), 10.0_f64.sqrt());
        let total = 2.0 * short + long;
        for (p, f) in by_length
            .iter()
            .zip([0.0, short / total, 2.0 * short / total, 1.0])
        {
            assert!((*p - chord(f)).norm() < 1e-12, "{p:?} vs fraction {f}");
        }
        // The flat limit agrees with the direct sum just below the threshold.
        let below = gaussian(&line, 5.999, false);
        let at = gaussian(&line, 6.0, false);
        for (a, b) in below.iter().zip(&at) {
            assert!((*a - *b).norm() < 1e-3, "{a:?} vs {b:?}");
        }
        // Rings collapse to their centroid.
        let mut ring = line.clone();
        ring.push(line[0]);
        let center = gaussian_by_length(&ring, 1e9, true)[0];
        assert!(gaussian(&ring, 1e9, true)
            .iter()
            .all(|p| (*p - Vec2D::new(2.0, 1.75)).norm() < 1e-12));
        assert!(center.x.is_finite() && center.y.is_finite());
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
