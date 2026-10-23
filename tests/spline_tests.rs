use catsmoothing::SplineError;
use catsmoothing::{line_tangents, lines_tangents, smooth_linestring, smooth_linestrings};
use catsmoothing::{BoundaryCondition, CatmullRom};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "Arrays must have same length");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < tol,
            "Expected {y} but got {x}, diff too large: {}",
            (x - y).abs()
        );
    }
}

fn mean(values: &[[f64; 2]]) -> [f64; 2] {
    let sum = values
        .iter()
        .fold([0.0, 0.0], |acc, &x| [acc[0] + x[0], acc[1] + x[1]]);
    let n = values.len() as f64;
    [sum[0] / n, sum[1] / n]
}

#[test]
fn test_spline_alpha_0() {
    let vertices = vec![
        [0.0, 0.0],
        [0.0, 0.5],
        [1.5, 1.5],
        [1.6, 1.5],
        [3.0, 0.2],
        [3.0, 0.0],
    ];
    let n_pts = 15.0;
    let spline =
        CatmullRom::new(vertices, None, Some(0.0), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid()[0];
    let grid_end = spline.grid().last().unwrap();
    let dots = ((grid_end - grid_start) * n_pts) as usize + 1;
    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0).unwrap();
    let mean_point = mean(&points);

    assert_close(&[mean_point[0]], &[1.5], 1e-3);
    assert_close(&[mean_point[1]], &[0.6099], 1e-3);
}

#[test]
fn test_spline_alpha_0_5() {
    let vertices = vec![
        [0.0, 0.0],
        [0.0, 0.5],
        [1.5, 1.5],
        [1.6, 1.5],
        [3.0, 0.2],
        [3.0, 0.0],
    ];
    let n_pts = 15.0;
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid()[0];
    let grid_end = spline.grid().last().unwrap();
    let dots = ((grid_end - grid_start) * n_pts) as usize + 1;
    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0).unwrap();
    let mean_point = mean(&points);

    assert_close(&mean_point, &[1.4570, 0.5289], 1e-3);
}

#[test]
fn test_spline_alpha_1() {
    let vertices = vec![
        [0.0, 0.0],
        [0.0, 0.5],
        [1.5, 1.5],
        [1.6, 1.5],
        [3.0, 0.2],
        [3.0, 0.0],
    ];
    let n_pts = 15.0;
    let spline =
        CatmullRom::new(vertices, None, Some(1.0), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid()[0];
    let grid_end = *spline.grid().last().unwrap();
    let dots = ((grid_end - grid_start) * n_pts).floor() as usize + 1;
    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0).unwrap();
    let mean_point = mean(&points);

    assert_close(&mean_point, &[1.4754, 0.3844], 1e-3);
}

#[test]
fn test_uniform_distances() {
    // Simple curved path
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let distances = spline.uniform_distances(5).unwrap();

    // Check count and monotonicity
    assert_eq!(distances.len(), 5);
    for i in 1..distances.len() {
        assert!(distances[i] > distances[i - 1]);
    }

    // Equal spacing by arc length along the curve
    let spans: Vec<f64> = distances
        .windows(2)
        .map(|w| sampled_arc_length(&spline, w[0], w[1]))
        .collect();
    let mean = spans.iter().sum::<f64>() / spans.len() as f64;
    for span in spans {
        assert!((span - mean).abs() / mean < 1e-4, "{span} vs {mean}");
    }
}

#[test]
fn test_compute_tangent_angles() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];

    let angles = lines_tangents(vec![vertices.clone()], vec![None]).unwrap();

    // For this case, we expect specific angles based on the Catmull-Rom spline properties
    // First angle should be ~0.983 radians (~56.3° upward)
    assert_close(&[angles[0][0]], &[0.983], 1e-3);
    // Middle angle should be 0 (horizontal tangent at peak)
    assert_close(&[angles[0][1]], &[0.0], 1e-6);
    // Last angle should be ~-0.983 radians (~-56.3° downward)
    assert_close(&[angles[0][2]], &[-0.983], 1e-3);

    // Test with Gaussian smoothing
    let smoothed_angles = line_tangents(vertices, Some(0.5)).unwrap();
    assert_eq!(smoothed_angles.len(), angles[0].len());
}

#[test]
fn test_invalid_input() {
    let vertices = vec![[0.0, 0.0]]; // Only one vertex
    assert!(matches!(
        line_tangents(vertices, None),
        Err(SplineError::TooFewVertices)
    ));
}

#[test]
fn test_smooth_linestring_np() {
    let mut rng = StdRng::seed_from_u64(123);
    let vertices: Vec<[f64; 2]> = (0..50)
        .map(|i| {
            let x = -3.0 + i as f64 * (2.5 - (-3.0)) / 49.0;
            let noise: f64 = StandardNormal.sample(&mut rng);
            let y = (-x.powi(2)).exp() + 0.1 * noise;
            [x, y]
        })
        .collect();
    let line_length: f64 = vertices
        .windows(2)
        .map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();

    let smoothed = smooth_linestring(vertices, None, Some(30), Some(2.0), None).unwrap();

    // Check that smoothed line has 30 points
    assert_eq!(smoothed.len(), 30);

    // Check that length of smoothed line is close to input length
    let smoothed_length: f64 = smoothed
        .windows(2)
        .map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();
    assert_close(&[line_length], &[smoothed_length], 2.8);
}

#[test]
fn test_smooth_linestring_dist() {
    let mut rng = StdRng::seed_from_u64(123);
    let vertices: Vec<[f64; 2]> = (0..50)
        .map(|i| {
            let x = -3.0 + i as f64 * (2.5 - (-3.0)) / 49.0;
            let noise: f64 = StandardNormal.sample(&mut rng);
            let y = (-x.powi(2)).exp() + 0.1 * noise;
            [x, y]
        })
        .collect();
    let orig_length: f64 = vertices
        .windows(2)
        .map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();

    let smoothed = smooth_linestring(vertices, Some(0.2), None, Some(2.0), None).unwrap();

    // Check that smoothed line has 86 points
    assert_eq!(smoothed.len(), 86);

    // Check that length of smoothed line is close to input length
    let smoothed_length: f64 = smoothed
        .windows(2)
        .map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();
    assert_close(&[orig_length], &[smoothed_length], 2.8);
}

#[test]
fn test_smooth_linestrings() {
    let mut rng = StdRng::seed_from_u64(123);
    let lines: Vec<Vec<[f64; 2]>> = (0..3)
        .map(|_| {
            (0..50)
                .map(|i| {
                    let x = -3.0 + i as f64 * (2.5 - (-3.0)) / 49.0;
                    let noise: f64 = StandardNormal.sample(&mut rng);
                    let y = (-x.powi(2)).exp() + 0.1 * noise;
                    [x, y]
                })
                .collect()
        })
        .collect();

    let line_lengths: Vec<f64> = lines
        .iter()
        .map(|vertices| {
            vertices
                .windows(2)
                .map(|w| {
                    let dx = w[1][0] - w[0][0];
                    let dy = w[1][1] - w[0][1];
                    (dx * dx + dy * dy).sqrt()
                })
                .sum()
        })
        .collect();

    // Define per-line parameters
    let distances = vec![None, Some(0.2), None];
    let n_pts_list = vec![Some(30), None, Some(40)];
    let gaussian_sigmas = vec![Some(2.0), None, Some(1.5)];
    let bc_types = vec![
        Some(BoundaryCondition::Clamped),
        Some(BoundaryCondition::Natural),
        None,
    ];
    let smoothed_lines =
        smooth_linestrings(lines, distances, n_pts_list, gaussian_sigmas, bc_types).unwrap();

    // Check that the smoothed lines have the expected number of points
    for (smoothed, expected_len) in smoothed_lines.iter().zip([30, 104, 40]) {
        assert_eq!(smoothed.len(), expected_len);
    }

    // Check that the smoothed line lengths are close to the original lengths
    for (smoothed, &orig_length) in smoothed_lines.iter().zip(&line_lengths) {
        let smoothed_length: f64 = smoothed
            .windows(2)
            .map(|w| {
                let dx = w[1][0] - w[0][0];
                let dy = w[1][1] - w[0][1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        assert_close(&[orig_length], &[smoothed_length], 2.9);
    }
}

// ======================== Regression tests ========================

#[test]
fn test_evaluate_returns_error_on_invalid_derivative_order() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();
    let distances = vec![0.0, 0.5, 1.0];

    // Valid orders succeed
    assert!(spline.evaluate(&distances, 0).is_ok());
    assert!(spline.evaluate(&distances, 1).is_ok());
    assert!(spline.evaluate(&distances, 2).is_ok());

    // Invalid order returns error instead of panicking
    assert!(matches!(
        spline.evaluate(&distances, 3),
        Err(SplineError::UnsupportedDerivativeOrder)
    ));
    assert!(matches!(
        spline.evaluate(&distances, 99),
        Err(SplineError::UnsupportedDerivativeOrder)
    ));
}

#[test]
fn test_second_derivative_varies_with_parameter() {
    // For a cubic Hermite spline, the second derivative of position should
    // vary linearly with the parameter t (since d²/dt²(t³) = 6t).
    // Test that evaluating at different distances produces different results.
    let vertices = vec![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let grid_end = *spline.grid().last().unwrap();
    let distances: Vec<f64> = (0..20).map(|i| grid_end * (i as f64) / 19.0).collect();
    let second_derivs = spline.evaluate(&distances, 2).unwrap();

    // The second derivative should not be constant across the spline
    // (which was the bug before: it was returning [6.0, 2.0, 0.0, 0.0] for all t)
    let first = second_derivs[0];
    let has_variation = second_derivs
        .iter()
        .any(|pt| (pt[0] - first[0]).abs() > 1e-10 || (pt[1] - first[1]).abs() > 1e-10);
    assert!(
        has_variation,
        "Second derivative should vary across the spline, not be constant"
    );
}

#[test]
fn test_second_derivative_at_segment_start_is_finite() {
    // At t=0, the second derivative is [6*0, 2, 0, 0] dot coeffs = [0, 2, 0, 0] dot coeffs
    // This should produce finite values
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let distances = vec![spline.grid()[0]];
    let result = spline.evaluate(&distances, 2).unwrap();
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
}

#[test]
fn test_evaluate_empty_distances() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let result = spline.evaluate(&[], 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_evaluate_boundary_distances() {
    // Test that distances at/beyond grid boundaries are handled correctly
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let grid_start = spline.grid()[0];
    let grid_end = *spline.grid().last().unwrap();

    // Test at exact boundaries and beyond
    let distances = vec![
        grid_start - 1.0, // before start
        grid_start,       // at start
        grid_end,         // at end
        grid_end + 1.0,   // beyond end
    ];
    let result = spline.evaluate(&distances, 0).unwrap();
    assert_eq!(result.len(), 4);
    for pt in &result {
        assert!(pt[0].is_finite());
        assert!(pt[1].is_finite());
    }
}

#[test]
fn test_uniform_distances_single_point() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    // n_pts < 2 should return a single grid point
    let result = spline.uniform_distances(1).unwrap();
    assert_eq!(result.len(), 1);
    assert_close(&result, &[spline.grid()[0]], 1e-12);
}

#[test]
fn test_first_derivative_at_known_curve() {
    // For a symmetric parabola-like curve, the tangent at the midpoint should be horizontal
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    // Evaluate first derivative at the midpoint
    let mid = vec![spline.grid()[1]];
    let tangent = spline.evaluate(&mid, 1).unwrap();

    // At the peak, dy should be ~0 (horizontal tangent)
    assert!(
        tangent[0][1].abs() < 1e-6,
        "Tangent y-component at peak should be ~0, got {}",
        tangent[0][1]
    );
}

#[test]
fn test_three_point_spline_all_boundary_conditions() {
    // Regression: splines should work with Natural and Clamped boundary conditions
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];

    for bc_type in [BoundaryCondition::Natural, BoundaryCondition::Clamped] {
        let spline = CatmullRom::new(vertices.clone(), None, Some(0.5), bc_type, None).unwrap();
        let grid_end = *spline.grid().last().unwrap();
        let distances = vec![0.0, grid_end / 2.0, grid_end];
        let result = spline.evaluate(&distances, 0).unwrap();
        assert_eq!(result.len(), 3);

        // Start and end points should match input vertices
        assert_close(&result[0], &[0.0, 0.0], 1e-10);
        assert_close(&result[2], &[2.0, 0.0], 1e-10);
    }
}

#[test]
fn test_closed_spline_wraps_correctly() {
    // A closed square should produce a smooth loop
    let vertices = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0], // duplicate first vertex
    ];
    let spline =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();

    let grid_end = *spline.grid().last().unwrap();
    let distances: Vec<f64> = (0..40).map(|i| grid_end * (i as f64) / 39.0).collect();
    let points = spline.evaluate(&distances, 0).unwrap();

    // First and last points should be very close (closed curve)
    let first = points.first().unwrap();
    let last = points.last().unwrap();
    assert_close(first, last, 1e-6);
}

#[test]
fn test_gaussian_smoothing_preserves_endpoints() {
    let vertices = vec![[0.0, 0.0], [0.5, 1.0], [1.0, 0.5], [1.5, 1.5], [2.0, 0.0]];

    let spline = CatmullRom::new(
        vertices.clone(),
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        Some(1.0),
    )
    .unwrap();

    // Evaluate at grid start and end
    let distances = vec![spline.grid()[0], *spline.grid().last().unwrap()];
    let points = spline.evaluate(&distances, 0).unwrap();

    // Endpoints should be the original vertices (Gaussian smoothing preserves endpoints)
    assert_close(&points[0], &vertices[0], 1e-6);
    assert_close(&points[1], &vertices[vertices.len() - 1], 1e-6);
}

#[test]
fn test_two_vertex_open_line_is_straight() {
    for bc_type in [BoundaryCondition::Natural, BoundaryCondition::Clamped] {
        let spline =
            CatmullRom::new(vec![[0.0, 0.0], [2.0, 1.0]], None, Some(0.5), bc_type, None).unwrap();
        let end = *spline.grid().last().unwrap();
        let points = spline
            .evaluate(&[0.0, end / 4.0, end / 2.0, end], 0)
            .unwrap();
        for (point, expected) in
            points
                .iter()
                .zip([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5], [2.0, 1.0]])
        {
            assert_close(point, &expected, 1e-12);
        }
    }
}

#[test]
fn test_derivatives_match_finite_differences_on_non_unit_grid() {
    let vertices = vec![[0.0, 0.0], [1.0, 2.0], [4.0, 0.0], [5.0, 3.0]];
    let explicit = CatmullRom::new(
        vertices.clone(),
        Some(vec![0.0, 2.0, 5.0, 5.5]),
        None,
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    let centripetal =
        CatmullRom::new(vertices, None, Some(0.5), BoundaryCondition::Clamped, None).unwrap();
    for spline in [explicit, centripetal] {
        let h = 1e-5;
        let grid = spline.grid();
        for w in grid.windows(2) {
            for frac in [0.25, 0.5, 0.75] {
                let t = w[0] + frac * (w[1] - w[0]);
                let p = spline.evaluate(&[t - h, t, t + h], 0).unwrap();
                let d1 = spline.evaluate(&[t], 1).unwrap()[0];
                let d2 = spline.evaluate(&[t], 2).unwrap()[0];
                for k in 0..2 {
                    let fd1 = (p[2][k] - p[0][k]) / (2.0 * h);
                    let fd2 = (p[2][k] - 2.0 * p[1][k] + p[0][k]) / (h * h);
                    assert!(
                        (d1[k] - fd1).abs() < 1e-6,
                        "first derivative {} vs {fd1} at t={t}",
                        d1[k]
                    );
                    assert!(
                        (d2[k] - fd2).abs() < 1e-3,
                        "second derivative {} vs {fd2} at t={t}",
                        d2[k]
                    );
                }
            }
        }
    }
}

#[test]
fn test_uniform_distances_respects_grid_origin() {
    let vertices = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let spline = CatmullRom::new(
        vertices,
        Some(vec![10.0, 11.0, 12.0]),
        None,
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    let distances = spline.uniform_distances(3).unwrap();
    assert_close(&distances, &[10.0, 11.0, 12.0], 1e-9);
}

#[test]
fn test_gaussian_smoothing_kernel_wider_than_line() {
    // sigma = 2 reaches 8 vertices on each side of a 3-vertex line, so the mirror must repeat.
    let sigma = 2.0_f64;
    let spline = CatmullRom::new(
        vec![[0.0, 0.0], [1.0, 10.0], [2.0, 0.0]],
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        Some(sigma),
    )
    .unwrap();
    let middle = spline.evaluate(&[spline.grid()[1]], 0).unwrap()[0];

    // Reference: mirror sequence for index 1 + k with period 4 visits x = [0, 1, 2, 1].
    let radius = (4.0 * sigma).ceil() as i64;
    let (mut x, mut y, mut total) = (0.0, 0.0, 0.0);
    for k in -radius..=radius {
        let w = (-((k * k) as f64) / (2.0 * sigma * sigma)).exp();
        let m = (1 + k).rem_euclid(4);
        x += w * [0.0, 1.0, 2.0, 1.0][m as usize];
        y += w * [0.0, 10.0, 0.0, 10.0][m as usize];
        total += w;
    }
    assert_close(&middle, &[x / total, y / total], 1e-12);
}

#[test]
fn test_repeated_vertices_are_dropped() {
    let with_repeats = vec![[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 0.0]];
    let unique = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let a = CatmullRom::new(
        with_repeats,
        None,
        Some(0.5),
        BoundaryCondition::Clamped,
        None,
    )
    .unwrap();
    let b = CatmullRom::new(unique, None, Some(0.5), BoundaryCondition::Clamped, None).unwrap();
    assert_eq!(a.grid(), b.grid());
    let d: Vec<f64> = (0..11).map(|i| b.grid()[2] * i as f64 / 10.0).collect();
    assert_eq!(a.evaluate(&d, 0).unwrap(), b.evaluate(&d, 0).unwrap());

    let all_same = vec![[1.0, 1.0]; 3];
    assert_eq!(
        CatmullRom::new(all_same, None, Some(0.5), BoundaryCondition::Natural, None).unwrap_err(),
        SplineError::TooFewVertices
    );
}

#[test]
fn test_closed_ring_vertices_survive_translation() {
    let ring = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 5.0]];
    for offset in [0.0, 1e6, 4e6] {
        let shifted: Vec<[f64; 2]> = ring
            .iter()
            .map(|p| [p[0] + offset, p[1] + offset])
            .collect();
        let spline =
            CatmullRom::new(shifted, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();
        assert_eq!(spline.grid().len(), 5, "offset {offset}");
    }
    // An explicitly repeated closing vertex is not doubled.
    let mut closed = ring.to_vec();
    closed.push(ring[0]);
    let spline = CatmullRom::new(closed, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();
    assert_eq!(spline.grid().len(), 5);
}

/// Arc length of the curve between two grid parameters, by dense polyline sampling.
fn sampled_arc_length(spline: &CatmullRom, from: f64, to: f64) -> f64 {
    let n = 20_000;
    let d: Vec<f64> = (0..=n)
        .map(|i| from + (to - from) * i as f64 / n as f64)
        .collect();
    let p = spline.evaluate(&d, 0).unwrap();
    p.windows(2)
        .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
        .sum()
}

#[test]
fn test_uniform_distances_equal_arc_length_on_noisy_line() {
    let mut rng = StdRng::seed_from_u64(7);
    let vertices: Vec<[f64; 2]> = (0..100)
        .map(|i| {
            let noise: f64 = StandardNormal.sample(&mut rng);
            let x = i as f64 / 99.0 * 5.5 - 3.0;
            [x, (-x * x).exp() + 0.1 * noise]
        })
        .collect();
    for bc_type in [
        BoundaryCondition::Natural,
        BoundaryCondition::Clamped,
        BoundaryCondition::Closed,
    ] {
        let spline = CatmullRom::new(vertices.clone(), None, Some(0.5), bc_type, None).unwrap();
        let distances = spline.uniform_distances(60).unwrap();
        assert_eq!(distances[0], spline.grid()[0]);
        assert_eq!(distances[59], *spline.grid().last().unwrap());
        let spans: Vec<f64> = distances
            .windows(2)
            .map(|w| sampled_arc_length(&spline, w[0], w[1]))
            .collect();
        let mean = spans.iter().sum::<f64>() / spans.len() as f64;
        let mut worst: f64 = 0.0;
        assert!((mean * 59.0 - spline.length()).abs() < 1e-6 * spline.length());
        for span in spans {
            worst = worst.max((span - mean).abs() / mean);
        }
        // 16-point Gauss-Legendre is accurate to about 1e-5 on sharply bent segments.
        assert!(
            worst < 1e-4,
            "{bc_type:?}: relative spacing deviation {worst:.2e}"
        );
    }
}
