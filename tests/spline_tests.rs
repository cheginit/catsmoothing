use catsmoothing::SplineError;
use catsmoothing::{line_tangents, lines_tangents, smooth_linestring, smooth_linestrings};
use catsmoothing::{
    BoundaryCondition, CatmullRom, Continuity, LineToSmooth, Presmoothing, SigmaUnits, Spacing,
    SplineOptions,
};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};

fn fit(
    vertices: Vec<[f64; 2]>,
    grid: Option<Vec<f64>>,
    alpha: Option<f64>,
    bc_type: BoundaryCondition,
    sigma: Option<f64>,
) -> Result<CatmullRom, SplineError> {
    let presmoothing = sigma.map(|sigma| Presmoothing::Gaussian {
        sigma,
        units: SigmaUnits::Vertices,
    });
    CatmullRom::new(
        vertices,
        SplineOptions {
            grid,
            alpha,
            bc_type,
            presmoothing,
            ..SplineOptions::default()
        },
    )
}

/// The options `smooth_linestrings` uses from Python: centripetal, clamped by default.
fn smooth_options(sigma: Option<f64>, bc_type: Option<BoundaryCondition>) -> SplineOptions {
    SplineOptions {
        alpha: Some(0.5),
        bc_type: bc_type.unwrap_or(BoundaryCondition::Clamped),
        presmoothing: sigma.map(|sigma| Presmoothing::Gaussian {
            sigma,
            units: SigmaUnits::Vertices,
        }),
        ..SplineOptions::default()
    }
}

/// Noisy samples of a Gaussian bump, the profile the Python benchmarks use.
fn noisy_bump(rng: &mut StdRng, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|i| {
            let x = -3.0 + 5.5 * i as f64 / (n - 1) as f64;
            let noise: f64 = StandardNormal.sample(rng);
            [x, (-x * x).exp() + 0.1 * noise]
        })
        .collect()
}

fn polyline_length(points: &[[f64; 2]]) -> f64 {
    points
        .windows(2)
        .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
        .sum()
}

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
    let spline = fit(vertices, None, Some(0.0), BoundaryCondition::Closed, None).unwrap();

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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();

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
    let spline = fit(vertices, None, Some(1.0), BoundaryCondition::Closed, None).unwrap();

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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

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

    let angles = lines_tangents(vec![vertices.clone()], vec![None], vec![false]).unwrap();

    // For this case, we expect specific angles based on the Catmull-Rom spline properties
    // First angle should be ~0.983 radians (~56.3° upward)
    assert_close(&[angles[0][0]], &[0.983], 1e-3);
    // Middle angle should be 0 (horizontal tangent at peak)
    assert_close(&[angles[0][1]], &[0.0], 1e-6);
    // Last angle should be ~-0.983 radians (~-56.3° downward)
    assert_close(&[angles[0][2]], &[-0.983], 1e-3);

    // Test with Gaussian smoothing
    let smoothed_angles = line_tangents(
        vertices,
        Some(Presmoothing::Gaussian {
            sigma: 0.5,
            units: SigmaUnits::Vertices,
        }),
        false,
    )
    .unwrap();
    assert_eq!(smoothed_angles.len(), angles[0].len());
}

#[test]
fn test_invalid_input() {
    let vertices = vec![[0.0, 0.0]]; // Only one vertex
    assert!(matches!(
        line_tangents(vertices, None, false),
        Err(SplineError::TooFewVertices)
    ));
}

#[test]
fn test_smooth_linestring_np() {
    let mut rng = StdRng::seed_from_u64(123);
    let vertices = noisy_bump(&mut rng, 50);
    let line_length: f64 = polyline_length(&vertices);

    let smoothed = smooth_linestring(
        vertices,
        Spacing::Count(30),
        smooth_options(Some(2.0), None),
    )
    .unwrap();

    // Check that smoothed line has 30 points
    assert_eq!(smoothed.len(), 30);

    // Check that length of smoothed line is close to input length
    let smoothed_length: f64 = polyline_length(&smoothed);
    assert_close(&[line_length], &[smoothed_length], 2.8);
}

#[test]
fn test_smooth_linestring_dist() {
    let mut rng = StdRng::seed_from_u64(123);
    let vertices = noisy_bump(&mut rng, 50);
    let orig_length: f64 = polyline_length(&vertices);

    let smoothed = smooth_linestring(
        vertices,
        Spacing::MaxDistance(0.2),
        smooth_options(Some(2.0), None),
    )
    .unwrap();

    // The fewest evenly spaced points with chords no longer than the distance.
    let chords: Vec<f64> = smoothed
        .windows(2)
        .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
        .collect();
    let n_spans = chords.len() as f64;
    let arc_span = chords.iter().sum::<f64>() / n_spans;
    assert!(chords.iter().all(|&c| c <= 0.2 + 1e-12));
    assert!(arc_span * n_spans / (n_spans - 1.0) > 0.2);

    // Check that length of smoothed line is close to input length
    let smoothed_length: f64 = polyline_length(&smoothed);
    assert_close(&[orig_length], &[smoothed_length], 2.8);
}

#[test]
fn test_smooth_linestrings() {
    let mut rng = StdRng::seed_from_u64(123);
    let lines: Vec<Vec<[f64; 2]>> = (0..3).map(|_| noisy_bump(&mut rng, 50)).collect();

    let line_lengths: Vec<f64> = lines.iter().map(|v| polyline_length(v)).collect();

    let spacings = [
        Spacing::Count(30),
        Spacing::MaxDistance(0.2),
        Spacing::Count(40),
    ];
    let options = [
        smooth_options(Some(2.0), Some(BoundaryCondition::Clamped)),
        smooth_options(None, Some(BoundaryCondition::Natural)),
        smooth_options(Some(1.5), None),
    ];
    let jobs: Vec<LineToSmooth> = lines
        .into_iter()
        .zip(spacings)
        .zip(options)
        .map(|((vertices, spacing), options)| LineToSmooth {
            vertices,
            spacing,
            options,
        })
        .collect();
    let smoothed_lines = smooth_linestrings(jobs).unwrap();

    assert_eq!(smoothed_lines[0].len(), 30);
    assert_eq!(smoothed_lines[2].len(), 40);
    // The distance-spaced line keeps every chord within the requested spacing.
    assert!(smoothed_lines[1]
        .windows(2)
        .all(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]) <= 0.2 + 1e-12));

    // Check that the smoothed line lengths are close to the original lengths
    for (smoothed, &orig_length) in smoothed_lines.iter().zip(&line_lengths) {
        let smoothed_length: f64 = polyline_length(smoothed);
        assert_close(&[orig_length], &[smoothed_length], 2.9);
    }
}

// ======================== Regression tests ========================

#[test]
fn test_evaluate_returns_error_on_invalid_derivative_order() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();
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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let distances = vec![spline.grid()[0]];
    let result = spline.evaluate(&distances, 2).unwrap();
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
}

#[test]
fn test_evaluate_empty_distances() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let result = spline.evaluate(&[], 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_evaluate_boundary_distances() {
    // Test that distances at/beyond grid boundaries are handled correctly
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    // n_pts < 2 should return a single grid point
    let result = spline.uniform_distances(1).unwrap();
    assert_eq!(result.len(), 1);
    assert_close(&result, &[spline.grid()[0]], 1e-12);
}

#[test]
fn test_first_derivative_at_known_curve() {
    // For a symmetric parabola-like curve, the tangent at the midpoint should be horizontal
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

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
        let spline = fit(vertices.clone(), None, Some(0.5), bc_type, None).unwrap();
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
    let spline = fit(vertices, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();

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

    let spline = fit(
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
        let spline = fit(vec![[0.0, 0.0], [2.0, 1.0]], None, Some(0.5), bc_type, None).unwrap();
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
    let explicit = fit(
        vertices.clone(),
        Some(vec![0.0, 2.0, 5.0, 5.5]),
        None,
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    let centripetal = fit(vertices, None, Some(0.5), BoundaryCondition::Clamped, None).unwrap();
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
    let spline = fit(
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
    // sigma = 2 reaches 8 vertices on each side of a 3-vertex line, so reflections repeat.
    let sigma = 2.0_f64;
    let spline = fit(
        vec![[0.0, 0.0], [1.0, 10.0], [2.0, 0.0]],
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        Some(sigma),
    )
    .unwrap();
    let middle = spline.evaluate(&[spline.grid()[1]], 0).unwrap()[0];

    // Point reflection through both ends extends x = k and y = 10 sin(pi k / 2).
    let radius = (4.0 * sigma).ceil() as i64;
    let (mut x, mut y, mut total) = (0.0, 0.0, 0.0);
    for k in -radius..=radius {
        let w = (-((k * k) as f64) / (2.0 * sigma * sigma)).exp();
        x += w * (1 + k) as f64;
        y += w * 10.0 * (std::f64::consts::FRAC_PI_2 * (1 + k) as f64).sin();
        total += w;
    }
    assert_close(&middle, &[x / total, y / total], 1e-12);
}

#[test]
fn test_repeated_vertices_are_dropped() {
    let with_repeats = vec![[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 0.0]];
    let unique = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let a = fit(
        with_repeats,
        None,
        Some(0.5),
        BoundaryCondition::Clamped,
        None,
    )
    .unwrap();
    let b = fit(unique, None, Some(0.5), BoundaryCondition::Clamped, None).unwrap();
    assert_eq!(a.grid(), b.grid());
    let d: Vec<f64> = (0..11).map(|i| b.grid()[2] * i as f64 / 10.0).collect();
    assert_eq!(a.evaluate(&d, 0).unwrap(), b.evaluate(&d, 0).unwrap());

    let all_same = vec![[1.0, 1.0]; 3];
    assert_eq!(
        fit(all_same, None, Some(0.5), BoundaryCondition::Natural, None).unwrap_err(),
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
        let spline = fit(shifted, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();
        assert_eq!(spline.grid().len(), 5, "offset {offset}");
    }
    // An explicitly repeated closing vertex is not doubled.
    let mut closed = ring.to_vec();
    closed.push(ring[0]);
    let spline = fit(closed, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();
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
        let spline = fit(vertices.clone(), None, Some(0.5), bc_type, None).unwrap();
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
        assert!(
            worst < 1e-6,
            "{bc_type:?}: relative spacing deviation {worst:.2e}"
        );
    }
}

#[test]
fn test_gaussian_smoothing_keeps_straight_lines() {
    // Unevenly spaced collinear vertices stay on their line.
    let vertices: Vec<[f64; 2]> = [0.0, 0.3, 1.0, 1.2, 2.5, 3.0, 4.6, 5.0]
        .iter()
        .map(|&t| [t, 2.0 * t + 1.0])
        .collect();
    let spline = fit(
        vertices,
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        Some(2.0),
    )
    .unwrap();
    for p in spline.evaluate(spline.grid(), 0).unwrap() {
        assert!(
            (p[1] - (2.0 * p[0] + 1.0)).abs() < 1e-12,
            "{p:?} left the line"
        );
    }

    // Evenly spaced vertices do not move at all, even next to the ends.
    let even: Vec<[f64; 2]> = (0..9).map(|i| [i as f64, 2.0 * i as f64 + 1.0]).collect();
    let spline = fit(
        even.clone(),
        None,
        Some(0.5),
        BoundaryCondition::Natural,
        Some(2.0),
    )
    .unwrap();
    for (p, v) in spline.evaluate(spline.grid(), 0).unwrap().iter().zip(&even) {
        assert_close(p, v, 1e-12);
    }
}

#[test]
fn test_gaussian_smoothing_treats_ring_seam_like_other_vertices() {
    let n = 12;
    let ring: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            [a.cos(), a.sin()]
        })
        .collect();
    let spline = fit(ring, None, Some(0.5), BoundaryCondition::Closed, Some(1.0)).unwrap();
    let radii: Vec<f64> = spline
        .evaluate(spline.grid(), 0)
        .unwrap()
        .iter()
        .map(|p| p[0].hypot(p[1]))
        .collect();
    for r in &radii {
        assert!((r - radii[0]).abs() < 1e-12, "radii differ: {radii:?}");
    }
    assert!(radii[0] < 1.0, "Gaussian smoothing shrinks a ring");
}

/// Signed curvature at each distance.
fn curvature(spline: &CatmullRom, distances: &[f64]) -> Vec<f64> {
    let d1 = spline.evaluate(distances, 1).unwrap();
    let d2 = spline.evaluate(distances, 2).unwrap();
    d1.iter()
        .zip(&d2)
        .map(|(v, a)| (v[0] * a[1] - v[1] * a[0]) / v[0].hypot(v[1]).powi(3))
        .collect()
}

#[test]
fn test_continuity_splits() {
    let mut rng = StdRng::seed_from_u64(11);
    let vertices: Vec<[f64; 2]> = (0..30)
        .map(|i| {
            let noise: f64 = StandardNormal.sample(&mut rng);
            let x = f64::from(i) * 0.2;
            [x, x.sin() + 0.05 * noise]
        })
        .collect();
    for bc_type in [
        BoundaryCondition::Natural,
        BoundaryCondition::Clamped,
        BoundaryCondition::Closed,
    ] {
        let build = |continuity| {
            let options = SplineOptions {
                alpha: Some(0.5),
                bc_type,
                continuity,
                ..SplineOptions::default()
            };
            CatmullRom::new(vertices.clone(), options).unwrap()
        };
        let c1 = build(Continuity::C1);
        let grid = c1.grid().to_vec();
        let mean_span = (grid[grid.len() - 1] - grid[0]) / (grid.len() - 1) as f64;
        let eps = 1e-6 * mean_span;
        // Joints at vertices, and for the splits the two knots inside each piece.
        let vertex_joints: Vec<f64> = if bc_type == BoundaryCondition::Closed {
            grid.clone()
        } else {
            grid[1..grid.len() - 1].to_vec()
        };
        let inner_knots: Vec<f64> = grid
            .windows(2)
            .flat_map(|w| [w[0] + 0.25 * (w[1] - w[0]), w[1] - 0.25 * (w[1] - w[0])])
            .collect();
        let jump = |spline: &CatmullRom, at: f64| -> f64 {
            // The seam of a closed curve joins the last segment to the first.
            let (left, right) = if bc_type == BoundaryCondition::Closed && at == grid[0] {
                (grid[grid.len() - 1] - eps, at + eps)
            } else if bc_type == BoundaryCondition::Closed && at == grid[grid.len() - 1] {
                (at - eps, grid[0] + eps)
            } else {
                (at - eps, at + eps)
            };
            let k = curvature(spline, &[left, right]);
            (k[0] - k[1]).abs()
        };
        let c1_worst = vertex_joints
            .iter()
            .map(|&g| jump(&c1, g))
            .fold(0.0, f64::max);
        assert!(
            c1_worst > 1.0,
            "{bc_type:?}: C1 control should jump in curvature"
        );

        for continuity in [Continuity::G2, Continuity::C2] {
            let spline = build(continuity);
            assert_eq!(spline.grid(), &grid[..]);
            // Same vertices and vertex tangents as the C1 curve.
            let at_vertices = spline.evaluate(&grid, 0).unwrap();
            for (p, q) in at_vertices.iter().zip(&c1.evaluate(&grid, 0).unwrap()) {
                assert_close(p, q, 1e-12);
            }
            for (p, q) in spline
                .evaluate(&grid, 1)
                .unwrap()
                .iter()
                .zip(&c1.evaluate(&grid, 1).unwrap())
            {
                assert_close(p, q, 1e-9);
            }
            // The enhanced curve stays near the C1 shape (measured: G2 2.5%, C2 4.1%).
            let dense: Vec<f64> = (0..=600)
                .map(|i| grid[0] + (grid[grid.len() - 1] - grid[0]) * f64::from(i) / 600.0)
                .collect();
            let drift = spline
                .evaluate(&dense, 0)
                .unwrap()
                .iter()
                .zip(&c1.evaluate(&dense, 0).unwrap())
                .map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
                .fold(0.0, f64::max);
            let mean_chord = c1.length() / (grid.len() - 1) as f64;
            assert!(
                drift < 0.1 * mean_chord,
                "{bc_type:?} {continuity:?}: drift {drift}"
            );
            let worst = vertex_joints
                .iter()
                .chain(&inner_knots)
                .map(|&g| jump(&spline, g))
                .fold(0.0, f64::max);
            assert!(
                worst < 1e-3,
                "{bc_type:?} {continuity:?}: curvature jump {worst}"
            );
            if continuity == Continuity::C2 {
                // Second derivatives match on both sides of every joint.
                for &g in vertex_joints.iter().chain(&inner_knots) {
                    if bc_type == BoundaryCondition::Closed
                        && (g == grid[0] || g == grid[grid.len() - 1])
                    {
                        continue;
                    }
                    let d2 = spline.evaluate(&[g - eps, g + eps], 2).unwrap();
                    let scale = 1.0 + d2[0][0].hypot(d2[0][1]);
                    assert!(
                        (d2[0][0] - d2[1][0]).hypot(d2[0][1] - d2[1][1]) < 1e-4 * scale,
                        "{bc_type:?}: second derivative jumps at {g}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_last_vertex_is_returned_exactly() {
    // At projected-CRS magnitudes, summing the last segment's coefficients misses the end
    // vertex by rounding, which leaves closed rings slightly open.
    let ring: Vec<[f64; 2]> = (0..30)
        .map(|i| {
            let a = std::f64::consts::TAU * f64::from(i) / 30.0;
            [4.0e6 + 10.0 * a.cos(), 5.0e5 + 13.0 * a.sin()]
        })
        .collect();
    for continuity in [Continuity::C1, Continuity::G2] {
        let options = SplineOptions {
            alpha: Some(0.5),
            bc_type: BoundaryCondition::Closed,
            continuity,
            ..SplineOptions::default()
        };
        let spline = CatmullRom::new(ring.clone(), options).unwrap();
        let ends = spline
            .evaluate(&[spline.grid()[0], *spline.grid().last().unwrap()], 0)
            .unwrap();
        assert_eq!(ends[0], ring[0]);
        assert_eq!(ends[1], ring[0]);
    }
}

#[test]
fn test_alpha_overrides_grid_and_still_drops_repeats() {
    let vertices = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let spline = fit(
        vertices,
        Some(vec![0.0, 1.0, 2.0, 3.0]),
        Some(0.5),
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    assert_eq!(spline.grid(), &[0.0, 1.0, 2.0]);
    let distances = spline.uniform_distances(5).unwrap();
    assert!(distances.iter().all(|d| d.is_finite()));

    let overflow = fit(
        vec![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
        None,
        Some(400.0),
        BoundaryCondition::Natural,
        None,
    );
    assert_eq!(overflow.unwrap_err(), SplineError::InvalidGrid);
}

#[test]
fn test_uniform_distances_across_a_reversal() {
    // With a uniform grid, the curve doubles back between the two close middle vertices, so
    // the speed passes through zero inside a segment.
    let spline = fit(
        vec![[0.0, 0.0], [1.0, 0.0], [1.01, 0.0], [2.0, 0.0]],
        None,
        None,
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    let distances = spline.uniform_distances(101).unwrap();
    let spans: Vec<f64> = distances
        .windows(2)
        .map(|w| sampled_arc_length(&spline, w[0], w[1]))
        .collect();
    let mean = spans.iter().sum::<f64>() / spans.len() as f64;
    let worst = spans
        .iter()
        .map(|s| (s - mean).abs() / mean)
        .fold(0.0, f64::max);
    assert!(worst < 1e-6, "relative spacing deviation {worst:.2e}");
}

#[test]
fn test_overflowing_lengths_fail_fast() {
    let spline = fit(
        vec![[0.0, 0.0], [1e155, 0.0], [2e155, 0.0]],
        None,
        None,
        BoundaryCondition::Natural,
        None,
    )
    .unwrap();
    assert!(!spline.length().is_finite());
    assert_eq!(
        spline.uniform_distances(3).unwrap_err(),
        SplineError::NonFiniteLength
    );
    let smoothed = smooth_linestring(
        vec![[0.0, 0.0], [1e155, 0.0], [2e155, 0.0]],
        Spacing::MaxDistance(1.0),
        SplineOptions::default(),
    );
    assert_eq!(smoothed.unwrap_err(), SplineError::NonFiniteLength);
}
