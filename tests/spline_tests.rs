use catsmoothing::{
    line_tangents, lines_tangents, smooth_linestring, smooth_linestrings, BoundaryCondition,
    CatmullRomRust, SplineError,
};
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
        CatmullRomRust::new(vertices, None, Some(0.0), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid[0];
    let grid_end = spline.grid.last().unwrap();
    let dots = ((grid_end - grid_start) * n_pts) as usize + 1;

    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0);
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
        CatmullRomRust::new(vertices, None, Some(0.5), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid[0];
    let grid_end = spline.grid.last().unwrap();
    let dots = ((grid_end - grid_start) * n_pts) as usize + 1;

    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0);
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
        CatmullRomRust::new(vertices, None, Some(1.0), BoundaryCondition::Closed, None).unwrap();

    let grid_start = spline.grid[0];
    let grid_end = *spline.grid.last().unwrap();

    let dots = ((grid_end - grid_start) * n_pts).floor() as usize + 1;
    let distances: Vec<f64> = (0..dots).map(|i| grid_start + (i as f64) / n_pts).collect();

    let points = spline.evaluate(&distances, 0);
    let mean_point = mean(&points);

    assert_close(&mean_point, &[1.4754, 0.3844], 1e-3);
}

#[test]
fn test_compute_tangent_angles() {
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];

    let angles = lines_tangents(vec![vertices.clone()], vec![None]).unwrap();

    // For this case, we expect specific angles based on the Catmull-Rom spline properties
    // First angle should be ~0.983 radians (~56.3 degrees, upward)
    assert_close(&[angles[0][0]], &[0.983], 1e-3);

    // Middle angle should be 0 (horizontal tangent at peak)
    assert_close(&[angles[0][1]], &[0.0], 1e-6);

    // Last angle should be ~-0.983 radians (~-56.3 degrees, downward)
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
fn test_uniform_distances() {
    // Create a simple curved path
    let vertices = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];

    let spline =
        CatmullRomRust::new(vertices, None, Some(0.5), BoundaryCondition::Natural, None).unwrap();

    let distances = spline.uniform_distances(5, 1e-6, 10);

    // Check number of points
    assert_eq!(distances.len(), 5);

    // Check that distances are monotonically increasing
    for i in 1..distances.len() {
        assert!(distances[i] > distances[i - 1]);
    }

    // Evaluate points at these distances
    let points = spline.evaluate(&distances, 0);

    // Check that consecutive points are roughly equidistant
    let diffs: Vec<f64> = points
        .windows(2)
        .map(|window| {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();

    let avg_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;

    // Check that all segments are within 1% of average length
    for diff in diffs {
        assert!((diff - avg_diff).abs() / avg_diff < 0.01);
    }
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
        .map(|window| {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();

    let smoothed =
        smooth_linestring(vertices, None, Some(30), Some(2.0), None, None, None).unwrap();

    // Check that smoothed line has same length as input
    assert_eq!(smoothed.len(), 30);

    // Check that length of smoothed line is close to length of input
    let smoothed_length: f64 = smoothed
        .windows(2)
        .map(|window| {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
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
        .map(|window| {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum();

    let smoothed =
        smooth_linestring(vertices, Some(0.2), None, Some(2.0), None, None, None).unwrap();

    // Check that smoothed line has same length as input
    assert_eq!(smoothed.len(), 86);

    // Check that length of smoothed line is close to length of input
    let smoothed_length: f64 = smoothed
        .windows(2)
        .map(|window| {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
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
                .map(|window| {
                    let dx = window[1][0] - window[0][0];
                    let dy = window[1][1] - window[0][1];
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
    let tolerance = Some(1e-6);
    let max_iterations = Some(100);

    // Smooth the lines
    let smoothed_lines = smooth_linestrings(
        lines,
        distances,
        n_pts_list,
        gaussian_sigmas,
        bc_types,
        tolerance,
        max_iterations,
    )
    .unwrap();

    // Check that the smoothed lines have the correct lengths
    for (smoothed, expected_len) in smoothed_lines.iter().zip([30, 104, 40]) {
        assert_eq!(smoothed.len(), expected_len);
    }

    // Check that the lengths of the smoothed lines are close to the original
    for (smoothed, &original_length) in smoothed_lines.iter().zip(&line_lengths) {
        let smoothed_length: f64 = smoothed
            .windows(2)
            .map(|window| {
                let dx = window[1][0] - window[0][0];
                let dy = window[1][1] - window[0][1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        assert_close(&[original_length], &[smoothed_length], 2.9);
    }
}
