// Runs in its own test binary so no other test initializes the global pool first.
use catsmoothing::{
    line_tangents, lines_tangents, smooth_linestring, smooth_linestrings, CatmullRom, LineToSmooth,
    Spacing, SplineOptions,
};

fn wave(phase: f64) -> Vec<[f64; 2]> {
    (0..40)
        .map(|i| {
            let x = f64::from(i) * 0.25;
            [x, (x + phase).sin()]
        })
        .collect()
}

#[test]
fn test_single_thread_pool_runs_batches_serially_in_order() {
    std::env::set_var("RAYON_NUM_THREADS", "1");
    assert_eq!(rayon::current_num_threads(), 1);
    let lines: Vec<Vec<[f64; 2]>> = (0..5).map(|k| wave(f64::from(k))).collect();
    let options = SplineOptions {
        alpha: Some(0.5),
        ..SplineOptions::default()
    };

    let batch = lines
        .iter()
        .map(|vertices| LineToSmooth {
            vertices: vertices.clone(),
            spacing: Spacing::Count(30),
            options: options.clone(),
        })
        .collect();
    let expected: Vec<_> = lines
        .iter()
        .map(|vertices| smooth_linestring(vertices.clone(), Spacing::Count(30), options.clone()))
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(smooth_linestrings(batch).unwrap(), expected);

    let tangents = lines_tangents(lines.clone(), vec![None; 5], vec![false; 5]).unwrap();
    for (vertices, angles) in lines.iter().zip(&tangents) {
        assert_eq!(
            *angles,
            line_tangents(vertices.clone(), None, false).unwrap()
        );
    }

    // Above the parallel threshold, one thread evaluates serially to the same points.
    let spline = CatmullRom::new(lines[0].clone(), options).unwrap();
    let distances: Vec<f64> = (0..5000).map(|i| f64::from(i) * 1e-3).collect();
    let chunked: Vec<[f64; 2]> = distances
        .chunks(1000)
        .flat_map(|chunk| spline.evaluate(chunk, 0).unwrap())
        .collect();
    assert_eq!(spline.evaluate(&distances, 0).unwrap(), chunked);
}
