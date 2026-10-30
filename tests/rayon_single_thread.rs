// Runs in its own test binary so no other test initializes the global pool first.
use catsmoothing::{
    lines_tangents, smooth_linestrings, CatmullRom, LineToSmooth, Spacing, SplineOptions,
};
use std::sync::mpsc;
use std::time::Duration;

type Outputs = (Vec<Vec<[f64; 2]>>, Vec<Vec<f64>>, Vec<[f64; 2]>);

/// Calls each entry point that dispatches to Rayon on the same lines.
fn run_all(lines: Vec<Vec<[f64; 2]>>, options: SplineOptions) -> Outputs {
    let batch = lines
        .iter()
        .map(|vertices| LineToSmooth {
            vertices: vertices.clone(),
            spacing: Spacing::Count(30),
            options: options.clone(),
        })
        .collect();
    let spline = CatmullRom::new(lines[0].clone(), options).expect("40 distinct vertices fit");
    // Above the parallel threshold, so only the one-thread check keeps it serial.
    let distances: Vec<f64> = (0..5000).map(|i| f64::from(i) * 1e-3).collect();
    let n = lines.len();
    (
        smooth_linestrings(batch).expect("every line has 40 vertices"),
        lines_tangents(lines, vec![None; n], vec![false; n]).expect("one flag per line"),
        spline
            .evaluate(&distances, 1)
            .expect("first derivatives are supported"),
    )
}

/// Runs `f` while the global pool's only worker is busy, so `f` finishes only if it never
/// hands work to the pool. Returns `None` when it does not finish in time.
fn with_worker_blocked<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    rayon::spawn(move || {
        started_tx.send(()).expect("the test waits for this signal");
        release_rx.recv().ok();
    });
    started_rx
        .recv()
        .expect("the blocking job runs on the worker");
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || done_tx.send(f()).ok());
    let result = done_rx.recv_timeout(Duration::from_secs(10)).ok();
    release_tx
        .send(())
        .expect("the blocking job waits for release");
    result
}

#[test]
fn test_single_thread_pool_runs_on_the_calling_thread() {
    std::env::set_var("RAYON_NUM_THREADS", "1");
    assert_eq!(rayon::current_num_threads(), 1);
    let lines: Vec<Vec<[f64; 2]>> = (0..5)
        .map(|k| {
            (0..40)
                .map(|i| {
                    let x = f64::from(i) * 0.25;
                    [x, (x + f64::from(k)).sin()]
                })
                .collect()
        })
        .collect();
    let options = SplineOptions {
        alpha: Some(0.5),
        ..SplineOptions::default()
    };

    let serial = with_worker_blocked({
        let (lines, options) = (lines.clone(), options.clone());
        move || run_all(lines, options)
    })
    .expect("one-thread calls never wait on the pool");
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| run_all(lines, options));
    assert_eq!(serial, parallel);
}
