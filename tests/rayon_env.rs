// Runs in its own test binary so no other test initializes the global pool first.
use catsmoothing::{CatmullRom, SplineOptions};

#[test]
fn test_rayon_num_threads_is_honored() {
    std::env::set_var("RAYON_NUM_THREADS", "2");
    let spline = CatmullRom::new(
        vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        SplineOptions::default(),
    )
    .unwrap();
    spline.evaluate(&[0.0, 0.5, 1.0], 0).unwrap();
    assert_eq!(rayon::current_num_threads(), 2);
}
