/// Whether the last vertex repeats the first, up to rounding relative to the ring's extent.
/// The tolerance scales with the ring so translating coordinates never changes the answer.
pub(crate) fn ends_coincide(vertices: &[[f64; 2]]) -> bool {
    const RELATIVE_TOLERANCE: f64 = 1e-9;
    let (Some(first), Some(last)) = (vertices.first(), vertices.last()) else {
        return false;
    };
    let (mut min, mut max) = (*first, *first);
    for v in vertices {
        for k in 0..2 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    let extent = (max[0] - min[0]).hypot(max[1] - min[1]);
    (last[0] - first[0]).hypot(last[1] - first[1]) <= RELATIVE_TOLERANCE * extent
}
