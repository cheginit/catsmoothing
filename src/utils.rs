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

pub(crate) fn interpolate(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&x_val| {
            if x_val <= xp[0] {
                return fp[0];
            }
            if x_val >= xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }
            let idx = match xp.binary_search_by(|probe| probe.partial_cmp(&x_val).unwrap()) {
                Ok(i) => i,
                Err(after) => after - 1,
            };
            let x0 = xp[idx];
            let x1 = xp[idx + 1];
            let y0 = fp[idx];
            let y1 = fp[idx + 1];
            let t = (x_val - x0) / (x1 - x0);
            y0 + t * (y1 - y0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_interpolate() {
        let xp = vec![0.0, 1.0, 2.0];
        let fp = vec![0.0, 10.0, 20.0];
        let x = vec![0.5, 1.5];
        let result = interpolate(&x, &xp, &fp);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10);
    }
}
