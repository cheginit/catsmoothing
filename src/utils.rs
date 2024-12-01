pub(crate) fn are_points_close(a: &[f64; 2], b: &[f64; 2]) -> bool {
    const RTOL: f64 = 1e-5;
    const ATOL: f64 = 1e-8;

    a.iter().zip(b.iter()).all(|(x, y)| {
        let abs_diff = (x - y).abs();
        abs_diff <= (ATOL + RTOL * y.abs())
    })
}

pub(crate) fn interpolate(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&x_val| {
            // Handle edge cases
            if x_val <= xp[0] {
                return fp[0];
            }
            if x_val >= xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }

            // Find index through binary search
            let idx = match xp.binary_search_by(|probe| probe.partial_cmp(&x_val).unwrap()) {
                Ok(exact) => exact,
                Err(after) => after - 1,
            };

            // Linear interpolation
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
