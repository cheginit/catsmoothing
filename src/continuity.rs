//! Curvature-continuous splines built from a C1 Catmull-Rom curve with continuity-enhancing
//! splits (Yuksel, "Continuity-Enhancing Degree Elevation and Splits", SIGGRAPH 2026,
//! Sections 3 and 5). Equation numbers below refer to that paper.

use crate::splines::{Segment, Vec2D};

/// Where each piece is split, as a fraction of its parameter span; the paper's default.
const LAMBDA: f64 = 0.25;

/// Bezier control points of one cubic piece.
type Bezier = [Vec2D; 4];

/// Handle positions that make the joint between `prev` and `next` C2 (or G2) continuous:
/// the new second handle of `prev` and the new third handle of `next`. `alpha` is the ratio
/// of the pieces' parameter spans, `prev` over `next`.
fn joint_handles(prev: &Bezier, next: &Bezier, alpha: f64, geometric: bool) -> (Vec2D, Vec2D) {
    let [_, before_far, before_near, joint] = *prev;
    let [_, after_near, after_far, _] = *next;
    let alpha2 = alpha * alpha;
    let alpha4 = alpha2 * alpha2;
    let d1 = after_near - joint;
    let d1_squared = d1.norm_squared();
    // Eq. 24 picks the free reparameterization term for G2; Eq. 20 with beta = 0 gives C2.
    let beta = if geometric && d1_squared > 0.0 {
        -d1.dot(&((after_far - after_near) * alpha2 - (before_far - before_near))) / d1_squared
    } else {
        -alpha2 - alpha
    };
    // Eqs. 22, 12, and 19.
    let target = after_near + (before_far - before_near - d1 * beta) * (1.0 / alpha2);
    let new_after_far = (after_far + target * alpha4) * (1.0 / (1.0 + alpha4));
    let new_before_far = before_near + (new_after_far - after_near) * alpha2 + d1 * beta;
    (new_before_far, new_after_far)
}

/// Replaces each piece of the C1 curve given by `vertices`, `grid`, and per-vertex
/// `tangents` with three cubic pieces. The joints at vertices become C2 (or G2 when
/// `geometric`), the two new joints inside each piece are C2, and every vertex is still
/// interpolated with its original tangent. Returns the knots and segments.
pub(crate) fn split(
    vertices: &[Vec2D],
    grid: &[f64],
    tangents: &[Vec2D],
    closed: bool,
    geometric: bool,
) -> (Vec<f64>, Vec<Segment>) {
    let n_pieces = vertices.len() - 1;
    let pieces: Vec<Bezier> = (0..n_pieces)
        .map(|i| {
            let third = (grid[i + 1] - grid[i]) / 3.0;
            [
                vertices[i],
                vertices[i] + tangents[i] * third,
                vertices[i + 1] - tangents[i + 1] * third,
                vertices[i + 1],
            ]
        })
        .collect();

    // Each piece's start keeps C2 with the previous piece through a moved third handle, and
    // its end keeps C2 with the next piece through a moved second handle (Section 5).
    let mut start_b2: Vec<Vec2D> = pieces.iter().map(|p| p[2]).collect();
    let mut end_b1: Vec<Vec2D> = pieces.iter().map(|p| p[1]).collect();
    let seam = closed.then_some((n_pieces - 1, 0));
    for (prev, next) in (1..n_pieces).map(|i| (i - 1, i)).chain(seam) {
        let alpha = (grid[prev + 1] - grid[prev]) / (grid[next + 1] - grid[next]);
        let (prev_b1, next_b2) = joint_handles(&pieces[prev], &pieces[next], alpha, geometric);
        end_b1[prev] = prev_b1;
        start_b2[next] = next_b2;
    }

    let (l, m, k) = (LAMBDA, 1.0 - LAMBDA, 1.0 - 2.0 * LAMBDA);
    let c = (2.0 + l * l - 4.0 * l) / m;
    let e = l * k / m;
    let mut knots = Vec::with_capacity(3 * n_pieces + 1);
    let mut segments = Vec::with_capacity(3 * n_pieces);
    knots.push(grid[0]);
    for (i, &[b0, b1, b2, b3]) in pieces.iter().enumerate() {
        // Eqs. 40 to 45: the first and last thirds of the two modified pieces.
        let p1 = b0 * m + b1 * l;
        let p2 = b0 * (m * m) + b1 * (2.0 * m * l) + start_b2[i] * (l * l);
        let r1 = end_b1[i] * (l * l) + b2 * (2.0 * m * l) + b3 * (m * m);
        let r2 = b2 * l + b3 * m;
        // Eqs. 50 to 53: the middle piece that joins them with C2 continuity.
        let q0 = p2 * c - p1 * k + r1 * l - r2 * e;
        let q3 = r1 * c - r2 * k + p2 * l - p1 * e;
        let q1 = q0 * (m / l) - p2 * (k / l);
        let q2 = q3 * (m / l) - r1 * (k / l);
        segments.push(Segment::from_bezier(b0, p1, p2, q0));
        segments.push(Segment::from_bezier(q0, q1, q2, q3));
        segments.push(Segment::from_bezier(q3, r1, r2, b3));
        let span = grid[i + 1] - grid[i];
        knots.extend([grid[i] + l * span, grid[i + 1] - l * span, grid[i + 1]]);
    }
    (knots, segments)
}
