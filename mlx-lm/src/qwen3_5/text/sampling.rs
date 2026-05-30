//! Qwen3.5-family-local sampling helper for MTP rejection sampling.
//!
//! The MTP rejection path applies one vocab-positional top-p keep mask
//! to both the draft and verify distributions. Co-located with the MoE
//! adapter since no other family consumes it.

use mlx_rs::{
    ops::{argsort_axis, cumsum, indexing::take_along_axis, softmax_axis},
    Array,
};

use crate::error::Error;

/// Vocab-positional top-p keep mask (`[1, vocab]` bool): slot `i` is
/// `true` iff token id `i` is in the smallest descending-probability
/// set whose preceding cumulative mass is below `p` — the same set
/// `crate::sampler::top_p_sample` keeps, indexed by vocab id.
pub(crate) fn top_p_keep_mask(logits: &Array, p: f32) -> Result<Array, Error> {
    let probs = softmax_axis(logits, -1, true)?;
    let order = argsort_axis(&probs.negative()?, -1)?;
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let csum = cumsum(&sorted_probs, -1, false, false)?;
    let prev = csum.subtract(&sorted_probs)?;
    let keep_sorted = prev.lt(Array::from_f32(p))?;
    // argsort(order) is the inverse permutation: maps each vocab id to
    // its sort position, so the keep flags land back in vocab order.
    let inverse = argsort_axis(&order, -1)?;
    Ok(take_along_axis(&keep_sorted, &inverse, -1)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    #[test]
    fn keeps_only_top_token_at_half() {
        let logits = Array::from_slice(&[-10.0_f32, 5.0, -10.0], &[1, 3]);
        let mask = top_p_keep_mask(&logits, 0.5).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[false, true, false]);
    }

    #[test]
    fn keeps_all_at_p_one() {
        let logits = Array::from_slice(&[0.1_f32, 0.5, 0.3, 0.05, 0.05], &[1, 5]);
        let mask = top_p_keep_mask(&logits, 1.0).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[true, true, true, true, true]);
    }
}
