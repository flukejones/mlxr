pub mod llama;
pub mod qwen3;

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    transforms::async_eval,
    Array,
};

use crate::{cache::KeyValueCache, nn::ModelInput};

/// Cache `1.0 / temp` once for the decode loop; `None` for greedy (temp 0).
pub fn inv_temp(temp: f32) -> Option<Array> {
    (temp != 0.0).then(|| array!(1.0 / temp))
}

/// One pipelined decode step: reshape `last_id` to `[B, 1]`, forward,
/// slice the last position, sample, `async_eval` the result so N+1 GPU
/// compute overlaps the caller's sync on N. The single per-token unit
/// `Generate` and the bench share, so measurement can't drift.
pub fn decode_step<M, C>(
    model: &mut M,
    cache: &mut Vec<Option<C>>,
    last_id: &Array,
    inv_temp: Option<&Array>,
) -> Result<Array, Exception>
where
    M: for<'a> Module<ModelInput<'a, C>, Output = Array, Error = Exception>,
    C: KeyValueCache + Default,
{
    let inputs = last_id.index((.., NewAxis));
    let logits = model.forward(ModelInput {
        inputs: &inputs,
        mask: None,
        cache,
    })?;
    let next = sample_logits(&logits.index((.., -1, ..)), inv_temp)?;
    async_eval([&next])?;
    Ok(next)
}

/// Greedy `argmax` when `inv_temp` is `None`, else categorical over
/// `logits * inv_temp` (caller caches `inv_temp = 1/temp`).
pub fn sample_logits(logits: &Array, inv_temp: Option<&Array>) -> Result<Array, Exception> {
    match inv_temp {
        None => argmax_axis!(logits, -1),
        Some(inv_temp) => categorical!(&logits.multiply(inv_temp)?),
    }
}
