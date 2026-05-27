pub mod llama;
pub mod qwen3;

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

use crate::{cache::KeyValueCache, nn::ModelInput};

/// One decode step: reshape `last_id` to `[B, 1]`, forward, slice the
/// last position, sample. The single per-token unit `Generate` and the
/// bench share, so measurement can't drift from production.
pub fn decode_step<M, C>(
    model: &mut M,
    cache: &mut Vec<Option<C>>,
    last_id: &Array,
    temp: f32,
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
    sample(&logits.index((.., -1, ..)), temp)
}

/// Greedy `argmax` at temp 0, else temperature-scaled categorical.
fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}
