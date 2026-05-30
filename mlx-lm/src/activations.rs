//! Shared compiled activation helpers.
//!
//! Each function wraps a `transforms::compile`-fused inner kernel in a
//! caller-owned cache so every decoder layer per token reuses one fused
//! graph instead of rebuilding it per call. The cache lives on the owning
//! module struct (e.g. `Mlp::swiglu_cache`) and is borrowed `&mut` per
//! call; mlx core's compile/encoder state is thread-local since mlx 0.31,
//! so the cache must be dropped on the thread that calls it.

// Each `as fn(...) -> ...` below coerces a zero-sized fn-item to a shared
// fn-pointer type. Without the cast every fn-item would yield a distinct
// `Compiled<F, _>` and the cache slot could not be reused. Clippy's
// trivial_casts diagnostic prints identical source/dest types, but the
// source is the fn-item ZST, not a fn-pointer.
#![allow(
    trivial_casts,
    reason = "fn-item ZST → fn-pointer coercion for shared compile cache"
)]

use std::sync::OnceLock;

use mlx_rs::{
    error::Exception,
    nn,
    ops::sigmoid,
    transforms::compile::{allocate_compile_id, shape::TwoArgs, CallMut, Compile, Compiled},
    Array,
};

/// Process-wide cache ids — one slot per logical activation, shared across
/// every cache instance. Lets MLX's `compiler_cache` reuse a single
/// compiled Metal kernel across all decoder layers instead of JIT-compiling
/// one redundant copy per layer.
fn swiglu_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn attention_gate_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}

pub type SwigluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type AttentionGateCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

/// Cached compiled-graph slot for [`swiglu`]. Owned by the calling module
/// (typically a per-layer `Mlp::swiglu_cache`). Initialised lazily on first
/// call. Custom `Debug` is opaque — the inner `Compiled` wraps a
/// `Box<dyn FnMut>` that has no `Debug` impl.
#[derive(Default)]
pub struct SwigluCache(pub Option<SwigluCompiled>);

#[derive(Default)]
pub struct AttentionGateCache(pub Option<AttentionGateCompiled>);

impl std::fmt::Debug for SwigluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwigluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for AttentionGateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionGateCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

/// `silu(gate) * x` as a compile-fused kernel. Caller passes a
/// `&mut SwigluCache` owned by the surrounding module; the compiled graph
/// is built on first call and reused thereafter.
pub fn swiglu(cache: &mut SwigluCache, gate: &Array, x: &Array) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            swiglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            swiglu_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (gate, x))
}

fn swiglu_inner((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

/// `output * sigmoid(gate)` — trailing fused op of Qwen3.5 full-attention.
/// Caller-owned cache, same shape as [`swiglu`].
pub fn attention_gate(
    cache: &mut AttentionGateCache,
    output: &Array,
    gate: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            attention_gate_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            attention_gate_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (output, gate))
}

fn attention_gate_inner((output, gate): (&Array, &Array)) -> Result<Array, Exception> {
    sigmoid(gate)?.multiply(output)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn swiglu_matches_manual_silu_multiply() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5, 2.0], &[2, 2]);
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut cache = SwigluCache::default();
        let fused = swiglu(&mut cache, &gate, &x).unwrap();
        let manual = nn::silu(&gate).unwrap().multiply(&x).unwrap();
        let max = fused
            .subtract(&manual)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max < 1e-5, "fused vs manual swiglu diverge: {max}");
    }

    /// Both activations compile the same `(&Array, &Array)` signature.
    /// Distinct compile ids must keep their graphs separate even when
    /// invoked in sequence with the same shapes — a TypeId-keyed cache
    /// would make `attention_gate` return `sigmoid(output) * gate`
    /// after `swiglu` warmed the slot.
    #[test]
    fn attention_gate_after_swiglu_does_not_collide() {
        let output = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let gate = Array::from_slice(&[0.0_f32, 1.0, -1.0, 2.0], &[2, 2]);

        let mut swiglu_cache = SwigluCache::default();
        let _ = swiglu(&mut swiglu_cache, &gate, &output).unwrap();

        let mut ag_cache = AttentionGateCache::default();
        let fused = attention_gate(&mut ag_cache, &output, &gate).unwrap();
        let manual = sigmoid(&gate).unwrap().multiply(&output).unwrap();
        let max = fused
            .subtract(&manual)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max < 1e-5, "attention_gate diverged after swiglu: {max}");
    }
}
