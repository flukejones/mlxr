//! KV-cache implementations for decoder-only models.
//!
//! - [`trait_def`] — the [`KeyValueCache`] trait + blanket `&mut T` impl
//! - [`kvcache`] — [`KVCache`], the default pre-allocated step-grown cache
//! - [`quantized_kvcache`] — [`QuantizedKVCache`], affine-quant K/V with
//!   independent `k_bits`/`v_bits`
//! - [`full_attn`] — [`FullAttnCache`], the shared full-attention slot
//! - [`options`] — [`CacheOptions`] / [`CacheKind`] + prefill-chunk helpers

pub mod full_attn;
pub mod kvcache;
pub mod options;
pub mod quantized_kvcache;
pub mod trait_def;

pub use full_attn::FullAttnCache;
pub use kvcache::{KVCache, DEFAULT_KV_CACHE_INIT_CAPACITY};
pub use options::{
    CacheKind, CacheOptions, DEFAULT_KV_GROUP_SIZE, DEFAULT_PREFILL_CHUNK, MIN_K_BITS,
};
pub use quantized_kvcache::QuantizedKVCache;
pub use trait_def::KeyValueCache;
