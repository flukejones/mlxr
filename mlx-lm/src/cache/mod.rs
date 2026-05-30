//! KV-cache implementations for decoder-only models.
//!
//! - [`trait_def`] — the [`KeyValueCache`] trait + blanket `&mut T` impl
//! - [`kvcache`] — [`KVCache`], the default pre-allocated step-grown cache
//! - [`full_attn`] — [`FullAttnCache`], the shared full-attention slot
//! - [`options`] — [`CacheOptions`] / [`CacheKind`] + prefill-chunk helpers

pub mod full_attn;
pub mod kvcache;
pub mod options;
pub mod trait_def;

pub use full_attn::FullAttnCache;
pub use kvcache::{KVCache, DEFAULT_KV_CACHE_INIT_CAPACITY};
pub use options::{CacheKind, CacheOptions, DEFAULT_PREFILL_CHUNK};
pub use trait_def::KeyValueCache;
