//! `FullAttnCache` — full-attention KV slot. Thin `KVCache` wrapper;
//! `Quantized` arm lands with quantised-KV.

use mlx_rs::{error::Exception, Array};

use super::kvcache::KVCache;
use super::options::CacheOptions;
use super::trait_def::KeyValueCache;

#[derive(Debug, Clone)]
pub enum FullAttnCache {
    Standard(KVCache),
}

impl FullAttnCache {
    pub fn from_options(_opts: CacheOptions) -> Self {
        Self::Standard(KVCache::new())
    }
}

impl Default for FullAttnCache {
    fn default() -> Self {
        Self::Standard(KVCache::new())
    }
}

impl KeyValueCache for FullAttnCache {
    fn is_quantized(&self) -> bool {
        match self {
            Self::Standard(c) => c.is_quantized(),
        }
    }

    fn group_size(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.group_size(),
        }
    }

    fn bits(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.bits(),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Standard(c) => c.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Standard(c) => c.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Standard(c) => c.update_and_fetch(keys, values),
        }
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Standard(c) => c.attention(queries, keys, values, scale, mask),
        }
    }
}
