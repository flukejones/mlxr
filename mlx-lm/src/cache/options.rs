//! `CacheOptions` — KV-cache backing + per-cache toggles.

/// Backing kind for full-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheKind {
    #[default]
    Dense,
    Quantized {
        group_size: i32,
        bits: i32,
    },
}

/// Default prefill chunk cap when neither user nor cache imposes one.
pub const DEFAULT_PREFILL_CHUNK: i32 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct CacheOptions {
    pub kind: CacheKind,
    /// Max tokens per prefill forward. `None` = single-pass.
    pub max_prefill_chunk: Option<i32>,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            kind: CacheKind::Dense,
            max_prefill_chunk: Some(DEFAULT_PREFILL_CHUNK),
        }
    }
}
