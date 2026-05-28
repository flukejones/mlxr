//! Shared post-load helpers.

/// MLX cache-pool cap applied after every weight load (20 MB, matches
/// mlx-swift LLM guidance). Override via `set_cache_limit_override` or
/// the `MLX_LM_CACHE_LIMIT_BYTES` env var.
pub const DEFAULT_CACHE_LIMIT_BYTES: usize = 20 * 1024 * 1024;

static CACHE_LIMIT_OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// First call wins. `0` disables reuse entirely.
pub fn set_cache_limit_override(bytes: usize) {
    let _ = CACHE_LIMIT_OVERRIDE.set(bytes);
}

/// Precedence: `set_cache_limit_override` > `MLX_LM_CACHE_LIMIT_BYTES` env > [`DEFAULT_CACHE_LIMIT_BYTES`].
fn resolved_cache_limit() -> usize {
    if let Some(&n) = CACHE_LIMIT_OVERRIDE.get() {
        return n;
    }
    if let Some(n) = parse_env_bytes("MLX_LM_CACHE_LIMIT_BYTES") {
        return n;
    }
    DEFAULT_CACHE_LIMIT_BYTES
}

/// Drain the MLX cache pool then apply the resolved cap. Reclaims the
/// safetensors scratch buffers parked in the reuse pool after load.
pub fn apply_post_load_memory_policy() {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::set_cache_limit(resolved_cache_limit());
}

fn parse_env_bytes(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<usize>().ok()
}
