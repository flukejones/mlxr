//! Qwen3.5 family: hybrid full-attention + gated-delta-net linear
//! attention. Dense + MoE text paths (vision lands in a later commit).

pub mod text;

use std::path::Path;

use crate::config::{Family, ModelConfig};
use crate::error::Error;
use crate::family::LoadedContext;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_qwen35()
        .ok_or_else(|| Error::config("qwen3_5::load_context: not a qwen3.5 config"))?;
    match &cfg.family {
        Family::Qwen35Moe(_) => text::adapter_moe::load_context_moe(cfg, env, dir),
        _ => text::load_context_dense(cfg, env, dir),
    }
}
