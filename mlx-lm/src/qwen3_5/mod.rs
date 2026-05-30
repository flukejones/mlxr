//! Qwen3.5 family: hybrid full-attention + gated-delta-net linear
//! attention. Dense text path (MoE + vision land in later commits).

pub mod text;

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_qwen35()
        .ok_or_else(|| Error::config("qwen3_5::load_context: not a qwen3.5 config"))?;
    text::load_context_dense(cfg, env, dir)
}
