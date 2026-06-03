//! Gemma 4 Unified family (`gemma4_unified`): encoder-free multimodal 12B.
//!
//! Text milestone: dense decoder (reusing [`crate::gemma4::text`]) + MTP
//! speculative decode (reusing [`crate::gemma4::mtp`]). Encoder-free vision
//! and audio front-ends land in follow-on milestones.

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

pub mod adapter;
pub mod config;

pub(crate) fn load_context(
    cfg: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    let env = cfg.family.as_gemma4_unified().ok_or_else(|| {
        Error::config("gemma4_unified::load_context: not a gemma4_unified config")
    })?;
    adapter::load_context(cfg, env, dir, draft_dir)
}
