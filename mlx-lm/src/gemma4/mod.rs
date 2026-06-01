//! Gemma 4 family (text). Dense base: sliding/global hybrid attention,
//! four norms per layer, GeGLU MLP, logit soft-capping, tied embeddings.
//! MoE / per-layer-input embeddings / KV-sharing / vision are deferred.

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

#[cfg(feature = "image")]
pub mod image;
pub mod text;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_gemma4()
        .ok_or_else(|| Error::config("gemma4::load_context: not a gemma4 config"))?;
    text::load_context(cfg, env, dir)
}
