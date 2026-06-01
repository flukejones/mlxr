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
#[cfg(feature = "image")]
pub mod vlm;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_gemma4()
        .ok_or_else(|| Error::config("gemma4::load_context: not a gemma4 config"))?;

    // A checkpoint carrying `vision_config` + `processor_config.json` is a
    // VLM; route to the multimodal adapter when the `image` feature is on,
    // else load text-only (tower keys are dropped by the text loader).
    #[cfg(feature = "image")]
    if env.vision_config.is_some() && dir.join("processor_config.json").exists() {
        return vlm::adapter::load_context_vlm(cfg, env, dir);
    }

    text::load_context(cfg, env, dir)
}
