//! VLM weight loader: split a `gemma4`+vision checkpoint into the text
//! `Model`, the bf16 `VisionModel` tower, and the quantized `EmbedVision`
//! projector. Vision tower weights are never quantized; text + projector are
//! quantized per `cfg.quantization()`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::gemma4::image::config::VisionConfig;
use crate::gemma4::image::vision::{EmbedVision, VisionModel};
use crate::gemma4::text::config::ModelConfig;
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::{is_shared_kv_layer_key, rewrite_outer_key};
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};

/// Quantiser-stat side keys to drop from the vision tower (it is not
/// quantized, so any stray clip buffers are inert).
const VISION_DROP_SUBSTRINGS: &[&str] = &["input_max", "input_min", "output_max", "output_min"];

/// One bucket a checkpoint key routes to after prefix rewriting.
enum Bucket {
    /// `vision_tower.…` (key with the prefix stripped). bf16, never quantized.
    Vision(String),
    /// `embed_vision.…` (key with the prefix stripped). Quantized.
    EmbedVision(String),
    /// Everything else → the text `Model` (post `rewrite_outer_key`).
    Text(String),
}

/// Map a `vision_tower.…`-stripped checkpoint key onto the `VisionModel`
/// param walk: drop the `ClippableLinear` `.linear.` wrapper segment, and
/// collapse the `encoder.layers.N` nesting to `encoder.N` (our `encoder` is a
/// bare `Vec`, not a wrapped sub-module).
fn rewrite_vision_key(key: &str) -> String {
    key.replace(".linear.", ".")
        .replace("encoder.layers.", "encoder.")
}

fn bucket_key(key: &str) -> Bucket {
    if let Some(rest) = key.strip_prefix("vision_tower.") {
        return Bucket::Vision(rewrite_vision_key(rest));
    }
    if let Some(rest) = key.strip_prefix("embed_vision.") {
        return Bucket::EmbedVision(rest.to_owned());
    }
    Bucket::Text(rewrite_outer_key(key))
}

/// Load the text model, vision tower, and projector from one checkpoint.
pub(crate) fn load_full_model(
    cfg: &Config,
    env: &ModelConfig,
    vision_cfg: &VisionConfig,
    model_dir: &Path,
) -> Result<(Model, VisionModel, EmbedVision), Error> {
    let mut text = Model::new(env.text_config.clone())?;
    let mut vision = VisionModel::new(vision_cfg)?;
    let mut embed_vision = EmbedVision::new(vision_cfg, env.text_config.hidden_size)?;
    if let Some(q) = cfg.quantization() {
        text = text.try_into_quantized(q.group_size, q.bits)?;
        embed_vision = embed_vision.try_into_quantized(q.group_size, q.bits)?;
    }

    let num_layers = env.text_config.num_hidden_layers;
    let num_kv_shared = env.text_config.num_kv_shared_layers;

    // Read shards, bucket, and quant-rewrite the text + projector keys (the
    // vision tower stays bf16). Drop KV-shared layers' K/V keys.
    let shards = list_shards(model_dir)?;
    let mut text_raw: HashMap<String, Array> = HashMap::new();
    let mut quant_raw: HashMap<String, Array> = HashMap::new();
    let mut vision_raw: HashMap<String, Array> = HashMap::new();
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            match bucket_key(&k) {
                Bucket::Vision(p) => {
                    if VISION_DROP_SUBSTRINGS.iter().any(|s| p.contains(s)) {
                        continue;
                    }
                    vision_raw.insert(p, v);
                }
                Bucket::EmbedVision(p) => {
                    quant_raw.insert(format!("embed_vision.{p}"), v);
                }
                Bucket::Text(key) => {
                    if is_shared_kv_layer_key(&key, num_layers, num_kv_shared) {
                        continue;
                    }
                    text_raw.insert(key, v);
                }
            }
        }
    }
    let text_weights = rewrite_quantised_keys(text_raw);
    // Projector keys come back as `embed_vision.embedding_projection.inner.*`;
    // strip the bucket prefix back off for the `EmbedVision` param walk.
    let embed_weights: HashMap<String, Array> = rewrite_quantised_keys(quant_raw)
        .into_iter()
        .map(|(k, v)| (k.strip_prefix("embed_vision.").unwrap_or(&k).to_owned(), v))
        .collect();

    let mut leftover: Vec<String> = Vec::new();
    bind(&mut text, text_weights, "text", &mut leftover);
    bind(&mut vision, vision_raw, "vision_tower", &mut leftover);
    bind(
        &mut embed_vision,
        embed_weights,
        "embed_vision",
        &mut leftover,
    );

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4 VLM loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }

    eval_params(text.parameters()).map_err(Error::Exception)?;
    eval_params(vision.parameters()).map_err(Error::Exception)?;
    eval_params(embed_vision.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok((text, vision, embed_vision))
}

/// Bind a bucket's weights into a module's parameter walk; record unbound keys.
fn bind<M: ModuleParameters>(
    module: &mut M,
    weights: HashMap<String, Array>,
    prefix: &str,
    leftover: &mut Vec<String>,
) {
    let mut params = module.parameters_mut().flatten();
    for (k, v) in weights {
        if let Some(slot) = params.get_mut(&*k) {
            **slot = v;
        } else {
            leftover.push(format!("{prefix}.{k}"));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn buckets_route_by_prefix() {
        assert!(matches!(
            bucket_key("vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"),
            Bucket::Vision(p) if p == "encoder.0.self_attn.q_proj.weight"
        ));
        assert!(matches!(
            bucket_key("embed_vision.embedding_projection.weight"),
            Bucket::EmbedVision(p) if p == "embedding_projection.weight"
        ));
        assert!(matches!(
            bucket_key("model.layers.0.self_attn.q_proj.weight"),
            Bucket::Text(p) if p == "model.layers.0.self_attn.q_proj.weight"
        ));
    }
}
