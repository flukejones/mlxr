//! Gemma 4 text config (`text_config`).
//!
//! The dense base reads the hybrid sliding/global fields, the per-layer
//! rope parameters, the norm/softcap/embed-scale knobs, and the layer-type
//! pattern. MoE (`enable_moe_block`/`num_experts`/…), per-layer-input
//! embeddings (`hidden_size_per_layer_input`), and KV-sharing
//! (`num_kv_shared_layers`) parse here but are unused until those
//! extensions land.

use std::collections::HashMap;

use serde::Deserialize;

use crate::family::EosSpec;
use crate::utils::rope::FloatOrString;

/// Gemma 4 envelope of `config.json`. Stored inside
/// [`crate::config::Family::Gemma4`]; per-layer hyperparameters live in
/// [`TextConfig`] under `text_config`. Quantisation lives on the outer
/// [`crate::config::ModelConfig`].
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    /// Optional explicit EOS token(s): a single id or a list.
    #[serde(default)]
    pub eos_token_id: Option<EosSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Sliding-attention head dim.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Full-attention head dim (Gemma 4 widens the global layers).
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: i32,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: i32,
    pub num_global_key_value_heads: Option<i32>,
    /// Trailing layers that reuse an earlier layer's K/V (E2B/E4B). 0 in
    /// the dense base.
    #[serde(default)]
    pub num_kv_shared_layers: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,

    /// Per-layer-kind rope params, keyed `"full_attention"` /
    /// `"sliding_attention"` → `{rope_theta, rope_type, partial_rotary_factor, factor}`.
    pub rope_parameters: Option<HashMap<String, HashMap<String, FloatOrString>>>,
    #[serde(default)]
    pub rope_traditional: bool,

    /// Full-attention layers may share one projection for K and V.
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f32,
    #[serde(default = "default_use_double_wide_mlp")]
    pub use_double_wide_mlp: bool,

    /// MoE expert routing (26B-A4B). Unused in the dense base.
    #[serde(default)]
    pub enable_moe_block: bool,
    pub num_experts: Option<i32>,
    pub top_k_experts: Option<i32>,
    pub moe_intermediate_size: Option<i32>,

    /// Per-layer input embeddings (E2B/E4B). `> 0` enables them; unused in
    /// the dense base.
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    #[serde(default = "default_vocab_size_per_layer_input")]
    pub vocab_size_per_layer_input: i32,

    /// Optional explicit per-layer kinds. Derived from
    /// `sliding_window_pattern` when absent.
    pub layer_types: Option<Vec<LayerKind>>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

const fn default_hidden_size() -> i32 {
    1536
}
const fn default_num_hidden_layers() -> i32 {
    35
}
const fn default_intermediate_size() -> i32 {
    6144
}
const fn default_num_attention_heads() -> i32 {
    8
}
const fn default_head_dim() -> i32 {
    256
}
const fn default_global_head_dim() -> i32 {
    512
}
const fn default_num_kv_heads() -> i32 {
    1
}
const fn default_rms_norm_eps() -> f32 {
    1e-6
}
const fn default_vocab_size() -> i32 {
    262144
}
const fn default_max_position_embeddings() -> i32 {
    131072
}
const fn default_sliding_window() -> i32 {
    512
}
const fn default_sliding_window_pattern() -> i32 {
    5
}
const fn default_final_logit_softcapping() -> f32 {
    30.0
}
const fn default_use_double_wide_mlp() -> bool {
    true
}
const fn default_tie_word_embeddings() -> bool {
    true
}
const fn default_vocab_size_per_layer_input() -> i32 {
    262144
}

impl TextConfig {
    /// Pattern-derived layer-type table when `layer_types` is absent:
    /// `["sliding"] * (P-1) + ["full"]`, tiled to N layers.
    pub fn layer_types_resolved(&self) -> Vec<LayerKind> {
        if let Some(explicit) = &self.layer_types {
            return explicit.clone();
        }
        let pattern_len = self.sliding_window_pattern as usize;
        (0..self.num_hidden_layers as usize)
            .map(|i| {
                if (i % pattern_len) == pattern_len - 1 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::SlidingAttention
                }
            })
            .collect()
    }

    /// Sliding-window pattern length (the distance between consecutive
    /// full-attention layers, counting the full layer). Derived from
    /// `layer_types` when present — mlx-community checkpoints can carry a
    /// null/stale `sliding_window_pattern` while `layer_types` holds the
    /// truth (gemma-4-31b has the field null but the first `full_attention`
    /// at index 5, implying a pattern of 6). Falls back to the explicit
    /// field, then the default.
    pub fn effective_sliding_window_pattern(&self) -> i32 {
        if let Some(types) = &self.layer_types {
            for (i, ty) in types.iter().enumerate() {
                if *ty == LayerKind::FullAttention {
                    return (i as i32) + 1;
                }
            }
        }
        self.sliding_window_pattern
    }
}

/// Per-layer attention kind. Unknown strings hard-error rather than
/// silently routing as sliding-attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    SlidingAttention,
    FullAttention,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    #[test]
    fn layer_types_derived_from_pattern() {
        let json = serde_json::json!({
            "num_hidden_layers": 6,
            "sliding_window_pattern": 3,
        });
        let cfg: TextConfig = serde_json::from_value(json).unwrap();
        let types = cfg.layer_types_resolved();
        use LayerKind::{FullAttention as F, SlidingAttention as S};
        assert_eq!(types, vec![S, S, F, S, S, F]);
        assert_eq!(cfg.effective_sliding_window_pattern(), 3);
    }

    #[test]
    fn explicit_layer_types_drive_effective_pattern() {
        // mlx-community gotcha: null sliding_window_pattern, truth in
        // layer_types — first full_attention index + 1 = pattern.
        let json = serde_json::json!({
            "num_hidden_layers": 6,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
        });
        let cfg: TextConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.effective_sliding_window_pattern(), 6);
    }
}
