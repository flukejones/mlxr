//! Gemma 4 dense text decoder: hybrid sliding/global attention, four
//! norms per layer, GeGLU MLP, embedding scaling, logit soft-capping,
//! tied embeddings.
//!
//! MoE expert routing, per-layer-input embeddings (E2B/E4B), and KV
//! sharing are deferred; each extends this base at its own consumer.

use mlx_rs::{
    builder::Builder,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{clip, expand_dims_axes},
    quantization::MaybeQuantized,
    Array, Dtype,
};

use crate::activations::{
    geglu, logit_softcap, residual_add_scale, GegluCache, LogitSoftcapCache, ResidualAddScaleCache,
};
use crate::cache::KeyValueCache;
use crate::error::Error;
use crate::gemma4::text::config::{LayerKind, TextConfig};
use crate::gemma4::text::rope::{build_layer_rope, LayerRope};
use crate::nn::{ModelInput, RmsNormNoScale};
use crate::utils::{create_attention_mask, AttentionMask};

/// fp16 max magnitude — residual sums are clipped to this before casting
/// back to fp16 to avoid overflow → inf.
const FP16_MAX: f32 = 65504.0;

/// Per-layer attention input. The dense base never sets `shared_kv` /
/// `offset` (always `None`); they are the seam the KV-sharing extension
/// consumes. Local to gemma4 so the shared [`crate::nn::AttentionInput`]
/// stays untouched by a gemma-only concern.
pub struct GemmaAttnInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
    pub shared_kv: Option<(Array, Array)>,
    pub offset: Option<i32>,
}

/// Hidden state + the layer's `(k, v)` (for downstream KV-shared layers)
/// + the pre-update offset.
pub struct AttentionOut {
    pub h: Array,
    pub shared_kv: (Array, Array),
    pub offset: i32,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,
    pub is_sliding: bool,
    pub use_k_eq_v: bool,

    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    /// `None` when K == V (full-attention layers with `attention_k_eq_v`).
    #[quantizable]
    #[param]
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub v_norm: RmsNormNoScale,

    #[param]
    pub rope: LayerRope,
}

impl Attention {
    pub fn new(args: &TextConfig, layer_idx: i32) -> Result<Self, Error> {
        let layer_kind = args.layer_types_resolved()[layer_idx as usize];
        let is_sliding = matches!(layer_kind, LayerKind::SlidingAttention);

        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let head_dim = if matches!(layer_kind, LayerKind::FullAttention) {
            args.global_head_dim
        } else {
            args.head_dim
        };

        let use_k_eq_v = args.attention_k_eq_v && !is_sliding;
        let n_kv_heads = match (use_k_eq_v, args.num_global_key_value_heads) {
            (true, Some(h)) => h,
            _ => args.num_key_value_heads,
        };

        let scale = 1.0_f32;

        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).bias(false).build()?,
            ))
        };
        let q_proj = linear(dim, n_heads * head_dim)?;
        let k_proj = linear(dim, n_kv_heads * head_dim)?;
        let v_proj = if use_k_eq_v {
            None
        } else {
            Some(linear(dim, n_kv_heads * head_dim)?)
        };
        let o_proj = linear(n_heads * head_dim, dim)?;

        let norm = |d: i32| -> Result<nn::RmsNorm, Error> {
            Ok(nn::RmsNormBuilder::new(d).eps(args.rms_norm_eps).build()?)
        };
        let q_norm = norm(head_dim)?;
        let k_norm = norm(head_dim)?;
        let v_norm = RmsNormNoScale::new(args.rms_norm_eps);

        let rope = build_layer_rope(
            head_dim,
            layer_kind,
            args.rope_traditional,
            args.rope_parameters.as_ref(),
        )?;

        Ok(Self {
            layer_idx,
            layer_kind,
            is_sliding,
            use_k_eq_v,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rope,
        })
    }

    #[allow(
        non_snake_case,
        reason = "local bindings mirror ML tensor names (B, L)"
    )]
    pub fn attend<C: KeyValueCache>(
        &mut self,
        input: GemmaAttnInput<'_, C>,
    ) -> Result<AttentionOut, Error> {
        let GemmaAttnInput {
            x,
            mask,
            mut cache,
            shared_kv,
            offset,
        } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Pre-update offset: what RoPE applies to fresh queries. Kept on
        // device (0-D Array) so the dynamic-rope kernel cache stays warm.
        let pre_offset = match (offset, cache.as_ref()) {
            (Some(o), _) => o,
            (None, Some(c)) => c.offset(),
            (None, None) => 0,
        };
        let pre_offset_arr = Array::from_int(pre_offset);

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?;
        let mut queries = self.q_norm.forward(&queries)?;

        let (keys, values) = if let Some(kv) = shared_kv {
            kv
        } else {
            let keys = self
                .k_proj
                .forward(x)?
                .reshape(&[B, L, self.n_kv_heads, self.head_dim])?;
            let k_for_attn = self.k_norm.forward(&keys)?.transpose_axes(&[0, 2, 1, 3])?;
            let k_for_attn = self.rope.forward_dynamic(&k_for_attn, &pre_offset_arr)?;

            let values = if self.use_k_eq_v {
                keys.clone()
            } else {
                self.v_proj
                    .as_mut()
                    .expect("non-keqv layer has v_proj")
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            };
            let v_for_attn = self
                .v_norm
                .forward(&values)?
                .transpose_axes(&[0, 2, 1, 3])?;

            (k_for_attn, v_for_attn)
        };

        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        queries = self.rope.forward_dynamic(&queries, &pre_offset_arr)?;

        // Concat with cache, then attend. Downstream KV-shared layers reuse
        // `(k_full, v_full)`.
        let (k_full, v_full) = if let Some(cache) = cache.as_mut() {
            cache.update_and_fetch(keys, values)?
        } else {
            (keys, values)
        };
        let h = mlx_rs::fast::scaled_dot_product_attention(
            &queries,
            &k_full,
            &v_full,
            self.scale,
            mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
            None,
        )?;

        let h = h.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let h = self.o_proj.forward(&h)?;

        Ok(AttentionOut {
            h,
            shared_kv: (k_full, v_full),
            offset: pre_offset,
        })
    }

    pub fn training_mode_set(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        if let Some(v) = self.v_proj.as_mut() {
            v.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        self.v_norm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    geglu_cache: GegluCache,
}

impl Mlp {
    pub fn new(args: &TextConfig) -> Result<Self, Error> {
        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).bias(false).build()?,
            ))
        };
        Ok(Self {
            gate_proj: linear(args.hidden_size, args.intermediate_size)?,
            down_proj: linear(args.intermediate_size, args.hidden_size)?,
            up_proj: linear(args.hidden_size, args.intermediate_size)?,
            geglu_cache: GegluCache::default(),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = geglu(&mut self.geglu_cache, &gate, &up)?;
        Ok(self.down_proj.forward(&activated)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// fp16-safe additive residual: promote → add → clip → cast back. No-op
/// for non-fp16.
fn clip_residual(x: &Array, y: &Array) -> Result<Array, Error> {
    if x.dtype() != Dtype::Float16 {
        return Ok(x.add(y)?);
    }
    let xf = x.as_dtype(Dtype::Float32)?;
    let yf = y.as_dtype(Dtype::Float32)?;
    let sum = xf.add(&yf)?;
    Ok(clip(&sum, (-FP16_MAX, FP16_MAX))?.as_dtype(Dtype::Float16)?)
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,

    /// Multiplicative per-layer scalar on the residual stream.
    #[param]
    pub layer_scalar: Param<Array>,

    residual_scale_cache: ResidualAddScaleCache,
}

impl DecoderLayer {
    pub fn new(args: &TextConfig, layer_idx: i32) -> Result<Self, Error> {
        let layer_kind = args.layer_types_resolved()[layer_idx as usize];
        let norm = || -> Result<nn::RmsNorm, Error> {
            Ok(nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?)
        };
        Ok(Self {
            layer_idx,
            layer_kind,
            self_attn: Attention::new(args, layer_idx)?,
            mlp: Mlp::new(args)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            pre_feedforward_layernorm: norm()?,
            post_feedforward_layernorm: norm()?,
            layer_scalar: Param::new(Array::ones::<f32>(&[1])?),
            residual_scale_cache: ResidualAddScaleCache::default(),
        })
    }

    pub fn forward_layer<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
    ) -> Result<Array, Error> {
        let h_pre = self.input_layernorm.forward(x)?;
        let AttentionOut { h, .. } = self.self_attn.attend(GemmaAttnInput {
            x: &h_pre,
            mask,
            cache,
            shared_kv: None,
            offset: None,
        })?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = clip_residual(x, &h)?;

        let mid = self.pre_feedforward_layernorm.forward(&h)?;
        let ff_mid = self.mlp.forward(&mid)?;
        let ff_out = self.post_feedforward_layernorm.forward(&ff_mid)?;

        // bf16/fp32: fuse `(h + ff_out) * layer_scalar`. fp16: clip then scale.
        if ff_out.dtype() != Dtype::Float16 {
            Ok(residual_add_scale(
                &mut self.residual_scale_cache,
                &h,
                &ff_out,
                self.layer_scalar.as_ref(),
            )?)
        } else {
            let h = clip_residual(&h, &ff_out)?;
            Ok(h.multiply(self.layer_scalar.as_ref())?)
        }
    }

    pub fn training_mode_set(&mut self, mode: bool) {
        self.self_attn.training_mode_set(mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Gemma4TextModel {
    pub vocab_size: i32,
    pub sliding_window_pattern: i32,
    pub embed_scale: f32,
    embed_scale_arr: std::sync::OnceLock<Array>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Gemma4TextModel {
    pub fn new(args: &TextConfig) -> Result<Self, Error> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            sliding_window_pattern: args.effective_sliding_window_pattern(),
            embed_scale: (args.hidden_size as f32).sqrt(),
            embed_scale_arr: std::sync::OnceLock::new(),
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, cache, .. } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        // Stage scale in h's dtype so the multiply stays bf16/fp16.
        let h_dtype = h.dtype();
        let embed_scale_arr = self.embed_scale_arr.get_or_init(|| {
            Array::from_f32(self.embed_scale)
                .as_dtype(h_dtype)
                .expect("embed_scale cast cannot fail")
        });
        h = h.multiply(embed_scale_arr)?;

        // Per-layer-kind masks: full-attn uses the global cache slot, sliding
        // uses slot 0 (a Sliding cache whose max_size bounds the window).
        // `return_array=Some(true)` forces explicit Array masks (the sliding
        // window restriction needs the array form).
        let pattern = self.sliding_window_pattern as usize;
        let global_idx = pattern.saturating_sub(1).min(cache.len().saturating_sub(1));
        let global_mask = mask_array(create_attention_mask(
            &h,
            &cache[global_idx..=global_idx],
            Some(true),
        )?)?;
        let sliding_mask = if pattern > 1 {
            mask_array(create_attention_mask(&h, &cache[0..1], Some(true))?)?
        } else {
            None
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let mask = match layer.layer_kind {
                LayerKind::FullAttention => global_mask.as_ref(),
                LayerKind::SlidingAttention => sliding_mask.as_ref(),
            };
            let cache_slot = cache.get_mut(i).and_then(|c| c.as_mut());
            h = layer.forward_layer(&h, mask, cache_slot)?;
        }

        Ok(self.norm.forward(&h)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode_set(mode);
        }
        self.norm.training_mode(mode);
    }
}

/// Extract the `Array` from an [`AttentionMask`], expanding a 2-D
/// `[T, kT]` mask to 4-D `[1, 1, T, kT]` so it broadcasts against
/// `[B, H, T, kT]` in the non-fused SDPA path. `Causal`/`None` → `None`.
fn mask_array(mask: Option<AttentionMask>) -> Result<Option<Array>, Error> {
    match mask {
        Some(AttentionMask::Array(a)) => {
            let a = if a.shape().len() == 2 {
                expand_dims_axes(&a, &[0, 1])?
            } else {
                a
            };
            Ok(Some(a))
        }
        _ => Ok(None),
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: TextConfig,
    pub final_logit_softcapping: Option<f32>,

    #[quantizable]
    #[param]
    pub model: Gemma4TextModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,

    softcap_cache: LogitSoftcapCache,
    softcap_array: std::sync::OnceLock<Array>,
}

impl Model {
    pub fn new(args: TextConfig) -> Result<Self, Error> {
        let final_logit_softcapping = if args.final_logit_softcapping > 0.0 {
            Some(args.final_logit_softcapping)
        } else {
            None
        };
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };
        let model = Gemma4TextModel::new(&args)?;
        Ok(Self {
            args,
            final_logit_softcapping,
            model,
            lm_head,
            softcap_cache: LogitSoftcapCache::default(),
            softcap_array: std::sync::OnceLock::new(),
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        let mut logits = if let Some(lm) = self.lm_head.as_mut() {
            lm.forward(&out)?
        } else {
            match &self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out)?,
                MaybeQuantized::Quantized(qe) => qe.as_linear(&out)?,
            }
        };
        if let Some(cap) = self.final_logit_softcapping {
            let logits_dtype = logits.dtype();
            let cap_arr = self.softcap_array.get_or_init(|| {
                Array::from_f32(cap)
                    .as_dtype(logits_dtype)
                    .expect("cap cast cannot fail")
            });
            logits = logit_softcap(&mut self.softcap_cache, &logits, cap_arr)?;
        }
        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma4TextModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm) = self.lm_head.as_mut() {
            lm.training_mode(mode);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use crate::cache::CacheOptions;
    use crate::gemma4::text::cache::make_caches;
    use mlx_rs::transforms::eval;

    /// Small synthetic gemma4 config: 3 sliding + 1 global layer, even
    /// head dims so rope is happy.
    fn synthetic() -> TextConfig {
        let json = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            // ≥ the test's prefill length: a single forward never exceeds
            // the sliding window (the adapter caps prefill chunks at the
            // window via `effective_prefill_chunk_opt`).
            "sliding_window": 8,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn attention_head_dim_per_layer_kind() {
        let cfg = synthetic();
        let sliding = Attention::new(&cfg, 0).unwrap();
        let global = Attention::new(&cfg, 3).unwrap();
        assert_eq!(sliding.layer_kind, LayerKind::SlidingAttention);
        assert_eq!(global.layer_kind, LayerKind::FullAttention);
        // Both head dims are 8 here; the dispatch picks the right source.
        assert_eq!(sliding.head_dim, cfg.head_dim);
        assert_eq!(global.head_dim, cfg.global_head_dim);
    }

    #[test]
    fn decoder_forward_shape_round_trips() {
        let cfg = synthetic();
        let vocab = cfg.vocab_size;
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());

        // Prefill 5 tokens.
        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, vocab]);

        // Decode one more token.
        let next = Array::from_slice(&[7_i32], &[1, 1]);
        let logits2 = model
            .forward(ModelInput {
                inputs: &next,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, vocab]);

        // Sliding slot (0) is windowed; global slot (3) is unbounded.
        assert_eq!(caches[0].as_ref().unwrap().max_size(), Some(8));
        assert_eq!(caches[3].as_ref().unwrap().max_size(), None);
        assert_eq!(caches[3].as_ref().unwrap().offset(), 6);
    }

    #[test]
    fn logit_softcap_bounds_output() {
        let cfg = synthetic();
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());
        let inputs = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        // final_logit_softcapping = 30.0 ⇒ |logits| < 30.
        let max_mag = logits.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_mag < 30.0, "softcap did not bound logits: {max_mag}");
    }
}
