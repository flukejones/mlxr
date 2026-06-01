//! Gemma 4 vision-language [`LanguageModel`] + [`UserInputProcessor`].
//!
//! `prepare()` runs the vision tower per image, projects through
//! `EmbedVision`, embeds the (image-token-masked) ids, stitches features into
//! the image-token slots, and decodes via `Model::forward_embeds`. `step()` is
//! the plain 1-D-rope text path. The text-only branch defers to `Model::forward`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp, r#where},
    Array,
};

use crate::cache::{effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::{ChatMessage, ChatTemplate, ContentPart, MessageContent};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::gemma4::image::multimodal::stitch_image_features;
use crate::gemma4::image::processor::Gemma4ImageProcessor;
use crate::gemma4::image::vision::{EmbedVision, PatchGrid, VisionModel};
use crate::gemma4::text::cache::{make_caches, LayerCache};
use crate::gemma4::text::config::{ModelConfig, TextConfig};
use crate::gemma4::text::text::Model;
use crate::gemma4::vlm::weights::load_full_model;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult, ProcessedImage, Text};
use crate::loader::{load_tokenizer, resolve_bos_id};
use crate::nn::ModelInput;
use crate::user_input::{Image, Prompt, UserInput};

/// Image placeholder the gemma chat template emits per `ContentPart::Image`.
const IMAGE_MARKER: &str = "<|image|>";

pub(crate) struct Gemma4VlmAdapter {
    model: Model,
    vision: VisionModel,
    embed_vision: EmbedVision,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    image_token_id: u32,
    cache_options: CacheOptions,
    vocab_size: i32,
}

impl Gemma4VlmAdapter {
    fn new(
        model: Model,
        vision: VisionModel,
        embed_vision: EmbedVision,
        env: &ModelConfig,
    ) -> Self {
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::default();
        let cache = make_caches(&args, cache_options);
        Self {
            model,
            vision,
            embed_vision,
            cache,
            args,
            image_token_id: env.image_token_id,
            cache_options,
            vocab_size,
        }
    }

    /// Run the tower + projector over every image and concatenate the soft
    /// tokens along axis 0 (`[total_soft_tokens, text_hidden]`).
    fn encode_images(&mut self, pixels: &Array, grids: &[[i32; 3]]) -> Result<Array, Error> {
        let mut feats: Vec<Array> = Vec::with_capacity(grids.len());
        for (i, &[_, ph, pw]) in grids.iter().enumerate() {
            let i = i as i32;
            let img = pixels.index((i..i + 1, .., .., ..));
            let out = self.vision.forward(&img, PatchGrid::new(ph, pw))?;
            let projected = self.embed_vision.forward(&out)?;
            let shape = projected.shape();
            feats.push(projected.reshape(&[shape[1], shape[2]])?);
        }
        if feats.len() == 1 {
            return Ok(feats.into_iter().next().expect("len == 1"));
        }
        Ok(concatenate_axis(&feats, 0)?)
    }
}

impl LanguageModel for Gemma4VlmAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.args, self.cache_options);
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let Some(image) = input.image else {
            let logits = self.model.forward(ModelInput {
                inputs: &input.text.tokens,
                mask: None,
                cache: &mut self.cache,
            })?;
            return Ok(PrepareResult::Logits(logits.index((.., -1, ..))));
        };

        let image_features = self.encode_images(&image.pixels, image.grids.as_slice())?;
        let input_ids = input.text.tokens;

        // Per-layer inputs index `embed_tokens_per_layer` with the ids, so the
        // image-token slots must map to a real id (0) — mirrors the reference
        // masking before `get_per_layer_inputs`.
        let is_image = input_ids.eq(Array::from_int(self.image_token_id as i32))?;
        let zeros = Array::from_int(0).as_dtype(input_ids.dtype())?;
        let masked_ids = r#where(&is_image, &zeros, &input_ids)?;

        let embeds = self.model.model.embed_scaled(&masked_ids)?;
        let stitched =
            stitch_image_features(&image_features, &embeds, &input_ids, self.image_token_id)?;
        let logits = self
            .model
            .forward_embeds(stitched, &masked_ids, &mut self.cache)?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let logits = self.model.forward(ModelInput {
            inputs: &inp,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        effective_prefill_chunk_opt(&self.cache, self.cache_options.max_prefill_chunk)
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        let _ = self.model.forward(ModelInput {
            inputs: tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(())
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.args, options);
        self.cache_options = options;
        Ok(())
    }
}

/// Gemma 4 `UserInputProcessor`: render chat, preprocess images, expand each
/// `<|image|>` marker to `boi + image_token×N + eoi`, tokenize, assert the
/// image-token count matches the soft-token total.
pub(crate) struct Gemma4Processor {
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
    image_processor: Gemma4ImageProcessor,
    bos_id: Option<u32>,
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
    pooling_kernel_size: i32,
    patch_size: i32,
}

impl UserInputProcessor for Gemma4Processor {
    fn family(&self) -> &'static str {
        "gemma4"
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        // Preprocess every image; collect channel-first pixel planes + grids.
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(input.images.len());
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(input.images.len());
        let mut soft_tokens: Vec<i32> = Vec::with_capacity(input.images.len());
        let mut dims: Option<(i32, i32)> = None;
        for image in input.images {
            let processed = match image {
                Image::Decoded(img) => self.image_processor.preprocess_image(img)?,
                Image::Pixels { .. } => {
                    return Err(Error::config(
                        "gemma4 vlm: Image::Pixels bypass not supported; pass Image::Decoded",
                    ));
                }
            };
            if let Some((h, w)) = dims {
                if (h, w) != (processed.height, processed.width) {
                    return Err(Error::shape(
                        "gemma4 vlm: multiple images must resize to identical dims",
                    ));
                }
            } else {
                dims = Some((processed.height, processed.width));
            }
            soft_tokens.push(processed.num_soft_tokens(self.pooling_kernel_size));
            grids.push([
                1,
                processed.height / self.patch_size,
                processed.width / self.patch_size,
            ]);
            planes.push(processed.pixel_values);
        }

        let prompt_text = render_prompt(&self.chat_template, input.prompt, grids.len())?;
        let expanded = self.expand_markers(&prompt_text, &soft_tokens)?;

        let enc = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        if let Some(bos) = self.bos_id {
            if ids.first() != Some(&(bos as i32)) {
                ids.insert(0, bos as i32);
            }
        }

        let observed = ids
            .iter()
            .filter(|&&t| (t as u32) == self.image_token_id)
            .count() as i32;
        let expected: i32 = soft_tokens.iter().sum();
        if observed != expected {
            return Err(Error::shape(format!(
                "gemma4 vlm: prompt has {observed} image tokens but {} image(s) expand to {expected}",
                grids.len()
            )));
        }

        let s = ids.len() as i32;
        let tokens = Array::from_slice(&ids, &[1, s]);

        let image = if grids.is_empty() {
            None
        } else {
            let (h, w) = dims.expect("non-empty images set dims");
            let n = grids.len() as i32;
            let mut all = Vec::with_capacity(planes.iter().map(Vec::len).sum());
            for p in planes {
                all.extend(p);
            }
            let pixels = Array::from_slice(&all, &[n, 3, h, w]);
            Some(ProcessedImage { pixels, grids })
        };

        Ok(LMInput {
            text: Text { tokens, mask: None },
            image,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

impl Gemma4Processor {
    /// Replace each `<|image|>` marker (left to right) with
    /// `boi + image_token×N_i + eoi` using the canonical token strings.
    fn expand_markers(&self, text: &str, soft_tokens: &[i32]) -> Result<String, Error> {
        let parts: Vec<&str> = text.split(IMAGE_MARKER).collect();
        let markers = parts.len() - 1;
        if markers != soft_tokens.len() {
            return Err(Error::shape(format!(
                "gemma4 vlm: template emitted {markers} image markers but {} image(s) supplied",
                soft_tokens.len()
            )));
        }
        let boi = self.token_str(self.boi_token_id)?;
        let img = self.token_str(self.image_token_id)?;
        let eoi = self.token_str(self.eoi_token_id)?;
        let mut out = String::with_capacity(text.len());
        for (i, seg) in parts.iter().enumerate() {
            out.push_str(seg);
            if i < markers {
                out.push_str(&boi);
                for _ in 0..soft_tokens[i] {
                    out.push_str(&img);
                }
                out.push_str(&eoi);
            }
        }
        Ok(out)
    }

    fn token_str(&self, id: u32) -> Result<String, Error> {
        self.tokenizer
            .id_to_token(id)
            .ok_or_else(|| Error::config(format!("gemma4 vlm: token id {id} has no string")))
    }
}

/// Render the chat template with one `ContentPart::Image` per image.
fn render_prompt(
    template: &ChatTemplate,
    prompt: Prompt,
    num_images: usize,
) -> Result<String, Error> {
    let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
    match prompt {
        Prompt::Text(text) => {
            if num_images == 0 {
                template.render(&[ChatMessage::user(text)], true, &kwargs)
            } else {
                let mut parts: Vec<ContentPart> =
                    (0..num_images).map(|_| ContentPart::Image).collect();
                parts.push(ContentPart::Text { text });
                let msg = ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Parts(parts),
                };
                template.render(&[msg], true, &kwargs)
            }
        }
        Prompt::Chat(messages) => template.render(&messages, true, &kwargs),
    }
}

pub(crate) fn load_context_vlm(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let vision_cfg = env
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::config("gemma4 vlm: config has no vision_config"))?;
    let (model, vision, embed_vision) = load_full_model(cfg, env, vision_cfg, dir)?;

    let tokenizer = load_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let image_processor = Gemma4ImageProcessor::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());

    let pooling_kernel_size = image_processor.config.pooling_kernel_size;
    let patch_size = image_processor.config.patch_size;
    let adapter = Gemma4VlmAdapter::new(model, vision, embed_vision, env);
    let processor = Gemma4Processor {
        tokenizer,
        chat_template,
        image_processor,
        bos_id,
        image_token_id: env.image_token_id,
        boi_token_id: env.boi_token_id,
        eoi_token_id: env.eoi_token_id,
        pooling_kernel_size,
        patch_size,
    };
    Ok((Box::new(adapter), Box::new(processor), eos_ids))
}
