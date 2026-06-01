//! Gemma 4 dense [`crate::LanguageModel`] adapter.
//!
//! Gemma 4 uses a per-layer sliding/global cache enum
//! ([`crate::gemma4::text::cache::LayerCache`]) instead of the bare
//! [`crate::cache::KVCache`] used by llama / qwen3. The
//! `Vec<Option<LayerCache>>` slots are built up front by
//! [`crate::gemma4::text::cache::make_caches`].

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::cache::{effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::gemma4::text::cache::{make_caches, LayerCache};
use crate::gemma4::text::config::{ModelConfig, TextConfig};
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::load_model;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::nn::ModelInput;

pub(crate) struct Gemma4Adapter {
    model: Model,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    cache_options: CacheOptions,
    vocab_size: i32,
}

impl Gemma4Adapter {
    fn load(cfg: &Config, env: &ModelConfig, dir: &Path) -> Result<Self, Error> {
        let model = load_model(cfg, env, dir)?;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::default();
        let cache = make_caches(&args, cache_options);
        Ok(Self {
            model,
            cache,
            args,
            cache_options,
            vocab_size,
        })
    }
}

impl LanguageModel for Gemma4Adapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.args, self.cache_options);
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let logits = self.model.forward(ModelInput {
            inputs: &input.text.tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
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

    /// Gemma 4's sliding layers cap each forward at `sliding_window` K/V
    /// positions; combine with the user cap (which may narrow further but
    /// never exceed the window).
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

pub(crate) fn load_context(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let model = Gemma4Adapter::load(cfg, env, dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());
    let processor = TextOnlyProcessor::new("gemma4", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
