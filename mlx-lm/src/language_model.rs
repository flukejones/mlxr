//! The two traits every model family implements.
//!
//! - [`UserInputProcessor`] turns a [`crate::user_input::UserInput`]
//!   into the model-facing [`LMInput`] (chat-template render +
//!   tokenise; vision/audio towers for multimodal families).
//! - [`LanguageModel`] owns the parsed model + its KV cache + the
//!   per-step decoder: [`LanguageModel::prepare`] primes the cache,
//!   [`LanguageModel::step`] produces one token's logits.
//!
//! [`crate::model_context::ModelContext`] holds one of each and
//! [`crate::model_context::generate`] drives them.

use crate::cache::CacheOptions;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::lm_input::{LMInput, LMOutput, PrepareResult, Text};
use crate::sampler::SamplerState;
use crate::user_input::{Prompt, UserInput};

/// Turn a [`UserInput`] into an [`LMInput`]. One impl per family — the
/// only place that knows the family's preprocessing details.
pub trait UserInputProcessor: Send {
    /// Short family identifier (`"llama"`, `"qwen3"`, …) for errors.
    fn family(&self) -> &'static str;

    /// Convert user-facing input to model-facing input.
    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error>;

    /// Decode generated token ids back to UTF-8.
    fn decode(&self, ids: &[u32]) -> Result<String, Error>;
}

/// One language model. Implementors hold the parsed module graph + KV
/// cache as fields; [`crate::model_context::ModelContext`] holds this
/// as `Box<dyn LanguageModel>`.
pub trait LanguageModel: Send {
    /// Reset the KV cache. Called at the start of every generate turn.
    fn reset(&mut self);

    /// Ingest the prompt + prime the KV cache. Returns
    /// [`PrepareResult::Logits`] if prefill already computed the
    /// next-token distribution, else [`PrepareResult::Primed`].
    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error>;

    /// Produce one token's logits from the previously-sampled token (a
    /// `[1]` int32 device array). The model advances its own cursor and
    /// reshapes to `[1, 1]` internally.
    fn step(&mut self, last_token: &mlx_rs::Array) -> Result<LMOutput, Error>;

    /// Text vocab size, used to validate sampled ids.
    fn vocab_size(&self) -> i32;

    /// Prompt tokens the cache holds per forward. `Some(W)` triggers
    /// chunked prefill; `None` = unbounded.
    fn prefill_chunk_size(&self) -> Option<i32> {
        None
    }

    /// Ingest one prefill chunk, advance the cache, drop logits. Only
    /// called when `prefill_chunk_size` is `Some`.
    fn prefill_chunk(&mut self, _tokens: &mlx_rs::Array) -> Result<(), Error> {
        Err(Error::Other(
            "prefill_chunk called on a model with no prefill_chunk_size override".into(),
        ))
    }

    /// True iff an MTP head is loaded.
    fn has_mtp(&self) -> bool {
        false
    }

    /// MTP step: committed token ids + next pending token. `None` =
    /// no MTP head. Greedy accepts-if-equal; sampled uses rejection.
    fn try_mtp_decode(
        &mut self,
        _last_token: &mlx_rs::Array,
        _sampler: &mut SamplerState,
    ) -> Result<Option<(Vec<u32>, mlx_rs::Array)>, Error> {
        Ok(None)
    }

    /// MTP draft depth. No-op without an MTP head.
    fn set_mtp_depth(&mut self, _n: u32) {}

    /// Pick the cache backing; rebuilds the per-layer vec. Call after
    /// load, before first turn.
    fn set_cache_options(&mut self, _options: CacheOptions) -> Result<(), Error> {
        Ok(())
    }
}

/// Text-only processor: renders the chat template + tokenises. Each
/// text-only family wraps its loaded tokenizer + template here.
pub struct TextOnlyProcessor {
    family: &'static str,
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
}

impl TextOnlyProcessor {
    pub fn new(
        family: &'static str,
        tokenizer: tokenizers::Tokenizer,
        chat_template: ChatTemplate,
    ) -> Self {
        Self {
            family,
            tokenizer,
            chat_template,
        }
    }
}

impl UserInputProcessor for TextOnlyProcessor {
    fn family(&self) -> &'static str {
        self.family
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        let rendered = match input.prompt {
            Prompt::Text(s) => s,
            Prompt::Chat(msgs) => self
                .chat_template
                .render(&msgs, true, &input.template_kwargs)?,
        };
        let enc = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let len = ids.len() as i32;
        let tokens = mlx_rs::Array::from_slice(&ids, &[1, len]);
        Ok(LMInput {
            text: Text { tokens, mask: None },
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::ChatMessage;

    fn dummy_processor() -> TextOnlyProcessor {
        let tok_json = r#"{
            "version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"<unk>":2},"unk_token":"<unk>"}
        }"#;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
        let template = ChatTemplate::from_source(
            "{% for m in messages %}{{ m.role }}={{ m.content }}|{% endfor %}",
        );
        TextOnlyProcessor::new("test", tokenizer, template)
    }

    #[test]
    fn text_prompt_round_trips() {
        let mut p = dummy_processor();
        let lm = p.prepare(UserInput::text("hello world")).unwrap();
        assert_eq!(lm.text.tokens.shape()[0], 1);
        assert!(lm.text.tokens.shape()[1] >= 1);
    }

    #[test]
    fn chat_prompt_renders_through_template() {
        let mut p = dummy_processor();
        let lm = p
            .prepare(UserInput::chat(vec![ChatMessage::user("hello")]))
            .unwrap();
        assert_eq!(lm.text.tokens.shape()[0], 1);
        assert!(lm.text.tokens.shape()[1] > 0);
    }
}
