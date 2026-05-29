//! Model-facing input.
//!
//! [`crate::UserInputProcessor`] turns a [`crate::UserInput`] into an
//! [`LMInput`]; [`crate::LanguageModel::prepare`] consumes it to seed
//! the KV cache, then [`crate::LanguageModel::step`] consumes one token
//! id at a time.

use mlx_rs::Array;

/// Output of a [`crate::UserInputProcessor::prepare`] call.
#[derive(Debug)]
pub struct LMInput {
    pub text: Text,
}

/// Tokenised text portion of an [`LMInput`].
#[derive(Debug)]
pub struct Text {
    /// `[1, S]` int32 token ids (batch dim always 1).
    pub tokens: Array,
    /// Optional `[1, S]` mask; `None` lets the model build its own.
    pub mask: Option<Array>,
}

/// Result of [`crate::LanguageModel::prepare`]: logits to sample now, or
/// "cache primed, call `step`".
pub enum PrepareResult {
    Primed,
    Logits(Array),
}

/// One step's output from [`crate::LanguageModel::step`].
pub struct LMOutput {
    /// `[1, 1, vocab_size]` logits over the next token.
    pub logits: Array,
}
