//! Qwen3 family: the [`crate::LanguageModel`] adapter over the qwen3
//! model graph in [`crate::models::qwen3`].

mod adapter;

pub use adapter::load_context;
