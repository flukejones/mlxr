//! Llama family: the [`crate::LanguageModel`] adapter over the llama
//! model graph in [`crate::models::llama`].

mod adapter;

pub use adapter::load_context;
