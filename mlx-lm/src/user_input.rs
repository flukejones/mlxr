//! Caller-facing input to [`crate::generate`].
//!
//! One struct carries the prompt (text or chat) plus template kwargs.
//! Image/audio modalities are added with the families that consume them.

use std::collections::HashMap;

use crate::chat_template::ChatMessage;

/// Top-level user-facing input handed to [`crate::generate`].
pub struct UserInput {
    /// What the user said: plain text or structured chat.
    pub prompt: Prompt,

    /// Named values forwarded to the chat-template render (e.g.
    /// `enable_thinking`). Empty by default.
    pub template_kwargs: HashMap<String, serde_json::Value>,
}

/// Conversation shape. `Text` is the one-shot fast path; `Chat` carries
/// structured history the model's Jinja template renders.
pub enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

impl UserInput {
    /// Plain-text prompt.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Prompt::Text(prompt.into()),
            template_kwargs: HashMap::new(),
        }
    }

    /// Structured chat conversation.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            prompt: Prompt::Chat(messages),
            template_kwargs: HashMap::new(),
        }
    }

    /// Set one template kwarg, builder-style.
    #[must_use]
    pub fn with_template_kwarg(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.template_kwargs.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_constructs() {
        let input = UserInput::text("hi");
        assert!(matches!(input.prompt, Prompt::Text(ref s) if s == "hi"));
    }

    #[test]
    fn chat_input_constructs() {
        let input = UserInput::chat(vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ]);
        let Prompt::Chat(ref msgs) = input.prompt else {
            panic!("expected Chat prompt");
        };
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }
}
