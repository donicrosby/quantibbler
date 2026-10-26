//! Shared OpenAI request/response wire types (chat + legacy completions).

use serde::{Deserialize, Serialize};

/// One chat message. Roles are validated at the handler edge (SR-7.1 leg).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    // Sampling knobs are ACCEPTED BUT GREEDY in v0.1 (design D6/OQ4/IA4):
    // parsed to keep the OpenAI request schema, honestly ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub temperature: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>),
}

impl StopSequence {
    /// All stop sequences for this request. `None` = no stops requested;
    /// every listed sequence is honored (v0.1 previously applied only the
    /// first and silently dropped the rest).
    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            StopSequence::One(s) => Some(std::slice::from_ref(s)),
            StopSequence::Many(v) => Some(v),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    // Accepted-but-greedy (IA4); see ChatCompletionRequest.
    #[serde(default)]
    #[allow(dead_code)]
    pub temperature: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Prefill wall-clock (BS-6). Carried by NON-streaming responses only: the SSE
/// chunk types below have no such field, so a stream cannot report one.
#[derive(Debug, Serialize)]
pub struct Timings {
    pub prompt_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

/// One streamed chunk (chat shape; also used for completions with `delta`
/// swapped for `text` at the serialization layer).
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TextChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamTextChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamTextChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}
