//! POST /v1/chat/completions (SR-3.* non-streaming, SR-4.* streaming) and
//! POST /v1/completions (SR-5.*) — one shared decode path (design D4), SSE
//! via axum `Sse` over an mpsc-backed stream (IA6), errors through the
//! single envelope (D7).

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio::sync::mpsc;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::types::*;
use crate::engine::{FinishReason, GenerationOutput};

const DEFAULT_MAX_TOKENS: u32 = 32;
const HARD_MAX_TOKENS: u32 = 4096;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{prefix}-{x}{n}", x = now())
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if let Some(expected) = &state.api_key {
        let got = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match got {
            Some(key) if key == expected => Ok(()),
            _ => Err(ApiError::unauthorized()),
        }
    } else {
        Ok(())
    }
}

fn validate_common(model: &str, state: &AppState, max_tokens: Option<u32>) -> Result<(), ApiError> {
    if !state.ready() {
        return Err(ApiError::not_ready());
    }
    if model != state.model_id {
        return Err(ApiError::model_not_found(model));
    }
    if let Some(n) = max_tokens {
        if n == 0 {
            return Err(ApiError::bad_request("max_tokens must be >= 1"));
        }
        if n > HARD_MAX_TOKENS {
            return Err(ApiError::bad_request(
                "max_tokens exceeds the v0.1 hard cap of 4096",
            ));
        }
    }
    Ok(())
}

/// Chat role validation + prompt text extraction (SR-7.1 leg: bad role is
/// an invalid parameter).
fn chat_prompt_text(messages: &[ChatMessage]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "messages must contain at least one message",
        ));
    }
    for m in messages {
        match m.role.as_str() {
            "system" | "user" | "assistant" => {}
            other => {
                return Err(ApiError::bad_request(format!(
                    "invalid message role {other:?}"
                )));
            }
        }
    }
    Ok(messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join(" "))
}

fn usage_of(out: &GenerationOutput) -> Usage {
    Usage {
        prompt_tokens: out.prompt_tokens as u32,
        completion_tokens: out.completion_tokens.len() as u32,
        total_tokens: (out.prompt_tokens + out.completion_tokens.len()) as u32,
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;
    let Json(req) = body.map_err(crate::api::json_rejection_error)?;
    validate_common(&req.model, &state, req.max_tokens)?;
    let prompt_text = chat_prompt_text(&req.messages)?;
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let stop = req.stop.as_ref().and_then(StopSequence::first);

    if req.stream {
        stream_response(&state, StreamShape::Chat, &prompt_text, max_tokens, stop).await
    } else {
        let engine = &state.engine;
        let out = generate_sync(engine, &prompt_text, max_tokens, stop).await?;
        let text = engine
            .tokenizer
            .decode(&out.completion_tokens, false)
            .map_err(|e| ApiError::internal(format!("detokenize: {e}")))?;
        Ok(Json(ChatCompletionResponse {
            id: new_id("chatcmpl"),
            object: "chat.completion".into(),
            created: now(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: text,
                },
                finish_reason: out.finish.as_str().into(),
            }],
            usage: usage_of(&out),
        })
        .into_response())
    }
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;
    let Json(req) = body.map_err(crate::api::json_rejection_error)?;
    validate_common(&req.model, &state, req.max_tokens)?;
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let stop = req.stop.as_ref().and_then(StopSequence::first);

    if req.stream {
        stream_response(&state, StreamShape::Text, &req.prompt, max_tokens, stop).await
    } else {
        let engine = &state.engine;
        let out = generate_sync(engine, &req.prompt, max_tokens, stop).await?;
        let text = engine
            .tokenizer
            .decode(&out.completion_tokens, false)
            .map_err(|e| ApiError::internal(format!("detokenize: {e}")))?;
        Ok(Json(CompletionResponse {
            id: new_id("cmpl"),
            object: "text_completion".into(),
            created: now(),
            model: req.model,
            choices: vec![TextChoice {
                index: 0,
                text,
                finish_reason: out.finish.as_str().into(),
            }],
            usage: usage_of(&out),
        })
        .into_response())
    }
}

/// Full decode with the permit held across generation (design D5).
pub async fn generate_sync(
    engine: &Arc<crate::engine::Engine>,
    prompt_text: &str,
    max_tokens: u32,
    stop: Option<&str>,
) -> Result<GenerationOutput, ApiError> {
    let tokens = engine
        .encode(prompt_text)
        .map_err(|e| ApiError::internal(format!("tokenize: {e}")))?;
    let permit = engine.acquire().await;
    engine
        .generate(&permit, &tokens, max_tokens as usize, stop)
        .map_err(|e| ApiError::internal(e.0))
}
enum StreamShape {
    Chat,
    Text,
}

/// SSE response: chunk events over an mpsc channel; when the client
/// disconnects the response body (and the receiver) drops, the generator
/// observes the closed channel and aborts that decode only (SR-4.3, IA6).
/// The concrete stream type is erased into the response before returning.
async fn stream_response(
    state: &Arc<AppState>,
    shape: StreamShape,
    prompt_text: &str,
    max_tokens: u32,
    stop: Option<&str>,
) -> Result<Response, ApiError> {
    let engine = Arc::clone(&state.engine);
    let id = match shape {
        StreamShape::Chat => new_id("chatcmpl"),
        StreamShape::Text => new_id("cmpl"),
    };
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let model_id = state.model_id.clone();
    let is_chat = matches!(shape, StreamShape::Chat);
    let stop_owned = stop.map(str::to_string);
    let prompt_owned = prompt_text.to_string();
    tokio::spawn(async move {
        stream_engine(
            engine,
            id,
            model_id,
            is_chat,
            &prompt_owned,
            max_tokens,
            stop_owned.as_deref(),
            tx,
        )
        .await;
    });
    let sse = Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    );
    Ok(sse.into_response())
}

/// The generation task behind one SSE stream: permit, tokenize, then one
/// chunk per token; the final chunk carries the finish reason; then the
/// literal `data: [DONE]` sentinel (SR-4.1/4.2/5.2).
#[allow(clippy::too_many_arguments)]
async fn stream_engine(
    engine: Arc<crate::engine::Engine>,
    id: String,
    model_id: String,
    is_chat: bool,
    prompt_text: &str,
    max_tokens: u32,
    stop: Option<&str>,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    let created = now();
    let tokens = match engine.encode(prompt_text) {
        Ok(t) => t,
        Err(e) => {
            // Mid-stream errors emit one envelope-shaped data event, then
            // DONE; the SSE framing contract stays intact.
            let _ = tx
                .send(Ok(Event::default().data(
                    serde_json::json!({ "error": { "message": e.0, "type": "server_error",
                        "param": null, "code": null } })
                    .to_string(),
                )))
                .await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            return;
        }
    };
    // The permit IS the concurrency slot (design D5): it lives for the
    // whole decode and releases on drop, including the disconnect-abort path.
    let _permit = engine.acquire().await;
    let mut session = match engine.decode_start(&tokens, max_tokens as usize, stop) {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(Ok(Event::default().data(err_event(&e.0)))).await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            return;
        }
    };
    // First content delta carries the assistant role for chat (SR-4.1).
    let mut first = true;
    loop {
        let step = engine.decode_step(&mut session);
        let token = match step {
            Ok(Some(t)) => t,
            Ok(None) => break,
            Err(e) => {
                let _ = tx.send(Ok(Event::default().data(err_event(&e.0)))).await;
                break;
            }
        };
        let piece = match engine.tokenizer.decode(&[token], false) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default().data(err_event(&format!("decode: {e}")))))
                    .await;
                break;
            }
        };
        let event = if is_chat {
            serde_json::to_string(&ChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta {
                        role: if first {
                            Some("assistant".into())
                        } else {
                            None
                        },
                        content: Some(piece),
                    },
                    finish_reason: None,
                }],
            })
        } else {
            serde_json::to_string(&TextChunk {
                id: id.clone(),
                object: "text_completion.chunk".into(),
                created,
                model: model_id.clone(),
                choices: vec![StreamTextChoice {
                    index: 0,
                    text: piece,
                    finish_reason: None,
                }],
            })
        };
        first = false;
        match event {
            Ok(json) => {
                if tx.send(Ok(Event::default().data(json))).await.is_err() {
                    // Client disconnected: abort this decode only; the
                    // permit drops with this task (SR-4.3).
                    break;
                }
            }
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default().data(err_event(&e.to_string()))))
                    .await;
                break;
            }
        }
    }
    let finish = session.finish().unwrap_or(FinishReason::Length).as_str();
    let final_event = if is_chat {
        serde_json::to_string(&ChatChunk {
            id,
            object: "chat.completion.chunk".into(),
            created,
            model: model_id,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish.into()),
            }],
        })
    } else {
        serde_json::to_string(&TextChunk {
            id,
            object: "text_completion.chunk".into(),
            created,
            model: model_id,
            choices: vec![StreamTextChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish.into()),
            }],
        })
    };
    if let Ok(json) = final_event {
        let _ = tx.send(Ok(Event::default().data(json))).await;
    }
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}

fn err_event(message: &str) -> String {
    serde_json::json!({
        "error": { "message": message, "type": "server_error", "param": null, "code": null }
    })
    .to_string()
}
