//! The chat endpoint on a REAL exported artifact directory.
//!
//! The committed fixtures are hand-built; a real export is not. The template
//! the trainer writes lives in the canonical `chat_template.jinja` sidecar and
//! never in `tokenizer.json`, so this leg is the one a fixture-only suite
//! cannot stand in for: it loads an artifact produced by the export path and
//! serves a chat completion through it.
//!
//! Gated on `QKB_REAL_ARTIFACT_DIR` (SKIP when unset): the artifact is
//! hundreds of megabytes and is not committed, so the leg runs where the
//! artifact is present and reports a skip where it is not.

#![allow(clippy::duplicate_mod)] // design D1: tests embed the binary's modules (no lib target)

#[path = "../src/api/mod.rs"]
pub mod api;
#[path = "../src/cli.rs"]
pub mod cli;
#[path = "../src/engine/mod.rs"]
pub mod engine;
#[path = "../src/model/mod.rs"]
pub mod model;
#[path = "../src/sampler.rs"]
pub mod sampler;

mod common;

use std::path::PathBuf;

use common::*;

/// The environment variable naming the real artifact directory to serve.
const REAL_ARTIFACT_ENV: &str = "QKB_REAL_ARTIFACT_DIR";

/// The pinned probe message this leg serves, and the reference render of it
/// committed beside the fixtures (produced by the Python stack's
/// `apply_chat_template` with the generation prompt on and thinking off).
const PROBE_CONTENT: &str = "hello";

/// The HTTP budget for the served request.
///
/// The fixture-sized default is not a statement about a real artifact: serving
/// one generated token here runs 196 unoptimized layers, measured at 54.6 s on
/// this artifact, so a shorter budget would report the debug build's speed as
/// a serving failure.
const SERVING_BUDGET: std::time::Duration = std::time::Duration::from_secs(300);

fn reference_prompt() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/chat_golden.json");
    let text = std::fs::read_to_string(&path).expect("read chat_golden.json");
    let golden: serde_json::Value = serde_json::from_str(&text).expect("chat_golden.json is JSON");
    golden["prompt"]
        .as_str()
        .expect("the golden carries the reference prompt")
        .to_string()
}

/// A real artifact serves chat from its own canonical sidecar, and the prompt
/// the server renders for a probe message is the reference render of that same
/// message.
#[test]
fn real_artifact_serves_chat_with_prompt_parity_ct_3_1() {
    let Ok(dir) = std::env::var(REAL_ARTIFACT_ENV) else {
        eprintln!("skipped: {REAL_ARTIFACT_ENV} is unset; no real artifact directory to serve");
        return;
    };
    let dir = PathBuf::from(dir);

    // The artifact's shape, asserted before the server sees it: the template
    // is the sidecar, and the tokenizer document carries no such field.
    let sidecar_path = dir.join("chat_template.jinja");
    let sidecar = std::fs::read_to_string(&sidecar_path)
        .unwrap_or_else(|e| panic!("{}: {e}", sidecar_path.display()));
    let tokenizer_path = dir.join("tokenizer.json");
    let document: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&tokenizer_path)
            .unwrap_or_else(|e| panic!("{}: {e}", tokenizer_path.display())),
    )
    .expect("the artifact's tokenizer.json parses");
    assert!(
        document.get("chat_template").is_none(),
        "this leg is only meaningful on an artifact whose tokenizer.json declares no template"
    );

    let (state, router) = ready_router_for(&dir);
    let port = serve_on_port(router);

    let template = state
        .engine
        .chat_template
        .clone()
        .expect("the real artifact declares its chat template");
    assert_eq!(
        template, sidecar,
        "the served template must be the artifact's own sidecar bytes"
    );

    let messages = vec![api::types::ChatMessage {
        role: "user".into(),
        content: PROBE_CONTENT.into(),
    }];
    let rendered =
        api::chat_template::render(&template, &messages).expect("render the probe message");
    assert_eq!(
        rendered,
        reference_prompt(),
        "the rendered prompt must equal the reference render of the same message"
    );

    let body = format!(
        r#"{{"model": "{}", "messages": [{{"role": "user", "content": "{PROBE_CONTENT}"}}], "max_tokens": 1}}"#,
        state.model_id
    );
    let resp = post_json_within(port, "/v1/chat/completions", &body, SERVING_BUDGET)
        .expect("chat request");
    assert_ne!(
        resp.status, 400,
        "a real artifact must not refuse the chat endpoint: {}",
        resp.body
    );
    assert_eq!(resp.status, 200, "{}", resp.body);

    // The served request tokenized the TEMPLATED prompt, not the bare content.
    let served: serde_json::Value = serde_json::from_str(&resp.body).expect("chat JSON");
    let templated_tokens = state.engine.encode(&rendered).expect("tokenize").len() as u64;
    assert_eq!(
        served["usage"]["prompt_tokens"],
        serde_json::json!(templated_tokens),
        "the served prompt_tokens must be the template-rendered prompt's count"
    );
}
