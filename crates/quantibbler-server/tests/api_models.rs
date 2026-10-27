//! SR-2.* (models list) and SR-3.* (non-streaming chat) plus SR-7.1 legs.

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

use common::*;

/// SR-2.1: GET /v1/models is an OpenAI-shaped list with exactly one entry,
/// object "model", id equal to the served id.
#[test]
fn models_lists_single_loaded_model() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = get(server.port, "/v1/models").expect("models request");
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("models JSON");
        assert_eq!(body["object"], serde_json::json!("list"));
        let entries = body["data"].as_array().expect("data array");
        assert_eq!(entries.len(), 1, "exactly one model: {body}");
        assert_eq!(entries[0]["object"], serde_json::json!("model"));
        assert_eq!(entries[0]["id"], serde_json::json!(FIXTURE_MODEL_ID));
    });
}

/// SR-2.2: no second model can be registered — no registry/hot-swap route
/// exists; probing returns 404 through the error envelope.
#[test]
fn no_second_model_can_be_registered() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        for probe in [
            "/v1/models/register",
            "/v1/models/load",
            "/models",
            "/v1/models/server-model-fixture",
        ] {
            let resp = get(server.port, probe).expect("probe request");
            assert_eq!(resp.status, 404, "probe {probe} must 404");
            let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
            assert!(
                body.get("error").is_some(),
                "probe {probe} must answer in the error envelope: {}",
                resp.body
            );
        }
    });
}

/// SR-3.1: non-streaming chat returns the OpenAI chat.completion shape.
#[test]
#[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
fn chat_nonstreaming_openai_shape() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = post_json(
            server.port,
            "/v1/chat/completions",
            &chat_body(FIXTURE_MODEL_ID, false),
        )
        .expect("chat request");
        assert_eq!(resp.status, 200, "body: {}", resp.body);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("chat JSON");
        assert_eq!(body["object"], serde_json::json!("chat.completion"));
        assert!(body["id"].is_string());
        assert_eq!(body["model"], serde_json::json!(FIXTURE_MODEL_ID));
        let choices = body["choices"].as_array().expect("choices");
        assert!(!choices.is_empty());
        assert_eq!(
            choices[0]["message"]["role"],
            serde_json::json!("assistant")
        );
        assert!(choices[0]["message"]["content"].is_string());
        let usage = &body["usage"];
        assert!(usage["prompt_tokens"].is_u64());
        assert!(usage["completion_tokens"].is_u64());
        assert_eq!(
            usage["total_tokens"],
            serde_json::json!(
                usage["prompt_tokens"].as_u64().unwrap()
                    + usage["completion_tokens"].as_u64().unwrap()
            )
        );
    });
}

/// SR-3.2: an unknown model id is rejected 404 with code model_not_found
/// in the envelope.
#[test]
fn chat_unknown_model_404() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = post_json(
            server.port,
            "/v1/chat/completions",
            &chat_body("some-other-model", false),
        )
        .expect("chat request");
        assert_eq!(resp.status, 404);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
        assert_eq!(body["error"]["code"], serde_json::json!("model_not_found"));
        assert_eq!(
            body["error"]["type"],
            serde_json::json!("invalid_request_error")
        );
    });
}

/// SR-7.1 leg: malformed JSON body maps to 400 invalid_request_error
/// through the envelope.
#[test]
fn chat_rejects_malformed_body_400() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = post_json(server.port, "/v1/chat/completions", "{not json")
            .expect("malformed chat request");
        assert_eq!(resp.status, 400);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
        assert_eq!(
            body["error"]["type"],
            serde_json::json!("invalid_request_error")
        );
        // Invalid parameter leg: bad role.
        let bad_role = r#"{"model": "server-model-fixture", "messages": [{"role": "wizzard", "content": "hi"}]}"#;
        let resp =
            post_json(server.port, "/v1/chat/completions", bad_role).expect("bad role request");
        assert_eq!(resp.status, 400);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
        assert_eq!(
            body["error"]["type"],
            serde_json::json!("invalid_request_error")
        );
    });
}

/// BS-7.1 / BS-7.2: `GET /v1/models` reports the artifact's DECLARED dtype.
#[test]
fn models_reports_the_serving_dtype_bs_7() {
    for (artifact, expected) in [(blob_fixture(), "fp32"), (bf16_blob_fixture(), "bf16")] {
        let server = spawn_server_artifact(&artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = get(server.port, "/v1/models").expect("models request");
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("models JSON");
        assert_eq!(body["data"][0]["dtype"], serde_json::json!(expected));
    }
}
