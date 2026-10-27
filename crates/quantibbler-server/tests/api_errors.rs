//! SR-7.2 / SR-7.3: not-ready maps to the 503 envelope; internal failure
//! maps to the 500 envelope; no error path leaks anything else.

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

/// SR-7.2: an inference request before readiness gets 503 with the
/// server_error envelope and code model_not_ready.
#[test]
fn errors_notready_503_envelope() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    for_each_serving_source(|artifact| {
        rt.block_on(async {
            let (_state, router) = not_ready_router_for(artifact);
            let port = serve_on_port(router);
            for path in ["/v1/chat/completions", "/v1/completions"] {
                let body = if path.ends_with("completions") && !path.ends_with("chat/completions") {
                    completion_body(FIXTURE_MODEL_ID, false)
                } else {
                    chat_body(FIXTURE_MODEL_ID, false)
                };
                let resp = post_json(port, path, &body).expect("request");
                assert_eq!(resp.status, 503, "path {path}");
                let parsed: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope");
                assert_eq!(parsed["error"]["type"], serde_json::json!("server_error"));
                assert_eq!(
                    parsed["error"]["code"],
                    serde_json::json!("model_not_ready")
                );
                assert!(parsed["error"]["message"].is_string());
                assert!(parsed["error"]["param"].is_null());
            }
        });
    });
}

/// SR-7.3: an internal failure during generation maps to 500 with the
/// server_error envelope — injected through the engine fault point.
#[test]
fn errors_internal_500_envelope() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    for_each_serving_source(|artifact| {
        rt.block_on(async {
            let (_state, router) = failing_router_for(artifact);
            let port = serve_on_port(router);
            let resp = post_json(
                port,
                "/v1/chat/completions",
                &chat_body(FIXTURE_MODEL_ID, false),
            )
            .expect("request");
            assert_eq!(resp.status, 500);
            let parsed: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope");
            assert_eq!(parsed["error"]["type"], serde_json::json!("server_error"));
            assert!(parsed["error"]["code"].is_null());
            // And a body-parse rejection never leaks a non-envelope either.
            let resp = post_json(port, "/v1/completions", "not-json-at-all").expect("request");
            assert_eq!(resp.status, 400);
            let parsed: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope");
            assert_eq!(
                parsed["error"]["type"],
                serde_json::json!("invalid_request_error")
            );
        });
    });
}
