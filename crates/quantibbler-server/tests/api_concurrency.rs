//! SR-5.1 (non-streaming completions) and SR-9.* (concurrency probes).

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

/// SR-5.1: non-streaming completions return the text_completion shape with
/// choices[].text and usage.
#[test]
fn completions_nonstreaming_openai_shape() {
    let server = spawn_server(&[]);
    assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
    let resp = post_json(
        server.port,
        "/v1/completions",
        &completion_body(FIXTURE_MODEL_ID, false),
    )
    .expect("completion request");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let body: serde_json::Value = serde_json::from_str(&resp.body).expect("completion JSON");
    assert_eq!(body["object"], serde_json::json!("text_completion"));
    let choices = body["choices"].as_array().expect("choices");
    assert!(!choices.is_empty());
    assert!(choices[0]["text"].is_string());
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].is_u64());
    assert!(usage["completion_tokens"].is_u64());
    assert!(usage["total_tokens"].is_u64());
    // Unknown model shares chat's 404 mapping.
    let resp = post_json(
        server.port,
        "/v1/completions",
        &completion_body("other-model", false),
    )
    .expect("completion request");
    assert_eq!(resp.status, 404);
    let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
    assert_eq!(body["error"]["code"], serde_json::json!("model_not_found"));
}

/// SR-9.1: N concurrent requests (above and below the semaphore bound) all
/// complete with valid, independent responses.
#[test]
fn n_concurrent_requests_all_complete() {
    let server = spawn_server(&["--max-concurrency", "2"]);
    assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
    // 8 > 2 (bound): queueing beyond the bound backpressures, all finish.
    let mut handles = Vec::new();
    for i in 0..8 {
        let port = server.port;
        handles.push(std::thread::spawn(move || {
            let body = format!(
                r#"{{"model": "{FIXTURE_MODEL_ID}", "prompt": "task {i}", "max_tokens": 3, "stream": false}}"#
            );
            let resp = post_json(port, "/v1/completions", &body)
                .expect("concurrent completion request");
            (i, resp.status, resp.body)
        }));
    }
    for h in handles {
        let (i, status, body) = h.join().expect("thread join");
        assert_eq!(status, 200, "request {i} failed: {body}");
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("completion JSON");
        assert_eq!(parsed["object"], serde_json::json!("text_completion"));
        assert!(parsed["choices"][0]["text"].is_string());
    }
}

/// SR-9.2: one failing request (malformed body) among in-flight
/// generations does not corrupt the others.
#[test]
fn one_failing_request_does_not_corrupt_others() {
    let server = spawn_server(&["--max-concurrency", "4"]);
    assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
    let mut handles = Vec::new();
    // 4 valid requests + 1 malformed, all in flight.
    for i in 0..5 {
        let port = server.port;
        handles.push(std::thread::spawn(move || {
            if i == 2 {
                let resp = post_json(port, "/v1/completions", "{broken").expect("bad request");
                return (i, resp.status, resp.body);
            }
            let body = format!(
                r#"{{"model": "{FIXTURE_MODEL_ID}", "prompt": "job {i}", "max_tokens": 3, "stream": false}}"#
            );
            let resp = post_json(port, "/v1/completions", &body).expect("request");
            (i, resp.status, resp.body)
        }));
    }
    let mut valid = 0;
    let mut failed = 0;
    for h in handles {
        let (i, status, body) = h.join().expect("thread join");
        if i == 2 {
            assert_eq!(status, 400, "the malformed request must 400");
            let parsed: serde_json::Value = serde_json::from_str(&body).expect("envelope JSON");
            assert!(parsed.get("error").is_some());
            failed += 1;
        } else {
            assert_eq!(
                status, 200,
                "request {i} corrupted by the failing one: {body}"
            );
            let parsed: serde_json::Value = serde_json::from_str(&body).expect("completion JSON");
            assert_eq!(parsed["object"], serde_json::json!("text_completion"));
            assert!(parsed["choices"][0]["text"].is_string());
            valid += 1;
        }
    }
    assert_eq!(valid, 4);
    assert_eq!(failed, 1);
    // Server still healthy afterwards.
    let health = get(server.port, "/health").expect("health");
    assert_eq!(health.status, 200);
}
