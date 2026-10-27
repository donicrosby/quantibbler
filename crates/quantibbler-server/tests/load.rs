//! SR-1.1 / SR-1.2 / SR-6.1 / SR-6.2: fail-closed startup, readiness, and
//! health reporting. Exercises the REAL binary process against the REAL
//! fixture model directory and every corrupted variant.

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

use std::time::Duration;

use common::*;

/// SR-1.1: the server reports ready via GET /health only after the model
/// directory loads, and the served id matches the loaded directory.
#[test]
fn load_ready_only_after_success() {
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        let ready = wait_ready(server.port, Duration::from_secs(30));
        assert!(
            ready,
            "{}: server never became ready; stderr: {}",
            artifact.display(),
            server.diagnostic()
        );
        let health = get(server.port, "/health").expect("health request");
        assert_eq!(health.status, 200);
        let body: serde_json::Value = serde_json::from_str(&health.body).expect("health JSON");
        assert_eq!(body["ready"], serde_json::json!(true));
        assert_eq!(body["model"], serde_json::json!(FIXTURE_MODEL_ID));
    });
}

/// SR-1.2 (parameterized over every corrupted variant): a failed load never
/// reaches ready, the process exits non-zero, and the diagnostic names the
/// failing file.
#[test]
fn load_failure_never_ready() {
    for (variant, failing_file) in corrupted_variants() {
        let dir = bad_fixture_dir(variant);
        let mut server = spawn_server_with_dir(&dir, None);
        // Give the process time to fail (it must NOT serve).
        let mut exited = None;
        for _ in 0..200 {
            if let Some(status) = server.try_wait() {
                exited = Some(status);
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        let status = exited.unwrap_or_else(|| {
            panic!(
                "variant {variant}: server kept running instead of failing; stderr: {}",
                server.diagnostic()
            )
        });
        assert!(
            !status.success(),
            "variant {variant}: exit status was success"
        );
        // The child exited, but its stderr is captured by an async drain
        // thread, so a read right after `try_wait()` can return only what that
        // thread has published SO FAR — under load the load error can still be
        // in flight. Poll until the expected line lands (the poll-not-race
        // pattern `wait_for_log` uses) before asserting.
        let diag = diagnostic_containing(&server, failing_file);
        assert!(
            diag.contains(failing_file),
            "variant {variant}: diagnostic must name {failing_file}; got: {diag}"
        );
        // Nothing was ever served: the port must refuse connections.
        assert!(
            std::net::TcpStream::connect(("127.0.0.1", server.port)).is_err()
                || get(server.port, "/health").map(|r| r.status).unwrap_or(0) == 0,
            "variant {variant}: something answered on the port after a failed load"
        );
    }
}

/// SR-6.1 / SR-6.2 (in-process): /health is 503 while not ready and 200
/// once ready; inference requests before readiness get the 503 envelope.
#[test]
fn health_readiness_gate() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    for_each_serving_source(|artifact| {
        rt.block_on(async {
            let (_state, router) = not_ready_router_for(artifact);
            let port = serve_on_port(router);
            // 503 while not ready (poll a few times; the router serves already).
            let mut saw_503 = false;
            for _ in 0..50 {
                if let Ok(resp) = get(port, "/health") {
                    if resp.status == 503 {
                        saw_503 = true;
                        let body: serde_json::Value =
                            serde_json::from_str(&resp.body).expect("health JSON");
                        assert_eq!(body["ready"], serde_json::json!(false));
                        break;
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(saw_503, "not-ready server never answered 503 on /health");
            // Inference before readiness -> 503 envelope with model_not_ready.
            let resp = post_json(
                port,
                "/v1/chat/completions",
                &chat_body(FIXTURE_MODEL_ID, false),
            )
            .expect("chat request");
            assert_eq!(resp.status, 503);
            let body: serde_json::Value = serde_json::from_str(&resp.body).expect("envelope JSON");
            assert_eq!(body["error"]["code"], serde_json::json!("model_not_ready"));
            assert_eq!(body["error"]["type"], serde_json::json!("server_error"));
        });
    });
}

/// Poll a failed server's captured stderr for `needle` (the async drain thread
/// publishes incrementally, so the load error may not be visible the instant
/// the child exits), then return the buffer either way so the failure message
/// carries what was actually captured.
fn diagnostic_containing(server: &ServerProcess, needle: &str) -> String {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let diag = server.diagnostic();
        if diag.contains(needle) || std::time::Instant::now() > deadline {
            return diag;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
