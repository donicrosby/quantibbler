//! qk-serve-bf16 BS-6 -- non-streaming responses report `timings.prompt_ms`;
//! streaming responses do not.

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

use std::time::Instant;

use common::*;

/// BS-6.1: both non-streaming surfaces carry a positive `prompt_ms` bounded by
/// the request wall time. The `+1` absorbs the millisecond ceiling `prompt_ms`
/// applies, so a sub-millisecond prefill reports 1 and never 0.
#[test]
fn non_streaming_responses_carry_prompt_ms_bs_6_1() {
    for_each_serving_source(|artifact| {
        let (_state, router) = ready_router_for(artifact);
        let port = serve_on_port(router);

        for (path, body) in [
            ("/v1/chat/completions", chat_body(FIXTURE_MODEL_ID, false)),
            ("/v1/completions", completion_body(FIXTURE_MODEL_ID, false)),
        ] {
            let start = Instant::now();
            let resp = post_json(port, path, &body).expect("request");
            let wall_ms = start.elapsed().as_millis() as u64;
            assert_eq!(resp.status, 200);
            let value: serde_json::Value = serde_json::from_str(&resp.body).expect("JSON body");
            let ms = value["timings"]["prompt_ms"]
                .as_u64()
                .unwrap_or_else(|| panic!("no timings.prompt_ms on {path}: {resp:?}"));
            assert!((1..=wall_ms + 1).contains(&ms));
        }
    });
}

/// BS-6.2: a streaming response carries no `timings` field in any SSE chunk.
#[test]
fn streaming_responses_omit_timings_bs_6_2() {
    for_each_serving_source(|artifact| {
        let (_state, router) = ready_router_for(artifact);
        let port = serve_on_port(router);
        let body = chat_body(FIXTURE_MODEL_ID, true);
        let request = String::new()
            + "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
            + &body.len().to_string()
            + "\r\nConnection: close\r\n\r\n"
            + &body;
        let resp = http_request(port, &request).expect("stream request");
        assert_eq!(resp.status, 200);
        assert!(!resp.body.contains("timings"));
    });
}
