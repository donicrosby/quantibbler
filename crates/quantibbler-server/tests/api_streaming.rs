//! SR-4.* (chat SSE streaming), SR-5.2 (completions streaming DONE).

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

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use common::*;

fn stream_request(port: u16, path: &str, body: &str) -> (u16, String, String, TcpStream) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write");
    stream.flush().expect("flush");
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf).expect("read");
    let text = String::from_utf8_lossy(&buf[..n]).to_string();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let ctype = text
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-type:"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_default();
    let head_end = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(text.len());
    (status, ctype, text[head_end..].to_string(), stream)
}

fn parse_data_events(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

/// SR-4.1: SSE framing — text/event-stream, data: {json} chunks, assistant
/// role on the first content delta.
#[test]
fn chat_streaming_sse_framing() {
    let server = spawn_server(&[]);
    assert!(wait_ready(server.port, Duration::from_secs(30)));
    let (status, ctype, body, mut stream) = stream_request(
        server.port,
        "/v1/chat/completions",
        &chat_body(FIXTURE_MODEL_ID, true),
    );
    assert_eq!(status, 200);
    assert!(
        ctype.starts_with("text/event-stream"),
        "content-type: {ctype}"
    );
    // Keep reading until the sentinel or timeout.
    let mut full = body;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !full.contains("data: [DONE]") && std::time::Instant::now() < deadline {
        let mut buf = [0u8; 8192];
        match std::io::Read::read(&mut stream, &mut buf) {
            Ok(0) => break,
            Ok(n) => full.push_str(&String::from_utf8_lossy(&buf[..n])),
            Err(_) => break,
        }
    }
    let events = parse_data_events(&full);
    assert!(
        events.last().map(|e| e == "[DONE]").unwrap_or(false),
        "last event must be [DONE]: {full}"
    );
    let chunks: Vec<serde_json::Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).expect("chunk JSON"))
        .collect();
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert_eq!(c["object"], serde_json::json!("chat.completion.chunk"));
    }
    // First content delta carries the assistant role.
    let first = &chunks[0];
    assert_eq!(
        first["choices"][0]["delta"]["role"],
        serde_json::json!("assistant")
    );
    assert!(first["choices"][0]["delta"]["content"].is_string());
    // Final chunk carries the finish reason; no content.
    let last = &chunks[chunks.len() - 1];
    assert!(last["choices"][0]["finish_reason"].is_string());
}

/// SR-4.2 / SR-5.2: both streaming endpoints end at the literal
/// `data: [DONE]` sentinel.
#[test]
fn chat_streaming_ends_at_done_sentinel() {
    let server = spawn_server(&[]);
    assert!(wait_ready(server.port, Duration::from_secs(30)));
    let (status, _ctype, body, mut stream) = stream_request(
        server.port,
        "/v1/chat/completions",
        &chat_body(FIXTURE_MODEL_ID, true),
    );
    assert_eq!(status, 200);
    let mut full = body;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !full.contains("data: [DONE]") && std::time::Instant::now() < deadline {
        let mut buf = [0u8; 8192];
        match std::io::Read::read(&mut stream, &mut buf) {
            Ok(0) => break,
            Ok(n) => full.push_str(&String::from_utf8_lossy(&buf[..n])),
            Err(_) => break,
        }
    }
    let events = parse_data_events(&full);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
}

#[test]
fn completions_streaming_ends_at_done_sentinel() {
    let server = spawn_server(&[]);
    assert!(wait_ready(server.port, Duration::from_secs(30)));
    let (status, _ctype, body, mut stream) = stream_request(
        server.port,
        "/v1/completions",
        &completion_body(FIXTURE_MODEL_ID, true),
    );
    assert_eq!(status, 200);
    let mut full = body;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !full.contains("data: [DONE]") && std::time::Instant::now() < deadline {
        let mut buf = [0u8; 8192];
        match std::io::Read::read(&mut stream, &mut buf) {
            Ok(0) => break,
            Ok(n) => full.push_str(&String::from_utf8_lossy(&buf[..n])),
            Err(_) => break,
        }
    }
    let events = parse_data_events(&full);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks: Vec<serde_json::Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).expect("chunk JSON"))
        .collect();
    assert!(!chunks.is_empty());
    assert_eq!(
        chunks[0]["object"],
        serde_json::json!("text_completion.chunk")
    );
    assert!(chunks[0]["choices"][0]["text"].is_string());
}

/// SR-4.3: a client disconnecting mid-stream aborts its decode only; a
/// concurrent stream completes normally.
#[test]
fn client_disconnect_aborts_only_its_stream() {
    let server = spawn_server(&["--max-concurrency", "2"]);
    assert!(wait_ready(server.port, Duration::from_secs(30)));

    // Stream A: connects, reads the first chunk, then hard-disconnects.
    let port = server.port;
    let mut stream_a = TcpStream::connect(("127.0.0.1", port)).expect("connect A");
    stream_a
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let body = chat_body(FIXTURE_MODEL_ID, true);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream_a.write_all(request.as_bytes()).expect("write A");
    let mut buf = [0u8; 2048];
    let _ = std::io::Read::read(&mut stream_a, &mut buf).expect("read A head");
    // Hard disconnect mid-stream.
    drop(stream_a);

    // Stream B: a concurrent stream that must complete unaffected.
    let mut stream_b = TcpStream::connect(("127.0.0.1", port)).expect("connect B");
    stream_b
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let body2 = completion_body(FIXTURE_MODEL_ID, true);
    let request2 = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body2}",
        body2.len()
    );
    stream_b.write_all(request2.as_bytes()).expect("write B");
    let mut full = String::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !full.contains("data: [DONE]") && std::time::Instant::now() < deadline {
        let mut buf = [0u8; 8192];
        match std::io::Read::read(&mut stream_b, &mut buf) {
            Ok(0) => break,
            Ok(n) => full.push_str(&String::from_utf8_lossy(&buf[..n])),
            Err(_) => break,
        }
    }
    assert!(
        full.contains("data: [DONE]"),
        "concurrent stream B must complete after A's disconnect: {full}"
    );
    // Server still healthy.
    let health = get(server.port, "/health").expect("health");
    assert_eq!(health.status, 200);
}
