//! Shared integration-test helpers.
//!
//! Each integration test declares the server's modules at its own crate
//! root via `#[path = "../src/..."]` (design D1: one binary, no lib target
//! in v0.1 — tests exercise the binary's modules directly), then pulls in
//! this module for spawn/HTTP/router helpers.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::api::{AppState, build_router};
use crate::model;

/// Repo-root anchored fixture paths.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/server-model")
}

pub fn bad_fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/server-model-bad")
        .join(name)
}

pub const FIXTURE_MODEL_ID: &str = "server-model-fixture";

/// A spawned `quantibbler-server` process; killed on drop.
pub struct ServerProcess {
    pub child: std::process::Child,
    pub port: u16,
    stderr: Arc<Mutex<String>>,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ServerProcess {
    /// stderr accumulated so far (failure diagnostics, SR-1.2).
    pub fn diagnostic(&self) -> String {
        self.stderr.lock().expect("stderr mutex").clone()
    }

    /// Exit status, if the child has exited (non-zero on failed load).
    pub fn try_wait(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().unwrap_or_default()
    }
}

fn drain_pipes(child: &mut std::process::Child) -> Arc<Mutex<String>> {
    let sink = Arc::new(Mutex::new(String::new()));
    let err_sink = Arc::clone(&sink);
    let mut err = child.stderr.take().expect("stderr piped");
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = Read::read_to_string(&mut err, &mut buf);
        *err_sink.lock().expect("stderr mutex") = buf;
    });
    let mut out = child.stdout.take().expect("stdout piped");
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = Read::read_to_string(&mut out, &mut buf);
    });
    sink
}

/// Spawn the built server binary with an explicit model dir and port.
pub fn spawn_server_with_dir(model_dir: &Path, port: Option<u16>) -> ServerProcess {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for free port");
    let port = port.unwrap_or_else(|| listener.local_addr().expect("addr").port());
    drop(listener);
    let exe = env!("CARGO_BIN_EXE_quantibbler-server");
    let mut child = std::process::Command::new(exe)
        .args([
            "--model-dir",
            &model_dir.to_string_lossy(),
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn server binary");
    let stderr = drain_pipes(&mut child);
    ServerProcess {
        child,
        port,
        stderr,
    }
}

/// Spawn the built server binary on the valid fixture model.
pub fn spawn_server(extra_args: &[&str]) -> ServerProcess {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for free port");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    let exe = env!("CARGO_BIN_EXE_quantibbler-server");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "--model-dir",
        &fixture_dir().to_string_lossy(),
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ]);
    cmd.args(extra_args);
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn server binary");
    let stderr = drain_pipes(&mut child);
    ServerProcess {
        child,
        port,
        stderr,
    }
}

/// Minimal HTTP/1.1 result.
pub struct HttpResult {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl HttpResult {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

pub fn http_request(port: u16, request: &str) -> std::io::Result<HttpResult> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    stream.set_write_timeout(Some(Duration::from_secs(15)))?;
    stream.write_all(request.as_bytes())?;
    stream.flush()?;
    let mut bytes = Vec::new();
    let _ = stream.read_to_end(&mut bytes);
    parse_response(&bytes)
}

pub fn get(port: u16, path: &str) -> std::io::Result<HttpResult> {
    http_request(
        port,
        &format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
    )
}

pub fn post_json(port: u16, path: &str, body: &str) -> std::io::Result<HttpResult> {
    http_request(
        port,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    )
}

/// Raw POST with full control over headers (auth tests).
pub fn post_raw(port: u16, path: &str, headers: &str, body: &str) -> std::io::Result<HttpResult> {
    http_request(
        port,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    )
}

/// Send a request and read the FIRST socket chunk only (headers + start of
/// the SSE body).
pub fn http_first_chunk(
    port: u16,
    request: &str,
) -> std::io::Result<(u16, String, String, TcpStream)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    stream.write_all(request.as_bytes())?;
    stream.flush()?;
    let mut buf = [0u8; 4096];
    let n = stream.read(&mut buf)?;
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
    Ok((status, ctype, text[head_end..].to_string(), stream))
}

/// Raw socket chunk read (SSE follow-ups).
pub fn read_chunk(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf[..n]).to_string())
}

fn parse_response(raw: &[u8]) -> std::io::Result<HttpResult> {
    let splitter = find(raw, b"\r\n\r\n").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "no header terminator")
    })?;
    let head = String::from_utf8_lossy(&raw[..splitter]).to_string();
    let body = String::from_utf8_lossy(&raw[splitter + 4..]).to_string();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad status line"))?;
    let headers = head
        .lines()
        .skip(1)
        .filter_map(|l| {
            l.split_once(':')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        })
        .collect();
    Ok(HttpResult {
        status,
        headers,
        body,
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// An in-process serving router over the fixture model (ready).
pub fn serving_router() -> (Arc<AppState>, axum::Router) {
    ready_router()
}

/// A router whose engine fails every generation (SR-7.3 injection).
pub fn failing_router() -> (Arc<AppState>, axum::Router) {
    let (state, router) = ready_router();
    state.engine.fail_generate.store(true, Ordering::SeqCst);
    (state, router)
}

/// A router that is loaded but NOT ready (SR-6.2 / SR-7.2).
pub fn not_ready_router() -> (Arc<AppState>, axum::Router) {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let state = Arc::new(AppState::new(loaded, 4, None));
    let router = build_router(Arc::clone(&state));
    (state, router)
}

fn ready_router() -> (Arc<AppState>, axum::Router) {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let state = Arc::new(AppState::new(loaded, 4, None));
    state.set_ready();
    let router = build_router(Arc::clone(&state));
    (state, router)
}

/// Serve a router on an ephemeral local port on a DEDICATED OS thread with
/// its own runtime, so tests may drive blocking HTTP from any thread
/// (including inside a current-thread block_on) without starving the
/// server's accept loop. The tokio listener is created INSIDE that thread's
/// runtime context so its IO registration lives on the server thread, not
/// the caller's (possibly starved) reactor.
pub fn serve_on_port(router: axum::Router) -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.set_nonblocking(true).expect("nonblocking");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server thread runtime");
        let _guard = rt.enter();
        let listener =
            tokio::net::TcpListener::from_std(listener).expect("tokio listener on server thread");
        rt.block_on(async {
            let _ = axum::serve(listener, router).await;
        });
    });
    port
}

/// Poll /health until 200 (or deadline expires).
pub fn wait_ready(port: u16, deadline: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if let Ok(resp) = get(port, "/health") {
            if resp.status == 200 {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    false
}

/// SR-1.2 rejection classes -> (variant dir name, failing file that MUST be
/// named in the startup diagnostic).
pub fn corrupted_variants() -> Vec<(&'static str, &'static str)> {
    vec![
        ("missing_manifest", "manifest.toml"),
        ("missing_tokenizer", "tokenizer.json"),
        ("malformed_tokenizer", "tokenizer.json"),
        ("missing_layer_file", "block0.mlp.fc.qpk1"),
        ("bad_magic", "block0.mlp.fc.qpk1"),
        ("wrong_version", "block0.mlp.fc.qpk1"),
        ("unknown_dtype_tag", "block0.mlp.fc.qpk1"),
        ("rank_zero", "block0.mlp.fc.qpk1"),
        ("rank_infeasible", "block0.mlp.fc.qpk1"),
        ("truncated", "block0.mlp.fc.qpk1"),
        ("trailing_bytes", "block0.mlp.fc.qpk1"),
        // Hardening-class variants (loader rejection rules):
        ("path_escape", "rejected"),
        ("symlink_layer", "not a regular file"),
        ("sidecar_too_big", "expected exactly"),
        ("bad_weight_tying", "unknown weight_tying"),
        ("eos_out_of_range", "eos_token_id"),
    ]
}

/// Standard chat body naming `model`.
pub fn chat_body(model: &str, stream: bool) -> String {
    format!(
        r#"{{"model": "{model}", "messages": [{{"role": "user", "content": "ab"}}], "max_tokens": 4, "stream": {stream}}}"#
    )
}

/// Standard completion body naming `model`.
pub fn completion_body(model: &str, stream: bool) -> String {
    format!(r#"{{"model": "{model}", "prompt": "ab", "max_tokens": 4, "stream": {stream}}}"#)
}
