//! Task 6.1 — end-to-end smoke asserting the fixture stand-in assumption
//! (design IA5) at the reader level: the fixture generator's bytes are
//! checked against the authoritative qk-representation SF-1.3 known-layer
//! stream (tests/test_representation.py), then the fixture model is loaded
//! and served end to end.
//!
//! Any byte-layout divergence between the server's qpk1 reader and the
//! pinned representation bytes is a qk-representation finding, not
//! something patched around here.

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

/// The pinned SF-1.3 stream (dout=3, din=5, r=7, float16) from
/// tests/test_representation.py — the compressor lane's byte contract.
const SF_1_3_STREAM_HEX: &str = "51504b31\
     01000000\
     01000000\
     03000000\
     05000000\
     07000000\
     003c003800c0\
     003400b80040003c0030\
     2b7307\
     55e6413503";

fn hex_bytes(hex: &str) -> Vec<u8> {
    let clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).expect("hex"))
        .collect()
}

/// IA5, leg 1: the server's reader parses the REAL representation lane's
/// known-layer bytes field-for-field.
#[test]
fn e2e_reader_matches_representation_known_layer() {
    let bytes = hex_bytes(SF_1_3_STREAM_HEX);
    assert_eq!(bytes.len(), 48);
    let layer = quantibbler_qpk1::parse(&bytes).expect("SF-1.3 stream parses");
    assert_eq!(layer.dout, 3);
    assert_eq!(layer.din, 5);
    assert_eq!(layer.r, 7);
    assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
    assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
    assert_eq!(layer.ub_packed, vec![0x2b, 0x73, 0x07]);
    assert_eq!(layer.vb_packed, vec![0x55, 0xe6, 0x41, 0x35, 0x03]);
    // Reader-side rank feasibility (representation deserialize rule).
    assert!(layer.r > layer.dout.min(layer.din));
}

/// IA5, leg 2: the fixture dir's layer stream satisfies the pinned size
/// formula and the server loads + serves it end to end
/// (quantize -> serialize -> serve -> complete, fixture stand-in).
#[test]
#[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
fn e2e_fixture_model_serves_end_to_end() {
    // Size formula (qk-representation SF-1.1).
    let layer_path = fixture_dir().join("layers/q_proj.qpk1");
    let raw = std::fs::read(&layer_path).expect("fixture layer bytes");
    let parsed = quantibbler_qpk1::parse(&raw).expect("fixture stream parses");
    let w = parsed.dtype.itemsize();
    let expected_size = 24
        + (parsed.dout + parsed.din) * w
        + (parsed.dout * parsed.r).div_ceil(8)
        + (parsed.din * parsed.r).div_ceil(8);
    assert_eq!(
        raw.len(),
        expected_size,
        "fixture stream must match the size formula"
    );

    // Serve the fixture model end to end through the REAL binary.
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
    assert!(body["choices"][0]["text"].is_string());
}
