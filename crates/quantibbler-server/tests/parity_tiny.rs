//! qk-attention E-2.2: the engine's transformer forward must match the HF
//! reference (post-quantization, fp32) on the tiny fixture within 1e-2.
//!
//! RED until tasks 2.2/2.3 wire the transformer forward; this file asserts
//! the budget loudly instead of letting the change ship unverified.

#![allow(clippy::duplicate_mod)] // design D1: tests embed the binary's modules

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

use std::path::PathBuf;

#[allow(unused_imports)] // first caller lands with qk-attention 2.2
use engine::ModelWeights;

const FIXTURE_DIR: &str = "../../tests/fixtures/parity-tiny";
const BUDGET: f32 = 1e-2;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

#[test]
fn block_forward_matches_hf_reference_within_budget() {
    let goldens: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("goldens.json")).expect("goldens.json"),
    )
    .expect("goldens JSON");

    let loaded = model::load(&fixture_dir()).expect("parity fixture loads");
    let weights = &loaded.weights;

    let input_ids: Vec<u32> = goldens["input_ids"]
        .as_array()
        .expect("input_ids array")
        .iter()
        .map(|v| v.as_u64().expect("u64 id") as u32)
        .collect();
    let golden_rows = goldens["logits"].as_array().expect("logits rows");
    assert_eq!(
        golden_rows.len(),
        input_ids.len(),
        "one golden row per position"
    );

    // Teacher-forced prefix invariance: position `pos` logits must match
    // whether the forward sees the whole sequence or just the prefix —
    // this is the property the 2.3 KV cache path has to preserve.
    let mut max_abs_diff: f32 = 0.0;
    for (pos, row) in golden_rows.iter().enumerate() {
        let seq = pos + 1;
        let logits = weights
            .forward_prefill(&input_ids[..seq])
            .expect("transformer forward (wired by qk-attention 2.2/2.3)");
        assert_eq!(logits.len(), seq * weights.vocab_size, "seq x vocab logits");
        let start = (seq - 1) * weights.vocab_size;
        let golden: Vec<f32> = row
            .as_array()
            .expect("golden row")
            .iter()
            .map(|v| v.as_f64().expect("f64 logit") as f32)
            .collect();
        for (got, want) in logits[start..].iter().zip(&golden) {
            max_abs_diff = max_abs_diff.max((got - want).abs());
        }
    }
    assert!(
        max_abs_diff <= BUDGET,
        "max |delta logit| = {max_abs_diff} exceeds the E-2.2 budget {BUDGET}"
    );
}
