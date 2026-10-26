//! qk-artifact-dtype: the remainder sidecars' DECLARED dtype, end to end.
//!
//! The bf16 twin is the repack utility's own output on the committed fp32
//! fixture (`tools/repack_remainder.py`); the layer streams, the tokenizer and
//! every other file are byte-identical, only the remainder sidecars and their
//! declarations moved. That makes the forward over identical token ids a direct
//! measurement of what the dtype declaration bought — and of what it cost.

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

/// Token ids inside the fixture's 32-token vocabulary.
const INPUT_IDS: [u32; 4] = [1, 5, 7, 9];

fn forward_logits(dir: &std::path::Path) -> Vec<f32> {
    let loaded = model::load(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let weights = &loaded.weights;
    let mut cache = engine::KvCache::new(weights.n_layers, weights.n_kv_heads * weights.head_dim);
    weights
        .forward_prefill(&INPUT_IDS, &mut cache)
        .expect("transformer forward")
}

/// RD-3.1: a bf16-remainder artifact decodes within a bound DERIVED from bf16's
/// format, not borrowed from the float64 1e-12 parity tolerance.
///
/// The bound is the design.md D9 accumulation formula instantiated on this
/// fixture: bf16 rounds each element to half an ulp (2^-8 relative), and a logit
/// here is a `d_model = 4` dot product through one block's norm chain, so at
/// most 2^4 rounding contributions reach one logit — 2^-8 x 2^4 = 2^-4. The
/// `max(1, max|logit|)` factor keeps the bound honest for a fixture whose
/// logits all sit below 1. It is computed from the fp32 run's OWN tensors, so it
/// moves with the artifact instead of being a constant that can rot.
#[test]
fn bf16_twin_decode_within_tolerance_rd3_1() {
    let fp32_logits = forward_logits(&fixture_dir());
    let bf16_logits = forward_logits(&bf16_fixture_dir());
    assert_eq!(fp32_logits.len(), bf16_logits.len());

    let max_abs_logit = fp32_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let bound = 2.0f32.powi(-4) * max_abs_logit.max(1.0);
    let max_delta = fp32_logits
        .iter()
        .zip(&bf16_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Non-vacuity: identical logits would mean the declared dtype never reached
    // the forward, and the bound would then prove nothing.
    assert!(
        max_delta > 0.0,
        "identical logits mean the declared dtype never reached the forward"
    );
    assert!(
        max_delta <= bound,
        "max |delta logit| = {max_delta} exceeds the derived bf16 bound {bound} \
         (2^-4 * max(1, {max_abs_logit}))"
    );
}
