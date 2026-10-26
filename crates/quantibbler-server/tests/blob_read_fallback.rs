//! qk-serve-bf16 BS-1.3 in-process: the mmap byte source and the forced read
//! fallback produce the SAME loaded model.
//!
//! The hatch is a process-global env var, so this file holds exactly ONE test:
//! a second test in the same binary could observe the hatch concurrently.

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
const PROBE: [u32; 4] = [1, 5, 7, 9];

fn logits(loaded: &model::LoadedModel) -> Vec<f32> {
    let w = &loaded.weights;
    let mut cache = engine::KvCache::new(w.n_layers, w.n_kv_heads * w.head_dim);
    w.forward_prefill(&PROBE, &mut cache).expect("prefill")
}

#[test]
fn both_byte_sources_produce_the_identical_loaded_model_bs_1_3() {
    let mapped = model::load(&blob_fixture()).expect("mmap load");
    assert_eq!(mapped.blob_load_method, Some(model::BlobLoadMethod::Mmap));

    // SAFETY (edition 2024): `set_var` is unsafe because the environment is
    // process-global. This file holds one test and no other thread reads it.
    unsafe { std::env::set_var(FORCE_READ_FALLBACK_ENV, "1") };
    let read = model::load(&blob_fixture()).expect("read-fallback load");
    unsafe { std::env::remove_var(FORCE_READ_FALLBACK_ENV) };

    assert_eq!(read.blob_load_method, Some(model::BlobLoadMethod::Read));
    assert_eq!(mapped.id, read.id);
    assert_eq!(mapped.d_model, read.d_model);
    assert_eq!(mapped.vocab_size, read.vocab_size);
    assert_eq!(mapped.weights.layers.len(), read.weights.layers.len());
    assert_eq!(
        logits(&mapped)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        logits(&read)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "mmap and read fallback must load bitwise-identical models"
    );
}
