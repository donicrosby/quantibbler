//! qk-serve-bf16 BS-9 — an fp32 blob and the bf16 repack of the same model
//! produce temperature-0 (greedy) token streams that are identical on a fixed
//! probe set, with both legs' final prefill logits finite and NOT bitwise
//! identical (the must-differ guard: bit identity would mean native bf16
//! serving never executed).

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

use std::path::Path;
use std::sync::Arc;

use common::*;

/// Five committed probe prompts of varying lengths. The fixture vocabulary is
/// the bare alphabet plus `ab`/`ba`, so every prompt tokenizes inside it.
const PROMPTS: [&str; 5] = ["a", "ab", "abc", "ab ba", "a b c d e"];

const MAX_TOKENS: usize = 8;

fn engine_for(artifact: &Path) -> engine::Engine {
    let loaded = model::load(artifact).unwrap_or_else(|e| panic!("{}: {e}", artifact.display()));
    engine::Engine::new(loaded, 1)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

/// Temperature-0 (greedy) token stream: the sampler is the argmax, so a
/// temperature knob is inert in v0.1.
fn greedy_tokens(engine: &Arc<engine::Engine>, prompt: &str) -> Vec<u32> {
    let rt = runtime();
    rt.block_on(async {
        let permit = engine.acquire().await;
        let tokens = engine.encode(prompt).expect("tokenize");
        let out = engine
            .generate(&permit, &tokens, MAX_TOKENS, None)
            .expect("generate");
        out.completion_tokens
    })
}

/// The probe prompt's token ids and its FINAL prefill logits.
fn prompt_ids(engine: &engine::Engine, prompt: &str) -> Vec<u32> {
    engine.encode(prompt).expect("tokenize")
}

fn final_prefill_logits(engine: &engine::Engine, ids: &[u32]) -> Vec<f32> {
    let w = &engine.weights;
    let mut cache = engine::KvCache::new(w.n_layers, w.n_kv_heads * w.head_dim);
    let all = w.forward_prefill(ids, &mut cache).expect("prefill");
    assert_eq!(all.len(), ids.len() * w.vocab_size);
    all[all.len() - w.vocab_size..].to_vec()
}

/// BS-9.1 / BS-9.2 / BS-9.3 on the committed blob pair.
#[test]
fn bf16_and_fp32_blobs_agree_on_the_probe_set_bs_9() {
    let fp32 = Arc::new(engine_for(&blob_fixture()));
    let bf16 = Arc::new(engine_for(&bf16_blob_fixture()));
    assert_eq!(fp32.weights.serving_dtype, model::ServingDtype::Fp32);
    assert_eq!(bf16.weights.serving_dtype, model::ServingDtype::Bf16);

    let mut deviations: Vec<f64> = Vec::new();
    for prompt in PROMPTS {
        let ids = prompt_ids(&fp32, prompt);
        let a = greedy_tokens(&fp32, prompt);
        let b = greedy_tokens(&bf16, prompt);
        assert_eq!(
            a, b,
            "BS-9.1: token streams diverged for prompt {prompt:?} (fp32={a:?} bf16={b:?})"
        );

        let la = final_prefill_logits(&fp32, &ids);
        let lb = final_prefill_logits(&bf16, &ids);
        assert_eq!(la.len(), lb.len());
        assert!(
            la.iter().chain(lb.iter()).all(|v| v.is_finite()),
            "BS-9.2: both legs' final prefill logits must be finite"
        );
        // BS-9.3: the must-differ guard. Bit identity means the native bf16
        // path never ran (a widen-at-load fell through).
        assert_ne!(
            la.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            lb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "BS-9.3: bf16-leg logits are bitwise identical to the fp32 leg for {prompt:?}"
        );
        let worst = la
            .iter()
            .zip(&lb)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0f64, f64::max);
        let median = {
            let mut deltas: Vec<f64> = la
                .iter()
                .zip(&lb)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .collect();
            deltas.sort_by(f64::total_cmp);
            deltas[deltas.len() / 2]
        };
        // BS-9.2: reported, not gated (per-linear backward error stays with
        // qk-kernels-dtype-generic).
        println!("BS-9.2 prompt {prompt:?}: worst |Δ|={worst:e}, median |Δ|={median:e}");
        assert!(worst.is_finite());
        deviations.push(worst);
    }

    let overall = deviations.iter().copied().fold(0.0f64, f64::max);
    println!(
        "BS-9.2 overall worst |Δ| logit = {overall:e} over {} prompts",
        PROMPTS.len()
    );
    assert!(
        overall < 1.0,
        "the two legs are far apart ({overall:e}); the bf16 artifact is not a repack of the fp32 one"
    );
}
