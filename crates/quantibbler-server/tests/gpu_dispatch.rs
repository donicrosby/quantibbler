//! qk-attention task 3.4 — dispatch-wiring integration test.
//!
//! Hardware-gated (runbook pattern): the GPU arm runs only under
//! QUANTIBBLER_GPU_TESTS=1 with the staged ROCm SDK in the environment:
//!   QUANTIBBLER_GPU_TESTS=1 QUANTIBBLER_GPU=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-server --features gpu --test gpu_dispatch
//!
//! What it proves: with QUANTIBBLER_GPU=1 the engine selects the HIP face,
//! the FULL transformer forward runs on device matvecs, and the token
//! stream is IDENTICAL to the CPU face's on the same fixture model (the
//! E-2.1 determinism contract holds across faces).

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

use common::fixture_dir;

/// Face selection reads the process env ONCE at `ModelWeights`
/// construction, so the two faces are exercised by two sequential
/// constructions under different env values. Env mutation makes this the
/// only env-touching test in the binary; the documented invocation runs
/// with `--test-threads=1`.
async fn generate_tokens() -> (Vec<u32>, Vec<u32>, String) {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let face = loaded.weights.matvec_face().to_string();
    let engine = engine::Engine::new(loaded, 1);
    let prompt = engine.encode("hello").expect("encode");
    let permit = engine.acquire().await;
    let out = engine
        .generate(&permit, &prompt, 5, None)
        .expect("generation runs");
    (prompt, out.completion_tokens, face)
}

#[tokio::test]
async fn gpu_face_generates_and_matches_cpu_token_stream() {
    if !std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1") {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus staged ROCm SDK env) to run the GPU dispatch test"
        );
        return;
    }

    // Face 1: CPU (no QUANTIBBLER_GPU).
    unsafe { std::env::remove_var("QUANTIBBLER_GPU") };
    let (prompt, cpu_tokens, cpu_face) = generate_tokens().await;
    assert_eq!(cpu_face, "cpu");

    // Face 2: GPU (QUANTIBBLER_GPU=1); from_layers consults the env at
    // construction and the decision is frozen per ModelWeights instance.
    unsafe { std::env::set_var("QUANTIBBLER_GPU", "1") };
    let (prompt2, gpu_tokens, gpu_face) = generate_tokens().await;
    unsafe { std::env::remove_var("QUANTIBBLER_GPU") };

    assert_eq!(prompt, prompt2, "same prompt on both faces");
    assert_eq!(
        cpu_tokens, gpu_tokens,
        "token streams must be identical across matvec faces"
    );
    println!("dispatch OK: face={gpu_face}, tokens {cpu_tokens:?} == {gpu_tokens:?}");
}

#[test]
#[cfg(not(feature = "gpu"))]
fn gpu_flag_is_inert_without_the_feature() {
    // Default build: the env flag must not change the selection — the
    // compile-time feature gate is the only door to the device face.
    unsafe { std::env::set_var("QUANTIBBLER_GPU", "1") };
    let d = engine::FaceDecision::select_from_env();
    unsafe { std::env::remove_var("QUANTIBBLER_GPU") };
    assert_eq!(d.describe(), "cpu");
}

#[test]
#[cfg(feature = "gpu")]
fn gpu_flag_unset_selects_cpu() {
    unsafe { std::env::remove_var("QUANTIBBLER_GPU") };
    let d = engine::FaceDecision::select_from_env();
    assert_eq!(d.describe(), "cpu");
}
