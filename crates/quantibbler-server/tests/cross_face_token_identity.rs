//! qk-flash-attention task 3.3 — hardware parity scenario
//! `cross-face-token-identity` on the fixture pipeline.
//!
//! Spec oracle: `openspec/changes/qk-flash-attention/specs/qk-server/spec.md`,
//! Requirement "Serving parity across attention faces", scenario heading
//! "Cross-face token identity on the fixture pipeline": greedy generation
//! through EACH attention face over identical inputs must yield IDENTICAL
//! token streams, and the final logits must agree inside the reassociation
//! budget (rel ≤ 1e-5). Bit-identity is explicitly NOT required — this face
//! reassociates.
//!
//! The kebab id `cross-face-token-identity` is coined by `tasks.md` row 3.3,
//! not by the delta: the delta's scenarios carry headings only (`grep` over
//! `changes/qk-flash-attention/specs/` finds no kebab ids). Cited this way so
//! no docstring points at a scenario label that exists nowhere.
//!
//! Gate pattern (house runbook, same double gate as `gpu_dispatch.rs` /
//! `gpu_attention_parity.rs`): the device arm runs only when the `gpu` feature
//! is compiled in AND `QUANTIBBLER_GPU_TESTS=1`, with the staged ROCm SDK on
//! the loader path. With the gate unset the scenario prints the clean skip
//! notice and returns (RC=0).
//!
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-server --features gpu \
//!     --test cross_face_token_identity -- --nocapture
//!
//! Non-vacuity: face selection reads the process env ONCE at `ModelWeights`
//! construction, so the two legs are two sequential constructions under
//! different env values (the `gpu_dispatch.rs` shape). Each leg asserts its
//! OWN label first — `cpu:scalar` and `gpu:tiled` — so a device leg that
//! silently soft-fell back to the scalar face cannot pass the identity check
//! by comparing the scalar face with itself. That label assert is the guard
//! that makes the token-identity claim mean something.

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

/// Greedy budget for both legs. One value for both, so the two streams are
/// compared over the same number of produced tokens. (`gpu`-gated with the
/// scenario: a default build has one face, so it has nothing to compare.)
#[cfg(feature = "gpu")]
const MAX_TOKENS: usize = 8;

/// The serving contract's reassociation budget (spec delta: "final logits
/// agree with relative error ≤ 1e-5").
#[cfg(feature = "gpu")]
const LOGITS_RTOL: f32 = 1e-5;

/// One greedy generation over the fixture pipeline, through whichever
/// attention face the environment selects at `ModelWeights` construction.
///
/// Returns `(face label, prompt tokens, completion tokens, final logits row)`.
/// The decoding control path mirrors `Engine::decode_start` / `decode_step`
/// exactly (prefill the whole prompt, then one trunk step per produced token,
/// argmax, stop on the fixture's EOS), so both legs take identical steps; the
/// only thing that can differ between them is the attention face itself.
#[cfg(feature = "gpu")]
fn greedy_generation() -> (String, Vec<u32>, Vec<u32>, Vec<f32>) {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let weights = &loaded.weights;
    let face = weights.attention_face().to_string();

    let prompt: Vec<u32> = loaded
        .tokenizer
        .encode("hello ab", false)
        .expect("tokenize the fixture prompt")
        .get_ids()
        .to_vec();
    assert!(!prompt.is_empty(), "the fixture prompt tokenizes non-empty");

    let vocab = weights.vocab_size;
    let mut cache = engine::KvCache::new(weights.n_layers, weights.n_kv_heads * weights.head_dim);

    // Prefill: the trunk over every prompt position, on the selected face.
    let flat = weights
        .forward_prefill(&prompt, &mut cache)
        .expect("prefill forward");
    assert_eq!(
        flat.len(),
        prompt.len() * vocab,
        "seq x vocab prefill logits"
    );
    let mut logits = flat[(prompt.len() - 1) * vocab..].to_vec();

    let mut tokens = Vec::with_capacity(MAX_TOKENS);
    for _ in 0..MAX_TOKENS {
        let next = sampler::greedy(&logits);
        tokens.push(next);
        if next == loaded.eos_token_id {
            break; // the engine's session stops on EOS and drops the token
        }
        let hidden = weights.embed(next).expect("embed the produced token");
        logits = weights
            .forward(&hidden, &mut cache)
            .expect("decode forward");
    }
    (face, prompt, tokens, logits)
}

/// The scenario: greedy generation through each face over identical inputs ⇒
/// identical token streams, final logits within the reassociation budget.
#[test]
#[cfg(feature = "gpu")]
fn scenario_cross_face_token_identity_on_the_fixture_pipeline() {
    if !std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1") {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged ROCm SDK env) to run the \
             cross-face attention parity scenario"
        );
        return;
    }

    // Leg 1 — the scalar face (the opt-in unset).
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };
    let (cpu_face, cpu_prompt, cpu_tokens, cpu_logits) = greedy_generation();
    assert_eq!(
        cpu_face, "cpu:scalar",
        "the reference leg must run on the scalar CPU face"
    );

    // Leg 2 — the tiled device face (opt-in set). `from_layers` consults the
    // env at construction and the decision is frozen per `ModelWeights`.
    unsafe { std::env::set_var("QUANTIBBLER_ATTN", "1") };
    let (gpu_face, gpu_prompt, gpu_tokens, gpu_logits) = greedy_generation();
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };

    assert_eq!(
        gpu_face, "gpu:tiled",
        "the device leg must actually select the tiled face — otherwise the identity check \
         below compares the scalar face with itself"
    );
    assert_eq!(
        cpu_prompt, gpu_prompt,
        "both faces must run over identical inputs"
    );
    assert_eq!(
        cpu_tokens, gpu_tokens,
        "the token streams must be identical across attention faces"
    );

    assert_eq!(
        cpu_logits.len(),
        gpu_logits.len(),
        "both faces must produce the same logits width"
    );
    let scale = cpu_logits
        .iter()
        .chain(gpu_logits.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()));
    let mut max_abs = 0.0f32;
    let mut max_elem_rel = 0.0f32;
    for (a, e) in cpu_logits.iter().zip(gpu_logits.iter()) {
        let delta = (a - e).abs();
        max_abs = max_abs.max(delta);
        if *a != 0.0 {
            max_elem_rel = max_elem_rel.max(delta / a.abs());
        }
    }
    let inf_norm_rel = if scale > 0.0 {
        max_abs / scale
    } else {
        max_abs
    };

    // Non-vacuity of the compared row: it must be a MULTI-KEY attention step —
    // over a single cached position the device call degenerates and the
    // equality says nothing. The prefill of the prompt guarantees it (the
    // fixture prompt is 6 tokens, so the first compared row already attends
    // over 6 keys), and the assertion keeps that true if the fixture or the
    // prompt ever shrinks.
    let context = cpu_prompt.len() + cpu_tokens.len();
    assert!(
        context > 1,
        "the compared logits row must attend over more than one key, got context {context}"
    );

    println!(
        "cross-face attention parity: device_face={gpu_face}, prompt={cpu_prompt:?}, \
         tokens={cpu_tokens:?} ({} tokens, identical), context={context} keys, \
         final-logits inf-norm rel={inf_norm_rel:.6e}, max|delta|={max_abs:.6e}, \
         max per-element rel={max_elem_rel:.6e}, budget={LOGITS_RTOL:.0e}",
        cpu_tokens.len()
    );

    assert!(
        inf_norm_rel <= LOGITS_RTOL,
        "final logits inf-norm relative error {inf_norm_rel:e} exceeds the reassociation \
         budget {LOGITS_RTOL:e}"
    );
}

/// Default build: without the `gpu` feature the compile-time gate is the only
/// door to a device face, so the opt-in cannot change the selection (the same
/// inertness `gpu_dispatch.rs` pins for matvecs). This is why the scenario
/// above is `#[cfg(feature = "gpu")]`-only: a default build has one face, and
/// comparing it with itself would be a vacuous parity claim.
#[test]
#[cfg(not(feature = "gpu"))]
fn attention_optin_is_inert_without_the_gpu_feature() {
    unsafe { std::env::set_var("QUANTIBBLER_ATTN", "1") };
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let label = loaded.weights.attention_face().to_string();
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };
    assert_eq!(
        label, "cpu:scalar",
        "without the gpu feature the attention opt-in must not change the face"
    );
}
