//! qk-sage-attention task 3.4 — engine-level cross-face parity scenario for the
//! quantized `gpu:sage` face on the fixture pipeline.
//!
//! Spec oracle: `openspec/changes/qk-sage-attention/specs/qk-server/spec.md`,
//! Requirement "Serving parity across the quantized attention face", scenario
//! heading "Quantized-face token identity on the fixture pipeline": greedy
//! generation through the exact scalar face and through the `gpu:sage` face
//! over identical inputs must yield IDENTICAL token streams, and the final
//! logits must agree inside the face's OWN quality gate ([SA] §4.3 — Cosine
//! Sim ≥ 0.999 and Relative L1 ≤ 0.02). The exact face's reassociation budget
//! (rel ≤ 1e-5, `cross_face_token_identity.rs`) does NOT apply here: the INT8
//! face is lossy by design, so its budget is the gate the model's shipped
//! record is admitted under, applied to the final logits row.
//!
//! The kebab id `quantized-face-token-identity` is coined by `tasks.md` row
//! 3.4, not by the delta: the delta's scenarios carry headings only, exactly as
//! the sibling flash-attention delta does.
//!
//! Why this scenario exists (and what it caught): nothing else walks the engine
//! call site `gqa_attention_sage`. The kernel battery
//! (`crates/quantibbler-kernels/tests/gpu_attention_sage.rs`) calls the kernel
//! DIRECTLY with fixtures already in the documented head-major order, and
//! `sage_dispatch.rs` covers gate admission / refusal / labels / soft fallback
//! — neither compares numbers through the engine. The shipped call site handed
//! `attention_sage` the cache's RAW row-major `(seq_kv, n_kv_heads, head_dim)`
//! stack while the kernel contract is head-major `(B, n_kv_heads, S, d)`; the
//! two orders coincide only at `n_kv_heads == 1`, and the served fixture has 2.
//! This scenario is the first to read a number off that call site, and it is
//! what caught the O(1) transposition (the sibling `gpu:tiled` face's identical
//! defect read 1.634708e-1 inf-norm rel on the same gfx1151 box, budget 1e-5).
//!
//! Gate pattern (house runbook, the same double gate as `gpu_dispatch.rs` /
//! `sage_dispatch.rs` / `cross_face_token_identity.rs`): the device arm runs
//! only when the `gpu` feature is compiled in AND `QUANTIBBLER_GPU_TESTS=1`,
//! with the staged ROCm SDK on the loader path. With the gate unset the
//! scenario prints the clean skip notice and returns (RC=0).
//!
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-server --features gpu \
//!     --test cross_face_sage_identity -- --nocapture
//!
//! Non-vacuity: face selection reads the process env ONCE at `ModelWeights`
//! construction, so the two legs are two sequential constructions under
//! different env values (the `cross_face_token_identity.rs` shape). Each leg
//! asserts its OWN label first — `cpu:scalar` and `gpu:sage` — so a device leg
//! that silently soft-fell back to the scalar face cannot pass the identity
//! check by comparing the scalar face with itself. The local build has ONE
//! `#[cfg(feature = "gpu")]` test, so no env mutex is needed.

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

/// Greedy budget for both legs, so the two streams are compared over the same
/// number of produced tokens. (`gpu`-gated with the scenario: a default build
/// has one face, so it has nothing to compare.)
#[cfg(feature = "gpu")]
const MAX_TOKENS: usize = 8;

/// The quantized face's parity budget — its OWN quality gate ([SA] §4.3,
/// spec delta `qk-server` "Sage attention face with measured quality gate":
/// "Cosine Sim ≥ 0.999 and Relative L1 ≤ 0.02"). These are the server gate's
/// constants (`crates/quantibbler-server/src/model/mod.rs`) and the gate tool's
/// (`tools/gen_attention_gate_record.py::COSINE_MIN` / `RELATIVE_L1_MAX`),
/// quoted rather than imported. They are deliberately NOT the exact face's
/// 1e-5: the INT8 face is lossy, so the exact face's budget would demand a
/// precision this face does not claim.
#[cfg(feature = "gpu")]
const COSINE_MIN: f64 = 0.999;
#[cfg(feature = "gpu")]
const RELATIVE_L1_MAX: f64 = 0.02;

/// The `gpu:sage` opt-in value (`QUANTIBBLER_ATTN=1` remains the exact tiled
/// face; unset the scalar root).
#[cfg(feature = "gpu")]
const SAGE_OPTIN: &str = "sage";

/// One greedy generation over the fixture pipeline, through whichever
/// attention face the environment selects at `ModelWeights` construction.
///
/// Returns `(face label, prompt tokens, completion tokens, final logits row)`.
/// The decoding control path mirrors `Engine::decode_start` / `decode_step`
/// exactly (prefill the whole prompt, then one trunk step per produced token,
/// argmax, stop on the fixture's EOS) — identical to
/// `cross_face_token_identity.rs`, so the only thing that can differ between
/// the two legs is the attention face itself.
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

/// The [SA] §4.3 accuracy metrics, the gate tool's own formulas
/// (`tools/gen_attention_gate_record.py::metrics`): flatten, then
/// `cosine = a·b / (‖a‖‖b‖)` and `relative L1 = Σ|a−b| / Σ|a|`. Computed in
/// f64 here for the same reason the tool does: the metrics compare two f32
/// rows, so the accumulation must not add its own error.
#[cfg(feature = "gpu")]
fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f64) {
    let a: Vec<f64> = reference.iter().map(|&v| f64::from(v)).collect();
    let b: Vec<f64> = candidate.iter().map(|&v| f64::from(v)).collect();
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    let abs_a: f64 = a.iter().map(|x| x.abs()).sum();
    let abs_diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
    (cosine, abs_diff / abs_a)
}

/// The scenario: greedy generation through the exact scalar face and the
/// quantized `gpu:sage` face over identical inputs ⇒ identical token streams,
/// final logits inside the quantized face's own [SA] §4.3 gate.
#[test]
#[cfg(feature = "gpu")]
fn scenario_quantized_face_token_identity_on_the_fixture_pipeline() {
    if !std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1") {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged ROCm SDK env) to run the \
             quantized cross-face attention parity scenario"
        );
        return;
    }

    // Leg 1 — the exact scalar face (the opt-in unset).
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };
    let (cpu_face, cpu_prompt, cpu_tokens, cpu_logits) = greedy_generation();
    assert_eq!(
        cpu_face, "cpu:scalar",
        "the reference leg must run on the exact scalar CPU face"
    );

    // Leg 2 — the quantized device face (opt-in set). `model::load` runs the
    // shipped gate first (the fixture directory carries a within-bounds
    // record) and the decision is frozen per `ModelWeights`.
    unsafe { std::env::set_var("QUANTIBBLER_ATTN", SAGE_OPTIN) };
    let (sage_face, sage_prompt, sage_tokens, sage_logits) = greedy_generation();
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };

    assert_eq!(
        sage_face, "gpu:sage",
        "the device leg must actually select the quantized face — otherwise the identity \
         check below compares the scalar face with itself"
    );
    assert_eq!(
        cpu_prompt, sage_prompt,
        "both faces must run over identical inputs"
    );
    assert_eq!(
        cpu_tokens, sage_tokens,
        "the token streams must be identical across attention faces"
    );

    assert_eq!(
        cpu_logits.len(),
        sage_logits.len(),
        "both faces must produce the same logits width"
    );

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

    let (cosine, relative_l1) = metrics(&cpu_logits, &sage_logits);

    println!(
        "quantized cross-face attention parity: device_face={sage_face}, prompt={cpu_prompt:?}, \
         tokens={cpu_tokens:?} ({} tokens, identical), context={context} keys, \
         cosine={cosine:.9e} (gate >= {COSINE_MIN}), relative L1={relative_l1:.9e} \
         (gate <= {RELATIVE_L1_MAX})",
        cpu_tokens.len()
    );

    assert!(
        cosine >= COSINE_MIN,
        "quantized-face final-logits Cosine Sim {cosine:e} is below the face's [SA] §4.3 \
         gate {COSINE_MIN:e}"
    );
    assert!(
        relative_l1 <= RELATIVE_L1_MAX,
        "quantized-face final-logits Relative L1 {relative_l1:e} is above the face's \
         [SA] §4.3 gate {RELATIVE_L1_MAX:e}"
    );
}

/// Default build: without the `gpu` feature the compile-time gate is the only
/// door to a device face, so the `gpu:sage` opt-in cannot change the selection
/// (the same inertness `sage_dispatch.rs` pins for its gate/label arms). This
/// is why the scenario above is `#[cfg(feature = "gpu")]`-only: a default build
/// has one face, and comparing it with itself would be a vacuous parity claim.
#[test]
#[cfg(not(feature = "gpu"))]
fn quantized_attention_optin_is_inert_without_the_gpu_feature() {
    unsafe { std::env::set_var("QUANTIBBLER_ATTN", "sage") };
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let label = loaded.weights.attention_face().to_string();
    unsafe { std::env::remove_var("QUANTIBBLER_ATTN") };
    assert_eq!(
        label, "cpu:scalar",
        "without the gpu feature the quantized opt-in must not change the face"
    );
}
