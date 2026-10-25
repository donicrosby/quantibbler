//! qk-flash-attention task 3.1 — server attention-face dispatch (RED).
//!
//! Scenarios, headings quoted verbatim from the change's `specs/qk-server`
//! delta (`## ADDED Requirements` → "Attention face dispatch (frozen decision,
//! soft fallback)"). The kebab ids below are coined by `tasks.md` row 3.1, NOT
//! by the delta: `grep` over `specs/` finds neither id, because the delta's
//! scenarios carry headings only — cited this way so no docstring points at a
//! scenario label that does not exist anywhere.
//!
//! * `face-selection-health` ("Face selection and honest health labeling") —
//!   the face is chosen ONCE at model load, the label is frozen for the process
//!   lifetime, `/health` reports it alongside `matvec_face`, and with the
//!   opt-in unset the face is `cpu:scalar` with no device touched.
//! * `soft-fallback-init-failure` ("Soft fallback on init failure") — the
//!   opt-in is set but device init fails: the server still comes up, serves on
//!   `cpu:scalar`, and `/health` never shows a stale GPU label.
//!
//! Gate pattern: the face is selected at `ModelWeights` construction, so the
//! only way to observe a selection is to construct a model under that
//! environment. The `gpu:tiled` arm is hardware-gated exactly like
//! `gpu_attention_parity` (unset `QUANTIBBLER_GPU_TESTS` ⇒ clean skip notice);
//! the `soft-fallback-init-failure` arm deliberately needs NO hardware in
//! either build — see the note on `IMPOSSIBLE_DEVICE`.
//!
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-server --features gpu --test attention_dispatch

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

use std::sync::Arc;
use std::sync::Mutex;

use common::{fixture_dir, get, serve_on_port};

/// The attention-face opt-in (mirrors `QUANTIBBLER_GPU` for matvecs).
const ATTN_OPTIN: &str = "QUANTIBBLER_ATTN";

/// An index no device can have. `usize::MAX / 2` is the value the kernels
/// crate's own `fail_closed_construction_without_a_usable_device` test uses to
/// construct a deterministic init failure, so the `soft-fallback-init-failure`
/// branch is exercised on a machine WITH a usable device too — the fallback is
/// constructed, not hoped for, and the arm needs no hardware. It is only read
/// by test builds (`QUANTIBBLER_ATTN_TEST_DEVICE` is `#[cfg(test)]`-gated in
/// the engine), so no shipped binary carries the override.
const IMPOSSIBLE_DEVICE: &str = "4611686018427387903";

/// Face selection reads the process env once at `ModelWeights` construction and
/// the test-only device-index override is process-global too, so every section
/// touching either is serialized here: the default test runner is
/// multi-threaded and an unserialized mutation would make the label assertions
/// flaky rather than wrong.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Take the env lock, IGNORING poisoning. A poisoned lock only means some other
/// scenario test already failed; re-reporting that as a lock error in every
/// remaining test would bury the real assertion behind a cascade.
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Read the `attention_face` label off the REAL `GET /health` body, twice, over
/// a ready server built on the fixture model. Two reads are the point: the
/// spec's "the label stays fixed for the process lifetime" is observed, not
/// assumed.
fn health_attention_faces() -> (String, String) {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    let state = Arc::new(api::AppState::new(loaded, 1, None));
    state.set_ready();
    let port = serve_on_port(api::build_router(Arc::clone(&state)));
    let read = || {
        let resp = get(port, "/health").expect("health request");
        assert_eq!(resp.status, 200, "ready server answers 200 on /health");
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("health JSON");
        assert!(
            body.get("matvec_face").is_some(),
            "the attention face is reported ALONGSIDE matvec_face; body was {body}"
        );
        body["attention_face"]
            .as_str()
            .unwrap_or_else(|| panic!("/health has no attention_face string: {body}"))
            .to_string()
    };
    (read(), read())
}

/// The label off a freshly constructed model (no HTTP): the same frozen
/// decision `/health` reports. Used by the default-build arm only (the `gpu`
/// build observes the same decision over HTTP), so it is gated to match — an
/// unused function would red `clippy -D warnings` in the `gpu` build.
#[cfg(not(feature = "gpu"))]
fn fresh_face_label() -> String {
    let loaded = model::load(&fixture_dir()).expect("fixture model loads");
    loaded.weights.attention_face().to_string()
}

fn clear_overrides() {
    unsafe {
        std::env::remove_var(ATTN_OPTIN);
        std::env::remove_var("QUANTIBBLER_ATTN_TEST_DEVICE");
    }
}

/// `face-selection-health` (CPU arm): opt-in unset ⇒ `cpu:scalar`, and the
/// label does not move between reads.
#[test]
fn scenario_face_selection_health_default_is_cpu_scalar() {
    let _guard = env_guard();
    clear_overrides();
    let (first, second) = health_attention_faces();
    assert_eq!(
        first, "cpu:scalar",
        "with the opt-in unset the server must report the scalar CPU face"
    );
    assert_eq!(
        first, second,
        "the attention face label is frozen for the process lifetime"
    );
}

/// `face-selection-health` (default build): without the `gpu` feature the
/// opt-in is inert — the compile-time feature is the only door to a device
/// face, exactly as `gpu_dispatch.rs` pins for matvecs.
#[test]
#[cfg(not(feature = "gpu"))]
fn scenario_face_selection_health_optin_is_inert_without_the_feature() {
    let _guard = env_guard();
    clear_overrides();
    unsafe { std::env::set_var(ATTN_OPTIN, "1") };
    let label = fresh_face_label();
    clear_overrides();
    assert_eq!(
        label, "cpu:scalar",
        "without the gpu feature the opt-in must not change the face"
    );
}

/// `soft-fallback-init-failure`: opt-in set, device init fails ⇒ the server
/// still constructs (startup does not abort) and reports the CPU face, never a
/// stale GPU label. Runs WITHOUT hardware in both builds: with `gpu` compiled
/// in the probe index is the impossible one (see `IMPOSSIBLE_DEVICE`), without
/// it the opt-in is inert by construction.
#[test]
fn scenario_soft_fallback_init_failure_stays_on_cpu_scalar() {
    let _guard = env_guard();
    clear_overrides();
    unsafe {
        std::env::set_var(ATTN_OPTIN, "1");
        std::env::set_var("QUANTIBBLER_ATTN_TEST_DEVICE", IMPOSSIBLE_DEVICE);
    }
    // A failed device init must not abort startup: the load path succeeds and
    // the server answers /health on the scalar face.
    let (first, second) = health_attention_faces();
    clear_overrides();
    assert_eq!(
        first, "cpu:scalar",
        "a failed device init must soft-fall-back to the scalar CPU face"
    );
    assert_eq!(
        first, second,
        "the fallback label is frozen too — never a stale GPU label"
    );
}

/// `face-selection-health` (device arm): opt-in enabled + a usable device ⇒
/// `gpu:tiled`, frozen. Hardware-gated per the runbook pattern.
#[test]
#[cfg(feature = "gpu")]
fn scenario_face_selection_health_reports_gpu_tiled_with_a_usable_device() {
    if !std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1") {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged ROCm SDK env) to run the \
             device face-selection arm"
        );
        return;
    }
    let _guard = env_guard();
    clear_overrides();
    // No device override: index 0 is the real device, and the documented
    // invocation supplies the staged SDK.
    unsafe { std::env::set_var(ATTN_OPTIN, "1") };
    let (first, second) = health_attention_faces();
    clear_overrides();
    assert_eq!(
        first, "gpu:tiled",
        "opt-in + a usable device must select the tiled attention face"
    );
    assert_eq!(
        first, second,
        "the attention face label is frozen for the process lifetime"
    );
}
