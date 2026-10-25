//! qk-sage-attention tasks 3.1–3.3 — the `gpu:sage` server face and its
//! startup quality gate (RED first; GREEN with the face + gate + gate tool).
//!
//! Scenarios, headings quoted verbatim from the change's `specs/qk-server`
//! delta (`## ADDED Requirements` → "Sage attention face with measured quality
//! gate" / "Quality gate provenance is shipped and reproducible"). The kebab
//! ids below are the ones `tasks.md` row 3.1 names:
//!
//! * `gate-pass-enables` — "Gate pass enables the face": the opt-in is set and
//!   the model's captured gate record is within bounds ⇒ the server starts with
//!   `"attention_face": "gpu:sage"` and the label stays fixed for the process
//!   lifetime.
//! * `gate-fail-refuses` — "Gate fail refuses the face": the opt-in is set and
//!   the record is missing or out of bounds ⇒ startup aborts with a typed error
//!   naming the gate. It does NOT silently fall back; quality is never the
//!   fallback trigger. Runnable WITHOUT hardware (the gate is checked before any
//!   device is touched), in both feature states.
//! * `soft-fallback-device-only` — "Soft fallback on device failure only": the
//!   gate passes but device init fails ⇒ the face falls back to the exact tiled
//!   face (or scalar). Needs no hardware in either build — see
//!   `IMPOSSIBLE_DEVICE`.
//! * `gate-record-regenerates` — "Gate record regeneration matches": the
//!   committed gate tool (`tools/gen_attention_gate_record.py`) re-derives the
//!   shipped record's metrics from the model directory + its activation fixture.
//!   The re-derivation itself is a command-level check (the tool's own `--check`
//!   mode, run in this card's evidence); what THIS file pins automatically is
//!   that the shipped record is present, bound to the served model, inside the
//!   clauses it declares, and names an activation fixture that is shipped.
//!
//! Gate pattern: the face is selected at `ModelWeights` construction, so the
//! only way to observe a selection is to load a model under that environment.
//! The `gpu:sage` label arm is hardware-gated exactly like
//! `gpu_attention_parity` (unset `QUANTIBBLER_GPU_TESTS` ⇒ clean skip notice).
//!
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-server --features gpu --test sage_dispatch

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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use common::{fixture_dir, get, serve_on_port};

/// The attention-face opt-in (mirrors `QUANTIBBLER_GPU` for matvecs).
const ATTN_OPTIN: &str = "QUANTIBBLER_ATTN";

/// The `gpu:sage` opt-in value (`QUANTIBBLER_ATTN=1` remains the tiled face).
const SAGE_OPTIN: &str = "sage";

/// The shipped gate record, resolved inside the model directory (spec:
/// "shipped with the model directory").
const GATE_RECORD: &str = "attention_gate.json";

/// An index no device can have. `usize::MAX / 2` is the value the kernels
/// crate's own `fail_closed_construction_without_a_usable_device` test uses to
/// construct a deterministic init failure, so the `soft-fallback-device-only`
/// branch is exercised on a machine WITH a usable device too. Only read by test
/// builds (`QUANTIBBLER_ATTN_TEST_DEVICE` is `#[cfg(test)]`-gated in the
/// engine).
const IMPOSSIBLE_DEVICE: &str = "4611686018427387903";

/// Face selection reads the process env once at model load and the test-only
/// device-index override is process-global too, so every section touching
/// either is serialized here.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn clear_overrides() {
    unsafe {
        std::env::remove_var(ATTN_OPTIN);
        std::env::remove_var("QUANTIBBLER_ATTN_TEST_DEVICE");
    }
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
        body["attention_face"]
            .as_str()
            .unwrap_or_else(|| panic!("/health has no attention_face string: {body}"))
            .to_string()
    };
    (read(), read())
}

/// A private copy of the fixture model directory, so a single file (the gate
/// record) can be doctored without touching the shared fixture the sibling
/// suites read. Removed on drop.
struct TempModelDir(PathBuf);

impl TempModelDir {
    fn new(name: &str) -> Self {
        let dst = std::env::temp_dir().join(format!("sage-gate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dst);
        copy_tree(&fixture_dir(), &dst);
        Self(dst)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// Rewrite the gate record through `edit`, or create it from the shipped
    /// record when `from_shipped` is set.
    fn edit_record(&self, edit: impl FnOnce(&mut serde_json::Value)) {
        let path = self.0.join(GATE_RECORD);
        let shipped = std::fs::read_to_string(fixture_dir().join(GATE_RECORD))
            .expect("shipped gate record is readable");
        let mut value: serde_json::Value =
            serde_json::from_str(&shipped).expect("shipped gate record is JSON");
        edit(&mut value);
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&value).expect("record JSON"),
        )
        .expect("record written");
    }
}

impl Drop for TempModelDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("create temp model dir");
    for entry in std::fs::read_dir(src).expect("read fixture dir") {
        let entry = entry.expect("dir entry");
        let target = dst.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy fixture file");
        }
    }
}

/// `gate-pass-enables`: the shipped record is admitted, and the load path runs
/// the gate (the SAME directory minus the record refuses — so the admission
/// observed here is the gate's, not a default).
#[test]
fn scenario_gate_pass_enables_the_shipped_record_is_admitted() {
    let _guard = env_guard();
    clear_overrides();
    unsafe { std::env::set_var(ATTN_OPTIN, SAGE_OPTIN) };

    let record = model::attention_gate_check(&fixture_dir(), common::FIXTURE_MODEL_ID)
        .expect("the shipped gate record is within bounds");
    let loaded = model::load(&fixture_dir()).expect("the shipped record admits the model");

    let missing = TempModelDir::new("pass-control");
    std::fs::remove_file(missing.path().join(GATE_RECORD)).expect("drop the record");
    let refused = load_err(missing.path(), "the same dir minus the record refuses");

    // The optional PPL clause, declared: a within-bounds delta is admitted and
    // carried (the other half of the clause is refused in the fail scenario).
    let declared = TempModelDir::new("pass-ppl");
    declared.edit_record(|r| {
        r["ppl"] = serde_json::json!({
            "eval_set": "unit-test eval",
            "delta": 0.011,
            "bound": 0.05,
        })
    });
    let with_ppl = model::attention_gate_check(declared.path(), common::FIXTURE_MODEL_ID)
        .expect("a within-bounds PPL clause is admitted");
    clear_overrides();

    assert_eq!(
        with_ppl.ppl.as_ref().map(|p| p.delta),
        Some(0.011),
        "the declared PPL clause survives parsing"
    );

    assert!(
        refused.contains(GATE_RECORD),
        "refusal names the record: {refused}"
    );
    println!(
        "gate pass: worst cosine {:.9} (>= {}), worst relative L1 {:.9} (<= {}) over {} calls; \
         face={}",
        record.worst.cosine,
        record.clauses.cosine_min,
        record.worst.relative_l1,
        record.clauses.relative_l1_max,
        record.calls.len(),
        loaded.weights.attention_face(),
    );
}

/// `gate-pass-enables` (device arm): opt-in + a within-bounds record + a usable
/// device ⇒ `gpu:sage`, frozen. Hardware-gated per the runbook pattern.
#[test]
#[cfg(feature = "gpu")]
fn scenario_gate_pass_enables_gpu_sage_with_a_usable_device() {
    if !std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1") {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged ROCm SDK env) to run the \
             gpu:sage face-selection arm"
        );
        return;
    }
    let _guard = env_guard();
    clear_overrides();
    unsafe { std::env::set_var(ATTN_OPTIN, SAGE_OPTIN) };
    let (first, second) = health_attention_faces();
    clear_overrides();
    assert_eq!(
        first, "gpu:sage",
        "a within-bounds record + a usable device must select the quantized face"
    );
    assert_eq!(
        first, second,
        "the attention face label is frozen for the process lifetime"
    );
}

/// `gate-fail-refuses`: every refusal class — missing, unparseable, a per-call
/// metric out of bounds, a `worst` that contradicts its own calls, a looser
/// declared clause set, an out-of-bounds PPL clause, and a record from another
/// model — aborts startup with a typed error naming the gate and the failing
/// clause, in-process AND (for the missing case) at the process boundary. Needs
/// no hardware in either build: the gate is checked before any device is
/// touched, so it refuses identically in a default build.
#[test]
fn scenario_gate_fail_refuses_the_face() {
    let _guard = env_guard();
    clear_overrides();
    unsafe { std::env::set_var(ATTN_OPTIN, SAGE_OPTIN) };

    // (1) missing record
    let missing = TempModelDir::new("fail-missing");
    std::fs::remove_file(missing.path().join(GATE_RECORD)).expect("drop the record");
    let err = load_err(missing.path(), "a missing record must refuse the face");
    assert!(
        err.contains("attention gate"),
        "refusal names the gate: {err}"
    );
    assert!(err.contains(GATE_RECORD), "refusal names the record: {err}");

    // (2) a per-call metric out of bounds (a stale quality claim)
    let oob = TempModelDir::new("fail-oob");
    oob.edit_record(|r| {
        r["calls"][3]["cosine"] = serde_json::json!(0.5);
        r["worst"]["cosine"] = serde_json::json!(0.5);
    });
    let err = load_err(oob.path(), "an out-of-bounds record must refuse");
    assert!(
        err.contains("attention gate"),
        "refusal names the gate: {err}"
    );
    assert!(
        err.contains("cosine"),
        "refusal names the failing clause: {err}"
    );

    // (3) a `worst` that contradicts the record's own per-call metrics: a
    // doctored summary is refused on its own numbers, and the refusal names the
    // disagreement.
    let doctored = TempModelDir::new("fail-doctored");
    doctored.edit_record(|r| r["worst"]["relative_l1"] = serde_json::json!(0.001));
    let err = load_err(doctored.path(), "a doctored worst must refuse");
    assert!(err.contains("attention gate"), "names the gate: {err}");
    assert!(
        err.contains("aggregate"),
        "the refusal names the disagreement: {err}"
    );

    // (4) a record generated under a looser clause set than the gate enforces
    let lax = TempModelDir::new("fail-lax-clause");
    lax.edit_record(|r| r["clauses"]["cosine_min"] = serde_json::json!(0.5));
    let err = load_err(lax.path(), "a looser declared clause must refuse");
    assert!(
        err.contains("cosine_min"),
        "the refusal names the widened clause: {err}"
    );

    // (5) a declared PPL delta outside the bound recorded at the gate run
    let ppl = TempModelDir::new("fail-ppl");
    ppl.edit_record(|r| {
        r["ppl"] = serde_json::json!({
            "eval_set": "unit-test eval",
            "delta": 5.0,
            "bound": 0.05,
        })
    });
    let err = load_err(ppl.path(), "an out-of-bounds PPL clause must refuse");
    assert!(
        err.contains("ppl_delta"),
        "the refusal names the PPL clause: {err}"
    );

    // (6) another model's record (stale provenance)
    let stale = TempModelDir::new("fail-stale");
    stale.edit_record(|r| r["model_id"] = serde_json::json!("some-other-model"));
    let err = load_err(stale.path(), "another model's record must refuse");
    assert!(err.contains("model_id"), "refusal names the binding: {err}");

    // (7) not JSON at all
    let garbage = TempModelDir::new("fail-garbage");
    std::fs::write(garbage.path().join(GATE_RECORD), "{not json").expect("write garbage");
    let err = load_err(garbage.path(), "an unparseable record must refuse");
    assert!(err.contains("attention gate"), "names the gate: {err}");

    // (8) at the process boundary: the binary exits non-zero and its diagnostic
    //     names the gate (SR-1.2 shape, spec: "startup aborts").
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_quantibbler-server"))
        .args([
            "--model-dir",
            missing.path().to_str().expect("utf-8 path"),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ])
        .env(ATTN_OPTIN, SAGE_OPTIN)
        .output()
        .expect("spawn the server binary");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    clear_overrides();

    assert!(
        !out.status.success(),
        "startup must abort, not serve degraded attention; stderr: {stderr}"
    );
    assert!(
        stderr.contains("attention gate"),
        "the startup diagnostic names the gate; stderr: {stderr}"
    );
}

/// `soft-fallback-device-only`: the gate passes but device init fails ⇒ the
/// exact face, never a stale `gpu:sage` label. Runs WITHOUT hardware in both
/// builds: with `gpu` compiled in the probe index is impossible, without it the
/// opt-in is inert by construction.
#[test]
fn scenario_soft_fallback_device_only_keeps_the_exact_face() {
    let _guard = env_guard();
    clear_overrides();
    unsafe {
        std::env::set_var(ATTN_OPTIN, SAGE_OPTIN);
        std::env::set_var("QUANTIBBLER_ATTN_TEST_DEVICE", IMPOSSIBLE_DEVICE);
    }
    let (first, second) = health_attention_faces();
    clear_overrides();
    assert_eq!(
        first, "cpu:scalar",
        "a failed sage device init must fall back to the exact face"
    );
    assert_eq!(
        first, second,
        "the fallback label is frozen too — never a stale gpu:sage"
    );
}

/// `gate-record-regenerates` (the automatic half): the shipped record is
/// present, bound to the served model, inside the clauses it declares, and names
/// an activation fixture that is shipped with the tree. The numeric
/// re-derivation is `tools/gen_attention_gate_record.py --check`, which recomputes
/// these very numbers and fails on drift.
#[test]
fn scenario_gate_record_regenerates_matches_the_shipped_fixture() {
    let record = model::attention_gate_check(&fixture_dir(), common::FIXTURE_MODEL_ID)
        .expect("shipped record admits the fixture model");
    let source = repo_root().join(&record.activations_source);
    assert!(
        source.is_file(),
        "the record names its activation fixture; {} is missing",
        source.display()
    );
    let cases: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&source).expect("activation fixture readable"),
    )
    .expect("activation fixture JSON");
    assert_eq!(
        cases.as_array().map(Vec::len),
        Some(record.calls.len()),
        "the record's call count must match the fixture it was generated from"
    );
    assert!(
        record.worst.cosine >= record.clauses.cosine_min,
        "worst cosine {} below the declared clause {}",
        record.worst.cosine,
        record.clauses.cosine_min
    );
    assert!(
        record.worst.relative_l1 <= record.clauses.relative_l1_max,
        "worst relative L1 {} above the declared clause {}",
        record.worst.relative_l1,
        record.clauses.relative_l1_max
    );
    println!(
        "record: model_id={} calls={} worst cos={:.9} relL1={:.9} ppl_declared={} \
         activations={}",
        record.model_id,
        record.calls.len(),
        record.worst.cosine,
        record.worst.relative_l1,
        record.ppl.is_some(),
        record.activations_source,
    );
}

/// `model::load`'s error, or a panic naming the expectation (`LoadedModel`
/// carries no `Debug`, so `expect_err` is not available).
fn load_err(dir: &Path, expectation: &str) -> String {
    match model::load(dir) {
        Ok(_) => panic!("{expectation}: load unexpectedly succeeded"),
        Err(e) => e,
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
