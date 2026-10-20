//! GPU attention parity — change `qk-flash-attention`, tasks 2.1/2.2.
//!
//! Scenarios (spec delta `specs/qk-kernels`): `gpu-parity-budget` (the device
//! face matches the NumPy reference within the reassociation budget, on the
//! fixture battery, ragged tails and causal/masked positions included) and
//! `fail-closed-construction` (no usable HIP device ⇒ a typed error carrying
//! the validated staging recipe, never a silent wrong-compute path).
//!
//! Gate pattern (openspec-project-runbook, same as `gpu_parity.rs`): the file
//! is compiled only under `--features gpu`, and the parity battery runs only
//! under `QUANTIBBLER_GPU_TESTS=1` against real hardware — unset, it prints a
//! skip notice (emulation is never a correctness proof for this face). The
//! fail-closed scenario deliberately runs in BOTH states: it must be provable
//! on a machine with no device at all, which is exactly where it will be
//! needed.
//!
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-kernels --features gpu --test gpu_attention_parity
//!
//! Budget: rel 1e-5 (`gpu-parity-budget`, E-3.1's attention analogue). The
//! device computes f32 against an f64 reference, so the comparison is
//! two-part for the same reason the reference scenario's is: an infinity-norm
//! relative bound at 1e-5, plus a per-element bound at 1e-5 restricted to
//! entries within `BAND` of their row's maximum — below that band, relative
//! error is dominated by cancellation rather than by the kernel.

#![cfg(feature = "gpu")]

use quantibbler_kernels::gpu::{GPU_INIT_HINT, GpuContext};
use quantibbler_kernels::gpu_attention::AttentionShape;

const RTOL: f64 = 1e-5;
/// Magnitude band (fraction of the row max) inside which the per-element
/// relative clause is meaningful. Measured in task 2.2; see design.md.
const BAND: f64 = 1e-1;

fn gpu_gate() -> bool {
    std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1")
}

fn load_battery() -> Vec<serde_json::Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("attention_tiled.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "attention fixture unreadable ({e}); regenerate with \
             `uv run python tools/gen_attention_fixtures.py`"
        )
    });
    serde_json::from_str(&text).expect("attention fixture is not valid JSON")
}

fn f32_flat(value: &serde_json::Value, field: &str) -> Vec<f32> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field} is not an array"))
        .iter()
        .map(|x| x.as_f64().expect("fixture element is not a number") as f32)
        .collect()
}

fn f64_flat(value: &serde_json::Value, field: &str) -> Vec<f64> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field} is not an array"))
        .iter()
        .map(|x| x.as_f64().expect("fixture element is not a number"))
        .collect()
}

fn usize_field(value: &serde_json::Value, field: &str) -> usize {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("fixture field {field} is missing")) as usize
}

/// Two-part parity gate; see the module docstring for why it has two parts.
fn assert_parity(actual: &[f32], expected: &[f64], row_len: usize, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: device returned {} values, reference has {}",
        actual.len(),
        expected.len()
    );
    let widened: Vec<f64> = actual.iter().map(|v| f64::from(*v)).collect();
    let norm = widened
        .iter()
        .chain(expected.iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(
        norm.is_finite() && norm > 0.0,
        "{label}: degenerate output norm"
    );
    let max_abs = widened
        .iter()
        .zip(expected.iter())
        .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        max_abs <= RTOL * norm,
        "{label}: infinity-norm relative error {:.3e} exceeds {RTOL:e} (norm {norm:.3e})",
        max_abs / norm
    );

    // Per-element clause on the well-conditioned band, row by row.
    let mut worst = 0.0f64;
    for row in 0..(expected.len() / row_len) {
        let lo = row * row_len;
        let hi = lo + row_len;
        let row_max = expected[lo..hi]
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        for i in lo..hi {
            if expected[i].abs() >= BAND * row_max && expected[i].abs() > 0.0 {
                worst = worst.max((widened[i] - expected[i]).abs() / expected[i].abs());
            }
        }
    }
    assert!(
        worst <= RTOL,
        "{label}: per-element relative error {worst:.3e} exceeds {RTOL:e} inside the {BAND:e} band"
    );
}

#[test]
fn gpu_parity_budget_rel_1e_5_over_fixture_battery() {
    if !gpu_gate() {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged ROCm SDK env) to run the \
             attention parity battery against real hardware"
        );
        return;
    }
    let battery = load_battery();
    assert!(!battery.is_empty(), "fixture battery is empty");
    let ctx = GpuContext::init().expect("GPU init failed; see error for the validated recipe");

    let mut ran = 0usize;
    for record in &battery {
        let shape = &record["shape"];
        let case = AttentionShape {
            batch: usize_field(shape, "batch"),
            n_heads: usize_field(shape, "n_heads"),
            n_kv_heads: usize_field(shape, "n_kv_heads"),
            seq_q: usize_field(shape, "seq"),
            seq_kv: usize_field(shape, "seq"),
            head_dim: usize_field(shape, "head_dim"),
            val_dim: usize_field(shape, "val_dim"),
        };
        let causal = record["causal"]
            .as_bool()
            .expect("fixture causal flag missing");
        let query = f32_flat(&record["inputs"], "query");
        let key = f32_flat(&record["inputs"], "key");
        let value = f32_flat(&record["inputs"], "value");
        let expected = f64_flat(&record["faces"]["numpy-reference"], "out");

        let out = ctx
            .attention(case, &query, &key, &value, causal)
            .unwrap_or_else(|e| panic!("attention launch failed on {case:?}: {e}"));
        // Rows of the output are `val_dim` wide, per (batch, head) plane.
        assert_parity(
            &out,
            &expected,
            case.val_dim,
            &format!(
                "batch={} heads={}/{}, seq={}, d={}, causal={causal}",
                case.batch, case.n_heads, case.n_kv_heads, case.seq_q, case.head_dim
            ),
        );
        ran += 1;
    }
    println!(
        "attention parity: {ran}/{} fixture cases ran on the device",
        battery.len()
    );
}

#[test]
fn fail_closed_construction_without_a_usable_device() {
    // Deliberately NOT env-gated: this is the path a machine with no device
    // takes, so it must hold in the unset state too.
    let impossible = usize::MAX / 2;
    match GpuContext::init_at(impossible) {
        Err(err) => {
            let text = err.to_string();
            assert!(
                text.contains("rocm-sdk"),
                "fail-closed error must embed the validated staging recipe; got: {text}"
            );
            assert!(
                text.len() >= GPU_INIT_HINT.len() / 2,
                "fail-closed error looks truncated: {text}"
            );
        }
        Ok(_) => panic!(
            "constructed a GPU context on device index {impossible} — fail-closed is violated"
        ),
    }
}
