//! Task 2.3 — the third parity face: cubek's own naive FA2 CPU reference.
//!
//! Two independent implementations of the same published algorithm, fed the
//! same bits and compared: that is the strongest available guard against a
//! shared misreading of the conventions (GQA head grouping, which dimension the
//! softmax scale divides, the causality rule) — every failure mode we hit while
//! landing the device face was a *convention* failure, and a convention is
//! exactly what one implementation cannot detect in itself.
//!
//! Three faces exist and none of them compares to itself:
//! 1. our NumPy reference (`src/quantibbler/kernels/attention_tiled.py`) — the
//!    verification fixture at the declared dtype the fixture records;
//! 2. the device face (`tests/gpu_attention_parity.rs`) — compared against (1);
//! 3. this face — cubek's `flash_attention_v2_reference_with_lse`, compared
//!    against (1). It is pure CPU: no device, no HIP, no kernel launch. Run it
//!    with `cargo test -p quantibbler-kernels --features cubek-cpu-reference
//!    --test cubek_cpu_reference_parity --offline -- --nocapture`.
//!
//! Opt-in by feature because `cubek-attention/cpu-reference` pulls
//! `cubek-test-utils` (and, in this workspace, cubecl's CPU/test support) — a
//! dependency a default or shipped build must not carry. The purity pair in
//! `crates/quantibbler-server/tests/cubek_purity.rs` asserts `cpu-reference`
//! stays out of the server graph whatever this face does.
//!
//! Conventions read out of the reference source before trusting it, all three
//! agree with ours: scale is `1/sqrt(head_dim)` (not `val_dim`) applied to the
//! scores; causality is bottom-right aligned (`j + seq_q > i + seq_kv`, i.e.
//! exactly `j > i` when the lengths are equal, which is every case here); K/V
//! are indexed by the *query* head count, so the recorded GQA layout has to be
//! expanded with the same consecutive-group mapping our reference documents
//! (`head h` reads `kv_head h / group`).
//!
//! One measured divergence is deliberately NOT asserted away — see
//! `fully_masked_row_conventions_differ` at the bottom of this file.

//! The face's cubecl edge is the alias `cubecl-ref` (same crate and version as
//! the `gpu` face's edge, minus `hip`), so the face compiles with no ROCm
//! toolchain — measured: cubecl-hip and hip-sys are absent from this feature's
//! graph while the `gpu` face keeps them. cubecl's `test-runtime = ["wgpu"]` and
//! `cubek-test-utils`' dependence on it means the wgpu backend rides along; that
//! is portable software, not an SDK, and it is unused by this reference.

#![cfg(feature = "cubek-cpu-reference")]

use cubecl_ref::ir::{ElemType, FloatKind};
use cubecl_ref::zspace::Shape;
use cubek_attention::eval::forward::cpu_reference::flash_attention_v2_reference_with_lse;
use cubek_attention::forward::definition::{
    AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
};
use cubek_test_utils::{HostData, HostDataVec, StridedLayout};
use serde_json::Value;

/// The witness budget: `RTOL` for both clauses, `BAND` as in the device face.
///
/// **Scope — LOCAL to this file.** The device face's budget
/// (`gpu_attention_parity.rs`) is a different problem class (f32 kernel vs f64
/// oracle, measured 9.0e-08 — 13× under) and nothing here touches it. One
/// defect class per budget.
///
/// **Shape — the two-part rule this change already accepted and measured for
/// the Python and device faces** (design.md, "Implementation crystallization
/// 2026-10-20", decision 2026-10-20): (1) infinity-norm relative
/// `max|a-e| <= RTOL * max(|a|,|e|)`, which covers the whole row; (2)
/// per-element relative `<= RTOL` restricted to entries within `BAND` of their
/// row's maximum. The unbanded per-element statistic that reddened this file
/// (fixture #0: cubek 1.146e-5 vs the 1e-5 budget) is still computed and
/// printed as a RECORDED DELTA, but is no longer the budget: on fixture #0 its
/// worst element is a 3.55e-3 entry (0.5% of its row max) where OUR OWN f32
/// reference sits 0.089 ulp(1.0f32) from the oracle and cubek's sits 0.341 ulp
/// — both sub-ulp at the accumulator scale, so that clause reads softmax
/// cancellation rather than either implementation. Same reason the other two
/// faces are already gated this way. `the_witness_budget_still_rejects_an_o1_convention_defect`
/// pins the reformulation's cost (sec-eng, 2026-10-20).
const RTOL: f64 = 1e-5;
const BAND: f64 = 1e-1;

const FIXTURE: &str = include_str!("fixtures/attention_tiled.json");

fn flat_f32(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("input array")
        .iter()
        .map(|x| x.as_f64().expect("input number") as f32)
        .collect()
}

fn flat_f64(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("reference array")
        .iter()
        .map(|x| x.as_f64().expect("reference number"))
        .collect()
}

fn host(data: Vec<f32>, shape: [usize; 4]) -> HostData {
    let shape = Shape::new(shape);
    let strides = StridedLayout::RowMajor.compute_strides(&shape);
    HostData {
        data: HostDataVec::F32(data),
        shape,
        strides,
    }
}

/// GQA expansion with the consecutive-group mapping our reference documents:
/// `group = n_heads / n_kv_heads` query heads share one kv head, `head h` reads
/// `kv_head h / group`. cubek's reference indexes K/V by the query head count,
/// so the fixture's `(B, n_kv, S, d)` layout is expanded here, batch-major.
///
/// Batch-aware because it has to be: this expander originally ignored `batch`,
/// which was invisible until the reformulated budget let the battery run past
/// case #0 — case #5 (`batch = 2`) then failed the length assert at
/// `160 != 80`. The batch-blind form was a latent defect masked by the very
/// assert that reddened the file.
fn expand_heads(
    data: &[f32],
    batch: usize,
    heads: usize,
    kv_heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    assert_eq!(
        data.len(),
        batch * kv_heads * seq * dim,
        "fixture input length"
    );
    assert_eq!(heads % kv_heads, 0, "n_kv_heads must divide n_heads");
    let group = heads / kv_heads;
    let mut out = Vec::with_capacity(batch * heads * seq * dim);
    for b in 0..batch {
        for kv in 0..kv_heads {
            let lo = (b * kv_heads + kv) * seq * dim;
            for _ in 0..group {
                out.extend_from_slice(&data[lo..lo + seq * dim]);
            }
        }
    }
    out
}

fn usizes(value: &Value, key: &str) -> usize {
    value.get(key).and_then(Value::as_u64).expect("dim") as usize
}

fn problem(case: &Value) -> AttentionProblem {
    let shape = &case["shape"];
    let seq = usizes(shape, "seq");
    AttentionProblem {
        dims: AttentionDims {
            batch: usizes(shape, "batch"),
            num_heads: usizes(shape, "n_heads"),
            seq_q: seq,
            seq_kv: seq,
            head_dim: usizes(shape, "head_dim"),
            val_dim: usizes(shape, "val_dim"),
        },
        masked: false,
        global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F32),
        ),
        options: AttentionOptions {
            causal: case["causal"].as_bool().expect("causal"),
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    }
}

/// The three statistics the witness is read by, from one pass over the output:
/// the asserted pair (`inf_norm_rel`, `banded_rel`) and the RECORDED delta
/// (`unbanded_rel`) that reddened this file before the reformulation. All
/// relative errors are taken against the reference (`|a-e| / |e|`), the same
/// form the device face's per-element clause uses.
struct Witness {
    max_abs: f64,
    inf_norm_rel: f64,
    banded_rel: f64,
    unbanded_rel: f64,
}

/// Two-part gate plus the recorded delta. `row_len` is the output row width
/// (`val_dim`), so the band is computed per output row exactly as the device
/// face does it.
fn witness(actual: &[f32], expected: &[f64], row_len: usize) -> Witness {
    assert_eq!(actual.len(), expected.len(), "output length vs reference");
    assert!(
        row_len > 0 && expected.len() % row_len == 0,
        "row width {row_len} must divide the output length {}",
        expected.len()
    );
    let widened: Vec<f64> = actual.iter().map(|v| f64::from(*v)).collect();
    let norm = widened
        .iter()
        .chain(expected.iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(norm.is_finite() && norm > 0.0, "degenerate output norm");
    let mut max_abs = 0.0f64;
    let mut unbanded = 0.0f64;
    let mut banded = 0.0f64;
    for row in 0..(expected.len() / row_len) {
        let (lo, hi) = (row * row_len, (row + 1) * row_len);
        let row_max = expected[lo..hi]
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        for i in lo..hi {
            let (a, e) = (widened[i], expected[i]);
            let abs = (a - e).abs();
            max_abs = max_abs.max(abs);
            if e != 0.0 {
                let rel = abs / e.abs();
                unbanded = unbanded.max(rel);
                if e.abs() >= BAND * row_max {
                    banded = banded.max(rel);
                }
            }
        }
    }
    Witness {
        max_abs,
        inf_norm_rel: max_abs / norm,
        banded_rel: banded,
        unbanded_rel: unbanded,
    }
}

/// Clause-specific failure messages: a reviewer must see WHICH clause the
/// reformulation asserts, and the measured margin, not just pass/fail.
fn assert_witness_parity(w: &Witness, label: &str) {
    assert!(
        w.inf_norm_rel <= RTOL,
        "{label}: infinity-norm relative error {:.6e} exceeds {RTOL:.0e} (max_abs {:.6e})",
        w.inf_norm_rel,
        w.max_abs
    );
    assert!(
        w.banded_rel <= RTOL,
        "{label}: per-element relative error {:.6e} exceeds {RTOL:.0e} for entries within \
         {BAND:.0e} of their row maximum (recorded unbanded delta {:.6e})",
        w.banded_rel,
        w.unbanded_rel
    );
}

/// Per-element dump for control experiments: `QUANTIBBLER_CUBEK_DUMP=<path>`
/// writes, per case, the reference's own output beside the recorded NumPy
/// oracle so a diff can be read element by element (the row structure of the
/// worst error is what localizes it to the accumulator or to `exp`).
fn dump_case(path: &str, case: &Value, actual: &[f32], expected: &[f64]) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open dump path");
    let actual: Vec<f64> = actual.iter().map(|v| f64::from(*v)).collect();
    writeln!(
        file,
        "{}",
        serde_json::json!({
            "shape": case["shape"],
            "causal": case["causal"],
            "actual": actual,
            "expected": expected,
        })
    )
    .expect("write dump line");
}

#[test]
fn cpu_reference_agrees_with_the_numpy_reference_on_the_fixture_battery() {
    let cases: Vec<Value> = serde_json::from_str(FIXTURE).expect("fixture parse");
    assert_eq!(cases.len(), 9, "fixture battery size");

    let mut worst_rel = 0.0f64;
    let mut worst_case = String::new();
    let mut worst_unbanded = 0.0f64;
    let mut worst_unbanded_case = String::new();
    for (index, case) in cases.iter().enumerate() {
        let shape = &case["shape"];
        let (heads, kv_heads) = (usizes(shape, "n_heads"), usizes(shape, "n_kv_heads"));
        let seq = usizes(shape, "seq");
        let (head_dim, val_dim) = (usizes(shape, "head_dim"), usizes(shape, "val_dim"));
        let batch = usizes(shape, "batch");

        let q = flat_f32(&case["inputs"]["query"]);
        let k = expand_heads(
            &flat_f32(&case["inputs"]["key"]),
            batch,
            heads,
            kv_heads,
            seq,
            head_dim,
        );
        let v = expand_heads(
            &flat_f32(&case["inputs"]["value"]),
            batch,
            heads,
            kv_heads,
            seq,
            val_dim,
        );

        let (q, k, v) = (
            host(q, [batch, heads, seq, head_dim]),
            host(k, [batch, heads, seq, head_dim]),
            host(v, [batch, heads, seq, val_dim]),
        );

        let problem = problem(case);
        let (out, lse) = flash_attention_v2_reference_with_lse(&q, &k, &v, None, &problem, None);

        let actual = match out.data {
            HostDataVec::F32(ref values) => values.clone(),
            _ => panic!("unexpected reference output dtype"),
        };
        let expected = flat_f64(&case["faces"]["numpy-reference"]["out"]);
        if let Ok(path) = std::env::var("QUANTIBBLER_CUBEK_DUMP") {
            dump_case(&path, case, &actual, &expected);
        }
        let w = witness(&actual, &expected, val_dim);

        let lse_values = match lse.data {
            HostDataVec::F32(ref values) => values.clone(),
            _ => panic!("unexpected lse dtype"),
        };
        let lse_worst = lse_values
            .iter()
            .fold(0.0f64, |acc, x| acc.max((*x as f64).abs()));
        let label = format!(
            "#{index} b={batch} h={heads}/{kv_heads} seq={seq} d={head_dim} val={val_dim} causal={}",
            case["causal"].as_bool().unwrap(),
        );

        println!(
            "{label}: inf_norm_rel={:.3e} banded_rel={:.3e} (budget {RTOL:.0e}) | max_abs={:.3e} \
             | recorded delta unbanded_rel={:.3e} | max|lse|={lse_worst:.6} finite={}",
            w.inf_norm_rel,
            w.banded_rel,
            w.max_abs,
            w.unbanded_rel,
            lse_values.iter().all(|x| x.is_finite()),
        );
        assert!(
            lse_values.iter().all(|x| x.is_finite()),
            "{label}: lse carries a non-finite entry with no fully-masked row in this case"
        );
        assert_witness_parity(&w, &label);
        if w.banded_rel > worst_rel {
            worst_rel = w.banded_rel;
            worst_case = label.clone();
        }
        if w.unbanded_rel > worst_unbanded {
            worst_unbanded = w.unbanded_rel;
            worst_unbanded_case = label;
        }
    }
    println!(
        "worst case: {worst_case} at banded_rel {worst_rel:.3e} | recorded delta worst: \
         {worst_unbanded_case} at unbanded_rel {worst_unbanded:.3e}"
    );
}

/// The divergence the third face exists to find, pinned rather than tolerated.
///
/// Reading cubek's reference source, its output denominator is clamped
/// (`let denom = if l > eps { l } else { eps }`) while its `lse` is `-inf` for a
/// row with no finite scores. Our reference — the verification fixture at the declared dtype —
/// computes the dense `0/0` and yields **NaN** at those positions, and our
/// device/spec contract asserts that NaN pattern position-for-position (task
/// 1.4). So the two implementations agree on `lse = -inf` and *disagree by
/// design* on the row's output entries: 0.0 there, NaN here.
///
/// This test asserts the divergence is exactly that shape and nothing else, so
/// that a future cubek release changing either side is a red test rather than a
/// silent convention drift. It is the reason the battery comparison above
/// excludes fully-masked rows (the fixture battery carries none).
#[test]
fn fully_masked_row_conventions_differ() {
    let (batch, heads, seq, dim) = (1usize, 1usize, 2usize, 2usize);
    // Row 0 attends nothing: every score is -inf. Row 1 attends normally.
    let mask_bool = vec![true, true, false, false];
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch,
            num_heads: heads,
            seq_q: seq,
            seq_kv: seq,
            head_dim: dim,
            val_dim: dim,
        },
        masked: true,
        global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F32),
        ),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };
    let q = host(vec![1.0, 0.0, 1.0, 0.0], [batch, heads, seq, dim]);
    let k = host(vec![1.0, 0.0, 1.0, 0.0], [batch, heads, seq, dim]);
    let v = host(vec![1.0, 0.0, 3.0, 4.0], [batch, heads, seq, dim]);
    let mask_shape = Shape::new([batch, heads, seq, seq]);
    // Strides are computed from the shape before the shape is moved into HostData.
    let mask_strides = StridedLayout::RowMajor.compute_strides(&mask_shape);
    let mask = HostData {
        data: HostDataVec::Bool(mask_bool),
        shape: mask_shape,
        strides: mask_strides,
    };

    let (out, lse) = flash_attention_v2_reference_with_lse(&q, &k, &v, Some(&mask), &problem, None);
    let out = match out.data {
        HostDataVec::F32(values) => values,
        _ => panic!("unexpected dtype"),
    };
    let lse = match lse.data {
        HostDataVec::F32(values) => values,
        _ => panic!("unexpected dtype"),
    };

    println!(
        "cubek: fully-masked row out={:?} lse={:?}",
        &out[0..dim],
        &lse[0..1]
    );
    assert_eq!(lse[0], f32::NEG_INFINITY, "cubek agrees on lse = -inf");
    assert!(
        out[0..dim].iter().all(|x| x.is_finite() && *x == 0.0),
        "cubek's clamped denominator yields exactly 0.0 on a fully-masked row, got {:?}",
        &out[0..dim]
    );
    assert!(
        out[dim..].iter().all(|x| x.is_finite()),
        "the normally-attended row must stay finite"
    );

    // Our side of the divergence, evaluated here rather than asserted from prose:
    // every score in the row is masked, so the row maximum is -inf and the dense
    // softmax computes exp(-inf - -inf) = NaN, weights NaN, output NaN. That is
    // the contract task 1.4 pins position-for-position in
    // `tests/test_attention_tiled.py::test_ref_fully_masked_row_yields_the_dense_nan_pattern`
    // (and which `attention_tiled.py` now produces from a genuinely empty
    // statistic, m = -inf and l = 0, rather than by NaN propagation).
    let scores = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    let peak = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights = scores.map(|s| (s - peak).exp());
    let dense_out = weights[0] * 1.0 / (weights[0] + weights[1]);
    println!("dense formula on the same row: weights={weights:?} out={dense_out}");
    assert!(
        dense_out.is_nan(),
        "the dense 0/0 must yield NaN on a fully-masked row"
    );

    println!(
        "divergence pinned: cubek 0.0 vs our dense 0/0 NaN on fully-masked rows; \
         both report lse = -inf"
    );
}

/// Proof of power (sec-eng, 2026-10-20 — mandatory component of the
/// reformulated budget): a widened shape must not swallow the defect class the
/// device battery catches. One O(1) convention defect — the GQA group mapping
/// replaced by `head h reads kv head h % kv_heads` instead of `h / group` — is
/// injected into the recorded K/V layout and pushed through the REAL reference
/// on a REAL fixture whose mapping is load-bearing (`n_heads = 8`,
/// `n_kv_heads = 2`, so half the heads get the wrong K/V). The gate must reject
/// it by orders of magnitude. The same call on the recorded layout runs first,
/// so this row cannot pass by measuring a broken baseline.
#[test]
fn the_witness_budget_still_rejects_an_o1_convention_defect() {
    let cases: Vec<Value> = serde_json::from_str(FIXTURE).expect("fixture parse");
    // Fixture #2 is the first case with more than one kv head (`h = 8/2`), so
    // the group mapping is load-bearing there and a modulo mapping really does
    // hand half the heads the wrong K/V.
    let case = &cases[2];
    let shape = &case["shape"];
    let (heads, kv_heads) = (usizes(shape, "n_heads"), usizes(shape, "n_kv_heads"));
    let seq = usizes(shape, "seq");
    let (head_dim, val_dim) = (usizes(shape, "head_dim"), usizes(shape, "val_dim"));
    let batch = usizes(shape, "batch");
    assert_eq!(batch, 1, "power row assumes a single batch plane");
    assert_eq!(heads % kv_heads, 0, "n_kv_heads must divide n_heads");
    assert!(
        kv_heads > 1,
        "a GQA-mapping defect is invisible with one kv head"
    );
    let group = heads / kv_heads;

    let q = flat_f32(&case["inputs"]["query"]);
    let k = flat_f32(&case["inputs"]["key"]);
    let v = flat_f32(&case["inputs"]["value"]);
    let expected = flat_f64(&case["faces"]["numpy-reference"]["out"]);
    let problem = problem(case);

    let run = |k: &[f32], v: &[f32]| -> Vec<f32> {
        let (q, k, v) = (
            host(q.clone(), [batch, heads, seq, head_dim]),
            host(k.to_vec(), [batch, heads, seq, head_dim]),
            host(v.to_vec(), [batch, heads, seq, val_dim]),
        );
        let (out, _) = flash_attention_v2_reference_with_lse(&q, &k, &v, None, &problem, None);
        match out.data {
            HostDataVec::F32(values) => values,
            _ => panic!("unexpected reference output dtype"),
        }
    };

    let healthy = witness(
        &run(
            &expand_heads(&k, batch, heads, kv_heads, seq, head_dim),
            &expand_heads(&v, batch, heads, kv_heads, seq, val_dim),
        ),
        &expected,
        val_dim,
    );
    println!(
        "power row healthy leg: inf_norm_rel={:.3e} banded_rel={:.3e}",
        healthy.inf_norm_rel, healthy.banded_rel
    );
    assert_witness_parity(&healthy, "power row healthy leg");

    // The injected defect: consecutive-group mapping replaced by modulo, in the
    // recorded (batch, n_kv_heads, seq, dim) layout the expander consumes.
    let mut k_bad = Vec::with_capacity(heads * seq * head_dim);
    let mut v_bad = Vec::with_capacity(heads * seq * val_dim);
    for head in 0..heads {
        let kv = head % kv_heads;
        k_bad.extend_from_slice(&k[kv * seq * head_dim..(kv + 1) * seq * head_dim]);
        v_bad.extend_from_slice(&v[kv * seq * val_dim..(kv + 1) * seq * val_dim]);
    }
    let defective = witness(&run(&k_bad, &v_bad), &expected, val_dim);
    println!(
        "power row injected GQA-mapping defect (head h reads kv h % {kv_heads}, not h / {group}): \
         inf_norm_rel={:.3e} banded_rel={:.3e} (budget {RTOL:.0e})",
        defective.inf_norm_rel, defective.banded_rel
    );
    assert!(
        defective.inf_norm_rel > RTOL && defective.banded_rel > RTOL,
        "the injected convention defect must miss BOTH clauses: inf-norm {:.3e}, banded {:.3e} \
         vs {RTOL:.0e}",
        defective.inf_norm_rel,
        defective.banded_rel
    );
    let margin = defective.inf_norm_rel.max(defective.banded_rel) / RTOL;
    println!("power row margin: {margin:.3e}x the budget");
    assert!(
        margin >= 1e2,
        "an O(1) convention defect must miss the reformulated budget by orders of magnitude, \
         got {margin:.3e}x"
    );
}
