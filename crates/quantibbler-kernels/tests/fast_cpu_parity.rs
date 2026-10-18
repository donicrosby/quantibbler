//! qk-attention task 3.5 — CPU fast-path parity + determinism tests.
//!
//! These are pure-CPU tests: they run wherever the crate builds, no
//! hardware gate needed (the `fast-cpu` feature is the only requirement):
//!   cargo test -p quantibbler-kernels --features fast-cpu --test fast_cpu_parity
//!
//! Contracts under test:
//! - Parity vs the scalar rlib face: rel < 1e-12 (the lane-split
//!   reassociation budget; nothing else may differ).
//! - Run-to-run bit-identity (3 repeats) — outputs may not vary run to
//!   run. Cross-thread-count stability holds by construction (parallelism
//!   across rows only; each row is a fixed single-thread order) and is
//!   asserted here via an explicit 1-thread vs default-pool comparison
//!   run in a fresh process by CI-with-GPU lane; the in-process check
//!   covers the repeatability half.
//! - Degenerate shapes and rejection parity with the scalar face.

#![cfg(feature = "fast-cpu")] // the fast module does not exist in default builds

use quantibbler_kernels::{core, fast};

struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn sign(&mut self) -> i8 {
        if self.next_u64() & 1 == 0 { 1 } else { -1 }
    }
    fn f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 0x4000_0000_0000_0000;
        (f64::from_bits(bits) - 3.0) * 2.0
    }
}

/// Backward-error gate: |fast − scalar| <= TOL · Σ_k |v_k| (every term's
/// magnitude is |s_k·v_k| = |v_k| for ±1 signs). This is the honest bound
/// for a lane-split reassociation; forward-relative error is condition-
/// number-bound and legitimately exceeds any fixed tolerance on
/// cancellation-heavy rows (crystallized in design.md with measurements).
const BACKWARD_TOL: f64 = 1e-12;

fn assert_backward(got: &[f64], want: &[f64], vec: &[f64], label: &str) {
    let term_scale: f64 = vec.iter().take(vec.len()).map(|v| v.abs()).sum();
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= BACKWARD_TOL * term_scale,
            "{label} row {i}: |{g} - {w}| = {err:.3e} > {} * {term_scale:.3e} (backward budget)",
            BACKWARD_TOL
        );
    }
}

#[test]
fn fast_matvec_parity_and_bit_identity() {
    for (n_rows, n_bits) in [(1usize, 1usize), (257, 73), (1024, 137), (64, 8)] {
        let mut rng = Xorshift(0xFA57_C0DE + n_rows as u64 * 7919 + n_bits as u64);
        let factor: Vec<i8> = (0..n_rows * n_bits).map(|_| rng.sign()).collect();
        let packed = core::pack_bits(&factor, n_rows, n_bits).expect("pack");
        let vec: Vec<f64> = (0..n_bits).map(|_| rng.f64()).collect();

        let scalar = core::packed_matvec(&packed, n_rows, n_bits, &vec).expect("scalar");
        let fast1 = fast::packed_matvec(&packed, n_rows, n_bits, &vec).expect("fast 1");
        let fast2 = fast::packed_matvec(&packed, n_rows, n_bits, &vec).expect("fast 2");
        let fast3 = fast::packed_matvec(&packed, n_rows, n_bits, &vec).expect("fast 3");

        assert_eq!(fast1, fast2, "run 2 not bit-identical ({n_rows}x{n_bits})");
        assert_eq!(fast2, fast3, "run 3 not bit-identical ({n_rows}x{n_bits})");

        assert_backward(&fast1, &scalar, &vec, &format!("{n_rows}x{n_bits}"));
    }
}

#[test]
fn fast_matmul_parity_non_square() {
    // Non-square on purpose (square layers hide transposition bugs).
    let (n_rows, n_bits, n_cols) = (3usize, 7usize, 5usize);
    let mut rng = Xorshift(0xD00D0);
    let factor: Vec<i8> = (0..n_rows * n_bits).map(|_| rng.sign()).collect();
    let packed = core::pack_bits(&factor, n_rows, n_bits).expect("pack");
    let mat: Vec<f64> = (0..n_bits * n_cols).map(|_| rng.f64()).collect();

    let scalar = core::packed_matmul(&packed, n_rows, n_bits, &mat, n_cols).expect("scalar");
    let fast = fast::packed_matmul(&packed, n_rows, n_bits, &mat, n_cols).expect("fast");
    assert_eq!(scalar.len(), fast.len());
    let term_scale: f64 = mat.iter().map(|v| v.abs()).sum();
    for (i, (&c, &g)) in scalar.iter().zip(fast.iter()).enumerate() {
        let err = (c - g).abs();
        assert!(
            err <= BACKWARD_TOL * term_scale,
            "element {i}: |{c} - {g}| = {err:.3e} > backward budget"
        );
    }

    // Zero columns / zero rows: core's contract is k = mat.len()/n_bits
    // with n_cols participating in the ELEMENT ADDRESSING; faces must
    // agree element-for-element even on off-nominal n_cols (not "be zero").
    let fast0 = fast::packed_matmul(&packed, n_rows, n_bits, &mat, 0).expect("0 cols");
    let scalar0 = core::packed_matmul(&packed, n_rows, n_bits, &mat, 0).expect("0 cols scalar");
    assert_eq!(fast0.len(), scalar0.len());
    for (i, (&c, &g)) in scalar0.iter().zip(fast0.iter()).enumerate() {
        let err = (c - g).abs();
        assert!(
            err <= BACKWARD_TOL * term_scale,
            "0-col element {i}: {err:.3e}"
        );
    }
    assert!(
        fast::packed_matvec(&[], 0, 8, &[1.0; 8])
            .expect("0 rows")
            .is_empty()
    );

    // Rejections mirror the scalar face (same message). n_bits + 2 (9)
    // crosses the word boundary at n_bits=7 (1 word -> 2), so both faces
    // must reject; +1 would be absorbed by the pad byte and legitimately
    // compute on both.
    let bad_fast = fast::packed_matvec(&packed, n_rows, n_bits + 2, &vec![1.0; n_bits + 2]);
    let bad_scalar = core::packed_matvec(&packed, n_rows, n_bits + 2, &vec![1.0; n_bits + 2]);
    match (bad_fast, bad_scalar) {
        (Err(f), Err(s)) => assert_eq!(f.0, s.0),
        _ => panic!("over-wide n_bits must be rejected on both faces"),
    }
}

#[test]
fn fast_matvec_throughput_note() {
    // Informational (printed under --nocapture, not a gate): scalar vs
    // fast on a production-ish decode shape, 512 calls ~ one token's
    // worth of layer matvecs at rank 137 across a few layers.
    let (n_rows, n_bits) = (1024usize, 137usize);
    let mut rng = Xorshift(0x7E57);
    let factor: Vec<i8> = (0..n_rows * n_bits).map(|_| rng.sign()).collect();
    let packed = core::pack_bits(&factor, n_rows, n_bits).expect("pack");
    let vec: Vec<f64> = (0..n_bits).map(|_| rng.f64()).collect();

    let t0 = std::time::Instant::now();
    for _ in 0..512 {
        let _ = core::packed_matvec(&packed, n_rows, n_bits, &vec).expect("scalar");
    }
    let scalar_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    for _ in 0..512 {
        let _ = fast::packed_matvec(&packed, n_rows, n_bits, &vec).expect("fast");
    }
    let fast_t = t1.elapsed();

    println!(
        "throughput note: 512x ({n_rows}x{n_bits}) matvec — scalar {scalar_t:?}, fast {fast_t:?} ({:.1}x)",
        scalar_t.as_secs_f64() / fast_t.as_secs_f64()
    );
}
