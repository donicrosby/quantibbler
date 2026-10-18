//! Hardware-gated GPU parity tests (qk-attention task 3.3).
//!
//! Gate pattern (openspec-project-runbook): run only under
//! `QUANTIBBLER_GPU_TESTS=1` against REAL hardware (gfx1151 via HIP);
//! unset, every test skips with a printed notice. The device environment
//! (staged ROCm SDK) must be present in the invoking shell:
//!   QUANTIBBLER_GPU_TESTS=1 \
//!   LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib \
//!   HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//!   cargo test -p quantibbler-kernels --features gpu --test gpu_parity
//!
//! Budget: E-3.1 — rel 1e-12 vs the CPU rlib face, and bit-identity of
//! repeated GPU runs (determinism, E-2.1's analog on the device face).

#![cfg(feature = "gpu")] // the gpu module does not exist in default builds

use quantibbler_kernels::core;
use quantibbler_kernels::gpu::GpuContext;

/// Deterministic xorshift64* — no rand dependency, stable across runs.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// ±1 sign for the bitplanes.
    fn sign(&mut self) -> i8 {
        if self.next_u64() & 1 == 0 { 1 } else { -1 }
    }

    /// f64 in [-2, 2) with 53 random mantissa bits.
    fn f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 0x4000_0000_0000_0000;
        (f64::from_bits(bits) - 3.0) * 2.0
    }
}

fn gpu_gate() -> bool {
    std::env::var("QUANTIBBLER_GPU_TESTS").is_ok_and(|v| v == "1")
}

#[test]
fn gpu_parity_packed_matvec_rel_1e_12_and_bit_identity() {
    if !gpu_gate() {
        println!(
            "skip: set QUANTIBBLER_GPU_TESTS=1 (plus staged ROCm SDK env) to run GPU parity against real hardware"
        );
        return;
    }
    let ctx = GpuContext::init().expect("GPU init failed; see error for the validated recipe");

    // Odd shapes on purpose: 73 bits forces a pad byte path (9 words, top
    // 7 bits dead) and 257 rows exercises a non-power-of-two grid.
    let (n_rows, n_bits) = (257usize, 73usize);
    let mut rng = Xorshift(0x5EED_B00Fu64 ^ 0xA11CE);
    let factor: Vec<i8> = (0..n_rows * n_bits).map(|_| rng.sign()).collect();
    let packed = core::pack_bits(&factor, n_rows, n_bits).expect("pack");
    let vec: Vec<f64> = (0..n_bits).map(|_| rng.f64()).collect();

    let cpu = core::packed_matvec(&packed, n_rows, n_bits, &vec).expect("cpu");
    let gpu1 = ctx
        .packed_matvec(&packed, n_rows, n_bits, &vec)
        .expect("gpu run 1");
    let gpu2 = ctx
        .packed_matvec(&packed, n_rows, n_bits, &vec)
        .expect("gpu run 2");

    // Repeated-run bit identity (device-face determinism).
    assert_eq!(
        gpu1, gpu2,
        "GPU output not bit-identical across repeated runs"
    );

    // E-3.1 budget.
    let mut worst = 0.0f64;
    for (i, (&c, &g)) in cpu.iter().zip(gpu1.iter()).enumerate() {
        let denom = c.abs().max(g.abs()).max(1e-300);
        let rel = (c - g).abs() / denom;
        worst = worst.max(rel);
        assert!(
            rel < 1e-12,
            "row {i}: rel {rel} >= 1e-12 (cpu {c}, gpu {g})"
        );
    }
    println!("parity OK: {n_rows} rows x {n_bits} bits, worst rel err {worst:.3e}");
}

#[test]
fn gpu_parity_degenerate_shapes() {
    if !gpu_gate() {
        println!("skip: set QUANTIBBLER_GPU_TESTS=1 to run");
        return;
    }
    let ctx = GpuContext::init().expect("GPU init");

    // Single row, single bit.
    let packed = core::pack_bits(&[1], 1, 1).expect("pack");
    let cpu = core::packed_matvec(&packed, 1, 1, &[3.5]).expect("cpu");
    let gpu = ctx.packed_matvec(&packed, 1, 1, &[3.5]).expect("gpu");
    assert!((cpu[0] - gpu[0]).abs() <= 1e-12 * cpu[0].abs().max(1e-300));
    assert_eq!(gpu[0], 3.5, "single +1 bit must pass the value through");

    // Zero rows: no launch, zeroed output.
    let gpu0 = ctx.packed_matvec(&[], 0, 8, &[1.0; 8]).expect("gpu 0-rows");
    assert!(gpu0.is_empty());

    // Rejections mirror the CPU face exactly (same messages).
    let bad = ctx.packed_matvec(&packed, 1, 9, &[1.0; 9]);
    let bad_cpu = core::packed_matvec(&packed, 1, 9, &[1.0; 9]);
    match (bad, bad_cpu) {
        (Err(g), Err(c)) => {
            assert_eq!(g.0, c.0, "GPU face must reject with the CPU face's message")
        }
        _ => panic!("over-wide n_bits must be rejected on both faces"),
    }
}

#[test]
fn gpu_parity_row_scale_shapes() {
    if !gpu_gate() {
        println!("skip: set QUANTIBBLER_GPU_TESTS=1 to run");
        return;
    }
    let ctx = GpuContext::init().expect("GPU init");

    // Rank-shaped geometry (dout x r) at several magnitudes, including one
    // wider than a single 256-thread cube (grid-stride coverage check).
    for (n_rows, n_bits) in [(1usize, 8usize), (64, 137), (1024, 137), (500, 8)] {
        let mut rng = Xorshift(0xC0FFEE + n_rows as u64 * 1000 + n_bits as u64);
        let factor: Vec<i8> = (0..n_rows * n_bits).map(|_| rng.sign()).collect();
        let packed = core::pack_bits(&factor, n_rows, n_bits).expect("pack");
        let vec: Vec<f64> = (0..n_bits).map(|_| rng.f64()).collect();
        let cpu = core::packed_matvec(&packed, n_rows, n_bits, &vec).expect("cpu");
        let gpu = ctx
            .packed_matvec(&packed, n_rows, n_bits, &vec)
            .expect("gpu");
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let rel = (c - g).abs() / c.abs().max(g.abs()).max(1e-300);
            assert!(rel < 1e-12, "{n_rows}x{n_bits} row {i}: rel {rel}");
        }
    }
    println!("scale-shape parity OK");
}
