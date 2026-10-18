//! CPU fast path (qk-attention task 3.5, user-directed): SIMD inner
//! products + rayon row-parallelism behind the opt-in `fast-cpu` feature.
//!
//! Determinism contract (crystallized in the change's design.md):
//! - Parallelism is ACROSS ROWS ONLY (row blocks to amortize task
//!   overhead). Each row's sum is one thread with a FIXED operation order
//!   (f64x4 lane-sequential partials + fixed horizontal reduce + scalar
//!   tail), so outputs are bit-identical regardless of thread count and
//!   fully repeatable run to run.
//! - The sanctioned deviation from the scalar face is the lane-split
//!   reassociation of each row's accumulation. Its honest bound is the
//!   BACKWARD error: |fast − scalar| <= C·eps·Σ_k |s_k·v_k| (error
//!   relative to the sum's magnitude scale). Forward-relative error is
//!   the WRONG criterion for any reassociation — it is bounded only by
//!   the sum's condition number and blows up on cancellation-heavy rows
//!   (measured: 1.53e-12 forward at 257x73 where the backward error was
//!   ~1e-15 of the term scale). Run-to-run bit-identity is still asserted
//!   exactly, in the parity tests.
//! - Validation is delegated to [`crate::core`] (one validation, all
//!   faces); degenerate shapes return before touching the pool.

use crate::core::KernelError;

/// Per-byte sign LUT: BYTES_TO_SIGNS[b] is the 8-wide f64 sign vector of
/// byte b's bits, LSB-first. One L1-resident lookup per byte replaces
/// branchy per-bit decode (the decode dominates ±1 inner products).
#[rustfmt::skip]
static BYTES_TO_SIGNS: [[f64; 8]; 256] = {
    let mut table = [[0.0f64; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut k = 0usize;
        while k < 8 {
            table[b][k] = if (b >> k) & 1 == 1 { 1.0 } else { -1.0 };
            k += 1;
        }
        b += 1;
    }
    table
};

/// Rows per rayon task. Small enough to load-balance, large enough that
/// per-task scheduling overhead vanishes against ~rows*n_bits flops.
const ROWS_PER_TASK: usize = 64;

/// One row's inner product, f64x4 lanes: lane-sequential partial sums over
/// 4-element groups in element order, then a fixed-order horizontal
/// reduce, then the scalar tail in element order. `vec` must have length
/// n_bits (validated by the caller); `row` points at this row's
/// `row_words` bytes.
fn row_dot(row: &[u8], vec: &[f64], n_bits: usize) -> f64 {
    use wide::f64x4;
    let mut acc = f64x4::ZERO;
    let n_vec4 = n_bits / 4;
    let mut t = 0usize;
    while t < n_vec4 {
        let base = t * 4;
        // 4-aligned groups always sit fully inside one byte: byte index is
        // base/8 with base % 8 ∈ {0, 4}, so signs for [base, base+4) are
        // BYTES_TO_SIGNS[byte][lane..lane+4], lane = base & 7.
        let byte_idx = base >> 3;
        let lane = base & 7;
        debug_assert!(lane == 0 || lane == 4, "quad groups are 4-aligned");
        let s = &BYTES_TO_SIGNS[row[byte_idx] as usize];
        let signs = f64x4::new([s[lane], s[lane + 1], s[lane + 2], s[lane + 3]]);
        let v = f64x4::new([vec[base], vec[base + 1], vec[base + 2], vec[base + 3]]);
        acc += signs * v;
        t += 1;
    }
    // Fixed-order horizontal reduce...
    let arr = acc.to_array();
    let vec4_sum = (arr[0] + arr[2]) + (arr[1] + arr[3]);
    // ...then the tail, SCALAR in element order (same arithmetic as the
    // scalar face for elements beyond the last full 4-group).
    let mut tail = 0.0f64;
    for k in n_vec4 * 4..n_bits {
        let bit = (row[k >> 3] >> (k & 7)) & 1;
        tail += if bit == 1 { 1.0 } else { -1.0 } * vec[k];
    }
    vec4_sum + tail
}

/// `unpack(packed) @ vec` — SIMD+rayon face of
/// [`crate::core::packed_matvec`]. Same validation, same error messages.
pub fn packed_matvec(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    vec: &[f64],
) -> Result<Vec<f64>, KernelError> {
    let n_words = crate::core::validate_packed(packed, n_rows, n_bits)?;
    if vec.len() != n_bits {
        return crate::core::err(format!("vec must have length {n_bits}, got {}", vec.len()));
    }
    if n_rows == 0 || n_bits == 0 {
        return Ok(vec![0.0; n_rows]);
    }
    use rayon::prelude::*;
    let mut out = vec![0.0f64; n_rows];
    out.par_chunks_mut(ROWS_PER_TASK)
        .enumerate()
        .for_each(|(block, out_chunk)| {
            for (r, out_r) in out_chunk.iter_mut().enumerate() {
                let row_idx = block * ROWS_PER_TASK + r;
                let row = &packed[row_idx * n_words..(row_idx + 1) * n_words];
                *out_r = row_dot(row, vec, n_bits);
            }
        });
    Ok(out)
}

/// Column-batched SIMD face of [`crate::core::packed_matmul`] — parallel
/// across ROWS, columns sequential within a row (column outputs of one row
/// share nothing; splitting them would add no determinism risk but also
/// little throughput at serving shapes).
pub fn packed_matmul(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    mat: &[f64],
    n_cols: usize,
) -> Result<Vec<f64>, KernelError> {
    let n_words = crate::core::validate_packed(packed, n_rows, n_bits)?;
    if n_bits != 0 && mat.len() % n_bits != 0 {
        return crate::core::err(format!(
            "mat length {} is not a multiple of n_bits {n_bits}",
            mat.len()
        ));
    }
    let k = mat.len().checked_div(n_bits.max(1)).unwrap_or(mat.len());
    if n_rows == 0 || k == 0 {
        return Ok(vec![0.0; n_rows * k]);
    }
    use rayon::prelude::*;
    let mut out = vec![0.0f64; n_rows * k];
    out.par_chunks_mut(ROWS_PER_TASK * k)
        .enumerate()
        .for_each(|(block, out_chunk)| {
            for r in 0..out_chunk.len() / k {
                let row_idx = block * ROWS_PER_TASK + r;
                let row = &packed[row_idx * n_words..(row_idx + 1) * n_words];
                let acc = &mut out_chunk[r * k..(r + 1) * k];
                // Mirror core's EXACT element addressing (mat[t*n_cols+j],
                // output width k from mat.len()/n_bits) — n_cols is part of
                // the arithmetic, not just the shape, and the faces must
                // agree element-for-element on off-nominal n_cols too.
                for t in 0..n_bits {
                    let bit = (row[t >> 3] >> (t & 7)) & 1;
                    let s = if bit == 1 { 1.0 } else { -1.0 };
                    let base = t * n_cols;
                    for (j, a) in acc.iter_mut().enumerate() {
                        *a += s * mat[base + j];
                    }
                }
            }
        });
    Ok(out)
}
