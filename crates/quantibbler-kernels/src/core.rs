//! Pyo3-free core layer: the ONE kernel implementation behind both faces.
//!
//! Plain-Rust signatures only — no pyo3 types cross this boundary. The rlib
//! face is this module's public API; the PyO3 face (in `lib.rs`) is thin
//! boundary conversion + error mapping over these functions (design D9).
//!
//! Numerics contract (design D8): float64 everywhere; accumulation pinned
//! left-to-right in row-element order so packed_matvec/packed_matmul are
//! bit-identical to the NumPy reference; R0 for joint_itq comes from a
//! splitmix64 fill mirrored bit-exactly by the Python side.
//!
//! Dtype genericity (change qk-kernels-dtype-generic) applies to the
//! FP-remainder products alone: [`packed_matvec`] and [`packed_matmul`] are
//! generic over the dense-operand scalar `T: KernelScalar`, accumulating in
//! `T::Acc` in the same element order. The `f64` instantiation is the v0.1
//! one, term for term — the bitplane pack/unpack core stays integer, and the
//! solver (`joint_itq`) stays float64.

use crate::dtype::KernelScalar;
use nalgebra::SVD;

pub const DEFAULT_MAX_ITER: usize = 50;
pub const DEFAULT_TOL: f64 = 1e-12;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX2: u64 = 0x94D0_49BB_1331_11EB;
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

/// Error type for kernel validation failures (mapped to Python ValueError).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError(pub String);

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for KernelError {}

pub fn err<T>(msg: impl Into<String>) -> Result<T, KernelError> {
    Err(KernelError(msg.into()))
}

/// splitmix64 next output from a mutable state.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(MIX1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX2);
    z ^ (z >> 31)
}

/// Deterministic `rank x rank` f64 matrix in [-1, 1), row-major, matching
/// `_splitmix_matrix` in `src/quantibbler/kernels.py` bit for bit.
pub fn splitmix_matrix(seed: u64, rank: usize) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut out = vec![vec![0.0f64; rank]; rank];
    for row in out.iter_mut() {
        for cell in row.iter_mut() {
            let z = splitmix64(&mut state);
            let u01 = (z >> 11) as f64 / TWO_POW_53;
            *cell = u01 * 2.0 - 1.0;
        }
    }
    out
}

/// Modified Gram-Schmidt orthonormalization, column by column. Every dot
/// product and update accumulates left-to-right in f64 element order;
/// iterator `sum()`/folds are sequential, so the order is bit-identical to
/// the NumPy reference.
pub fn gram_schmidt(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, KernelError> {
    let rank = a.len();
    let mut r0 = a.to_vec();
    for j in 0..rank {
        let mut v: Vec<f64> = r0.iter().map(|row| row[j]).collect();
        for i in 0..j {
            let d: f64 = r0.iter().zip(&v).map(|(row, &vt)| row[i] * vt).sum();
            for (vt, row) in v.iter_mut().zip(&r0) {
                *vt -= d * row[i];
            }
        }
        let s: f64 = v.iter().map(|vt| vt * vt).sum();
        let nrm = s.sqrt();
        if nrm == 0.0 {
            return err("joint_itq produced a degenerate zero column building R0");
        }
        for (row, &vt) in r0.iter_mut().zip(&v) {
            row[j] = vt / nrm;
        }
    }
    Ok(r0)
}

/// `a @ b` with explicit left-to-right f64 accumulation.
fn matmul_lr(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|a_row| {
            (0..b.first().map_or(0, |row| row.len()))
                .map(|j| a_row.iter().zip(b).map(|(x, brow)| x * brow[j]).sum())
                .collect()
        })
        .collect()
}

/// `a^T @ b` with explicit left-to-right f64 accumulation.
fn matmul_at_b_lr(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.first().map_or(0, |row| row.len());
    let p = b.first().map_or(0, |row| row.len());
    let mut out = vec![vec![0.0f64; p]; m];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, acc) in out_row.iter_mut().enumerate() {
            // dot over the lockstep rows of a^T and b: sum_t a[t][i] * b[t][j]
            *acc = a
                .iter()
                .zip(b)
                .map(|(arow, brow)| arow[i] * brow[j])
                .sum::<f64>();
        }
    }
    out
}

/// Pack a ±1 factor (row-major `n_rows x n_cols`) into uint8 bitplanes:
/// byte `j` holds elements `8j..8j+7`, element `8j+k` at bit `k`
/// (LSB-first); +1 -> bit 1, -1 -> bit 0, pad bits zero.
pub fn pack_bits(factor: &[i8], n_rows: usize, n_cols: usize) -> Result<Vec<u8>, KernelError> {
    if n_rows * n_cols != factor.len() {
        return err(format!(
            "factor length {} does not match {}x{}",
            factor.len(),
            n_rows,
            n_cols
        ));
    }
    let n_words = n_cols.div_ceil(8);
    let mut out = vec![0u8; n_rows * n_words];
    for i in 0..n_rows {
        for j in 0..n_words {
            let mut byte = 0u8;
            for k in 0..8 {
                let idx = j * 8 + k;
                if idx >= n_cols {
                    break;
                }
                match factor[i * n_cols + idx] {
                    1 => byte |= 1u8 << k,
                    -1 => {}
                    other => {
                        return err(format!(
                            "pack_bits expects values in {{-1, +1}}; got {other} at row {i}, column {idx}"
                        ));
                    }
                }
            }
            out[i * n_words + j] = byte;
        }
    }
    Ok(out)
}

/// Restore a ±1 factor from packed bitplanes with explicit logical width
/// `n_bits`; pad bits beyond `n_bits` are ignored, never validated.
pub fn unpack_bits(packed: &[u8], n_rows: usize, n_bits: usize) -> Result<Vec<i8>, KernelError> {
    let n_words = n_bits.div_ceil(8);
    if n_rows * n_words != packed.len() {
        return err(format!(
            "packed length {} does not match {} rows x {} bitplanes",
            packed.len(),
            n_rows,
            n_words
        ));
    }
    let mut out = vec![0i8; n_rows * n_bits];
    // Degenerate guard: chunks_mut(0) panics, and zero logical bits means
    // every row is empty -- the zeroed (n_rows x 0) output IS the result.
    if n_bits == 0 {
        return Ok(out);
    }
    for (out_row, packed_row) in out.chunks_mut(n_bits).zip(packed.chunks_exact(n_words)) {
        for (k, cell) in out_row.iter_mut().enumerate() {
            let bit = (packed_row[k >> 3] >> (k & 7)) & 1;
            *cell = if bit == 1 { 1 } else { -1 };
        }
    }
    Ok(out)
}

/// int8 sign with the documented zero tie-break: `v >= 0.0` maps to `+1`
/// (including exact zeros, +0.0 and -0.0); `v < 0.0` maps to `-1`.
pub fn sign_estimate(x: &[f64]) -> Vec<i8> {
    x.iter()
        .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect()
}

/// Accumulate one packed row against a dense slice of dtype `T`,
/// left-to-right in row-element order, summing in `T::Acc` (for `f64` this is
/// term-for-term the v0.1 accumulation and therefore bit-identical).
fn accumulate_row<T: KernelScalar>(
    packed: &[u8],
    row_words: usize,
    row: usize,
    n_bits: usize,
    vec: &[T],
) -> T::Acc {
    let base = row * row_words;
    let mut acc = <T::Acc as KernelScalar>::ZERO;
    for k in 0..n_bits {
        let bit = (packed[base + (k >> 3)] >> (k & 7)) & 1;
        let s = if bit == 1 {
            <T::Acc as KernelScalar>::ONE
        } else {
            <T::Acc as KernelScalar>::NEG_ONE
        };
        acc = acc + s * vec[k].to_acc();
    }
    acc
}

pub fn validate_packed(packed: &[u8], n_rows: usize, n_bits: usize) -> Result<usize, KernelError> {
    if n_bits == 0 && !packed.is_empty() {
        // allowed: zero logical bits (empty packed tolerated if n_rows == 0)
        if n_rows != 0 {
            return err("n_bits=0 with non-empty packed rows");
        }
    }
    let n_words = n_bits.div_ceil(8);
    if n_rows * n_words != packed.len() {
        return err(format!(
            "packed length {} does not match {} rows x {} bitplanes",
            packed.len(),
            n_rows,
            n_words
        ));
    }
    Ok(n_words)
}

/// `unpack_bits(packed) @ vec` without materializing the binary matrix.
///
/// Generic over the dense operand's dtype `T`; the result carries the same
/// dtype, and the accumulation is the one in [`accumulate_row`]. `T = f64` is
/// the v0.1 instantiation.
pub fn packed_matvec<T: KernelScalar>(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    vec: &[T],
) -> Result<Vec<T>, KernelError> {
    let n_words = validate_packed(packed, n_rows, n_bits)?;
    if vec.len() != n_bits {
        return err(format!("vec must have length {n_bits}, got {}", vec.len()));
    }
    Ok((0..n_rows)
        .map(|i| T::from_acc(accumulate_row::<T>(packed, n_words, i, n_bits, vec)))
        .collect())
}

/// Multiply one packed row (row `i`'s `row_words` bytes) by one matrix
/// column `j`, accumulating left-to-right in element order: element `t`
/// contributes `sign(bit t) * mat[t * n_cols + j]`.
fn accumulate_packed_row<T: KernelScalar>(
    packed_row: &[u8],
    n_bits: usize,
    j: usize,
    mat: &[T],
    n_cols: usize,
) -> T::Acc {
    let mut acc = <T::Acc as KernelScalar>::ZERO;
    for t in 0..n_bits {
        let bit = (packed_row[t >> 3] >> (t & 7)) & 1;
        let s = if bit == 1 {
            <T::Acc as KernelScalar>::ONE
        } else {
            <T::Acc as KernelScalar>::NEG_ONE
        };
        acc = acc + s * mat[t * n_cols + j].to_acc();
    }
    acc
}

/// Column-batched `packed_matvec` (packed x dense-float only), generic over
/// the dense operand's dtype `T`. `T = f64` is the v0.1 instantiation and its
/// accumulation is unchanged.
pub fn packed_matmul<T: KernelScalar>(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    mat: &[T],
    n_cols: usize,
) -> Result<Vec<T>, KernelError> {
    let n_words = validate_packed(packed, n_rows, n_bits)?;
    if n_bits != 0 && mat.len() % n_bits != 0 {
        return err(format!(
            "mat length {} is not a multiple of n_bits {n_bits}",
            mat.len()
        ));
    }
    let k = mat.len().checked_div(n_bits.max(1)).unwrap_or(mat.len());
    let mut out = vec![<T as KernelScalar>::ZERO; n_rows * k];
    // Degenerate guard: chunks_mut(0) panics; zero columns means the
    // zeroed (n_rows x 0) output IS the result.
    if k == 0 {
        return Ok(out);
    }
    for (out_row, packed_row) in out.chunks_mut(k).zip(packed.chunks_exact(n_words)) {
        for (j, acc) in out_row.iter_mut().enumerate() {
            *acc = T::from_acc(accumulate_packed_row::<T>(
                packed_row, n_bits, j, mat, n_cols,
            ));
        }
    }
    Ok(out)
}

/// Joint-ITQ alternating Procrustes ([LB2] Sec 4.4, Algorithm 1 lines 6-11).
///
/// Z = [u_hat; v_hat]; R0 from splitmix64(seed) + modified Gram-Schmidt;
/// loop `B <- sign(Z R)`, `(Phi, , Psi^T) <- svd(B^T Z)`, `R <- Psi Phi^T`;
/// stop when `||R_new - R_old||_F <= tol` (with `tol > 0`; `tol == 0` runs a
/// fixed `max_iter` iterations, matching Algorithm 1's fixed-T form) or at
/// `max_iter`. Returns (u_t, v_t, n_iter, converged).
#[allow(clippy::too_many_arguments)]
pub fn joint_itq(
    u_hat: &[f64],
    v_hat: &[f64],
    dout: usize,
    din: usize,
    rank: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<(Vec<i8>, Vec<i8>, usize, bool), KernelError> {
    if rank == 0 || dout + din == 0 {
        return err("joint_itq expects non-empty factors");
    }
    if u_hat.len() != dout * rank || v_hat.len() != din * rank {
        return err(format!(
            "joint_itq shape mismatch: u_hat {}x{}, v_hat {}x{rank}",
            dout,
            u_hat.len() / dout.max(1),
            din,
        ));
    }
    // Z = [u_hat; v_hat], (dout+din) x rank, row-major
    let mut z: Vec<Vec<f64>> = Vec::with_capacity(dout + din);
    for i in 0..dout {
        z.push(u_hat[i * rank..(i + 1) * rank].to_vec());
    }
    for i in 0..din {
        z.push(v_hat[i * rank..(i + 1) * rank].to_vec());
    }
    // R0: splitmix fill + modified Gram-Schmidt (bit-matches the NumPy leg)
    let raw = splitmix_matrix(seed, rank);
    let mut rr = gram_schmidt(&raw)?;
    let mut n_iter = 0usize;
    let mut converged = false;
    for it in 1..=max_iter {
        // B <- sign(Z R)
        let zr = matmul_lr(&z, &rr);
        let b: Vec<Vec<f64>> = zr
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
                    .collect()
            })
            .collect();
        // B^T Z (r x r), then R <- Psi Phi^T from its SVD
        let btz = matmul_at_b_lr(&b, &z);
        let btz_m = nalgebra::DMatrix::from_fn(rank, rank, |i, j| btz[i][j]);
        let svd = SVD::new(btz_m, true, true);
        let (phi, psi_t) = match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return err("SVD failed to produce factors"),
        };
        // R_new[i][j] = sum_t psi_t[(t, i)] * phi[(j, t)]  (Psi Phi^T)
        let mut r_new = vec![vec![0.0f64; rank]; rank];
        for i in 0..rank {
            for j in 0..rank {
                let mut acc = 0.0f64;
                for t in 0..rank {
                    acc += psi_t[(t, i)] * phi[(j, t)];
                }
                r_new[i][j] = acc;
            }
        }
        let mut delta_sq = 0.0f64;
        for i in 0..rank {
            for j in 0..rank {
                let d = r_new[i][j] - rr[i][j];
                delta_sq += d * d;
            }
        }
        let delta = delta_sq.sqrt();
        rr = r_new;
        n_iter = it;
        if tol > 0.0 && delta <= tol {
            converged = true;
            break;
        }
    }
    let zr = matmul_lr(&z, &rr);
    let signs: Vec<Vec<i8>> = zr
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| if x >= 0.0 { 1i8 } else { -1i8 })
                .collect()
        })
        .collect();
    let mut u_t = vec![0i8; dout * rank];
    let mut v_t = vec![0i8; din * rank];
    for i in 0..dout {
        u_t[i * rank..(i + 1) * rank].copy_from_slice(&signs[i]);
    }
    for i in 0..din {
        v_t[i * rank..(i + 1) * rank].copy_from_slice(&signs[dout + i]);
    }
    Ok((u_t, v_t, n_iter, converged))
}
