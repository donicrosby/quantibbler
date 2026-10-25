//! cubek-attention device face — change `qk-flash-attention`, task 2.2.
//!
//! Substrate: `cubek-attention` 0.3.0 on the pinned cubecl 0.11 line
//! (design.md "Substrate decision 2026-10-19"). This module is the *wrapper*,
//! not a kernel: it validates on the host, adapts our engine conventions to
//! the substrate's contract, launches through `launch_ref`, and reads the
//! result back. Correctness is proven against the NumPy reference by
//! `tests/gpu_attention_parity.rs`; nothing here is believed on its own.
//!
//! Two contract gaps, both adapted here and both counted against the perf
//! probe (task 4) rather than hidden:
//!
//! 1. **No GQA in the substrate.** `AttentionDims` carries a single
//!    `num_heads`, so key/value heads are materialized up to `n_heads` on the
//!    host (`repeat_interleave` groups — head `h` reads kv head
//!    `h // group`), matching the NumPy reference's convention exactly. A
//!    first-class GQA interface upstream would delete this copy.
//! 2. **No *configurable* softmax-scale field.** The substrate applies the
//!    standard `1/√head_dim` scale internally, so Q must be passed
//!    **unscaled** — the first version of this adapter pre-scaled Q as the
//!    design's source audit suggested ("no softmax-scale field") and the
//!    parity battery caught the duplicate factor immediately: the device
//!    matched the dense reference at scale `1/(2√d)` (error 3.5e-1) and at
//!    exactly `1/√d` on the host side (error **9.0e-8**). The measurement is
//!    the correction; see the change's design.md crystallization entry.
//!
//! Fail-closed boundary: construction failures are typed ([`GpuContext`]'s
//! `init_at`, [`crate::gpu::GPU_INIT_HINT`]) and every per-call rejection is a
//! hard `KernelError`; a wrong-shaped launch never silently produces a value.
//! All device index arithmetic is validated against
//! [`crate::gpu::GPU_INDEX_CEILING`] before any allocation.
//!
//! The module is a wrapper *plus* the change `qk-sage-attention` INT8 tile
//! path ([`GpuContext::attention_sage`], [`sage_int8_tile_kernel`]). That one
//! is ours rather than the substrate's on purpose: the substrate has no INT8
//! attention stage, and the task-2.1 probe established that no substrate op
//! supplies a hardware byte-dot on this target either (`Dp4aOp` lowers to
//! `expand_dp4a_polyfill` on both HIP backends), so the tile path is hand-rolled
//! epilogues on the landed `gpu.rs` forms. It is parity-tested against the
//! Python reference by `tests/gpu_attention_sage.rs`.

use crate::core::KernelError;
use crate::gpu::{GPU_INDEX_CEILING, GpuContext};
use cubecl::prelude::*;
use cubek_attention::forward::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions,
};
use cubek_attention::forward::launch::{BlueprintStrategy, Strategy, launch_ref};

/// The [SA] §4 INT8 tile path — change `qk-sage-attention`, task 2.3.
///
/// One thread per `(batch, head, query token)` row; that thread owns the row's
/// whole score/softmax/accumulator sequence, so no cross-thread reduction,
/// barrier or atomic is needed and every value is a per-thread local. The two
/// INT8 dots (`Q̂K̂ᵀ` and `P̂V̂`) accumulate in `i32` on the device — [SA]'s whole
/// point — and everything else (per-token Q/K scales, per-block γ, the static
/// P̃ scale, per-channel V, the online-softmax statistics and the f32
/// accumulator) is the reference scheme of
/// `src/quantibbler/kernels/attention_sage.py` reproduced in f32.
///
/// Scratch layout (all flat, `rows = B·n_heads·S`): `qhat` `[rows·d]` i8,
/// `scores` `[rows·block]` f32, `phat` `[rows·block]` i8, `vscale` `[rows·val]`
/// f32, `kmean` `[rows·d]` f32. Each is indexed by the thread's own row, so a
/// thread only ever touches its own slice — the per-row duplicates of
/// what is logically a per-`(b, kv)` tile are the price of needing no
/// synchronization at all.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn sage_int8_tile_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    out: &mut Tensor<f32>,
    qhat: &mut Tensor<i8>,
    scores: &mut Tensor<f32>,
    phat: &mut Tensor<i8>,
    vscale: &mut Tensor<f32>,
    kmean: &mut Tensor<f32>,
    n_batch: u32,
    n_heads: u32,
    n_kv: u32,
    seq: u32,
    d: u32,
    val: u32,
    block: u32,
    causal: u32,
    inv_sqrt_d: f32,
) {
    let row = ABSOLUTE_POS as u32;
    let rows = n_batch * n_heads * seq;
    if row < rows {
        let span = n_heads * seq;
        let b = row / span;
        let r2 = row % span;
        let h = r2 / seq;
        let i = r2 % seq;
        let group = n_heads / n_kv;
        let kvh = h / group;
        let d_i = i32::cast_from(d);
        let val_i = i32::cast_from(val);

        // Query block this row belongs to; every row of a block walks the same
        // key blocks, and the causal stopping point is the block's end.
        let mut i1 = ((i / block) + 1u32) * block;
        if i1 > seq {
            i1 = seq;
        }
        let mut j_stop = seq;
        if causal == 1u32 {
            j_stop = i1;
        }

        // --- ψ_Q: per-token symmetric INT8 with 1/√d folded in ([SA] §4.6).
        let qbase = row * d;
        let mut qpeak = 0.0f32;
        let mut c = 0i32;
        while c < d_i {
            let x = q[(qbase + u32::cast_from(c)) as usize] * inv_sqrt_d;
            let ax = if x < 0.0f32 { 0.0f32 - x } else { x };
            if ax > qpeak {
                qpeak = ax;
            }
            c += 1;
        }
        let mut qdelta = 1.0f32;
        if qpeak > 0.0f32 {
            qdelta = qpeak / 127.0f32;
        }
        c = 0;
        while c < d_i {
            let x = q[(qbase + u32::cast_from(c)) as usize] * inv_sqrt_d;
            // ψ's code: rint(x/δ) clamped to ±127. Rounding is half-to-even —
            // NumPy's `rint`, which is what the reference quantizes with — and
            // it is written out here rather than factored into a `#[cube]`
            // helper because on this stack a helper is not callable from a
            // launch body (task-2.1 probe, third trap: hot primitives inline).
            let qz = x / qdelta;
            let qfl = qz.floor();
            let qfr = qz - qfl;
            let qup = qfl + 1.0f32;
            let qtie = if (i32::cast_from(qfl) % 2i32) != 0i32 {
                qup
            } else {
                qfl
            };
            let qrv = if qfr > 0.5f32 {
                qup
            } else {
                if qfr == 0.5f32 { qtie } else { qfl }
            };
            let qraw = i32::cast_from(qrv);
            // Split clamp rather than `clamp()`: the vetted device op surface has
            // no integer `clamp` (FloatOps::clamp is float-only), and the
            // value-form if is what the macro expands cleanly.
            let qpos = if qraw > 127i32 { 127i32 } else { qraw };
            let qcode = if qpos < -127i32 { -127i32 } else { qpos };
            qhat[(qbase + u32::cast_from(c)) as usize] = i8::cast_from(qcode);
            c += 1;
        }

        // --- running softmax statistics / accumulator ([SA] Eq. 5, f32).
        // The masking sentinel is a finite `-3e38` rather than `-inf`: the
        // reference masks with `-inf`, and a finite sentinel reproduces the two
        // properties that matter (it never wins a max against a real score, and
        // `exp(sentinel - m_new)` underflows to exactly 0) while keeping every
        // intermediate finite, so no `inf - inf` can appear. Written as a
        // literal, not a `const`, because a bare const path is expanded as a
        // *plain* `f32` and then `m = ...` no longer resolves to the
        // `NativeExpand` assignment (measured: E0308 at the launch attribute).
        // `f32::cast_from(0) - 3.0e38f32`, not the bare `-3.0e38f32` literal:
        // the cube macro const-folds a *constant* initializer into a plain `f32`
        // local, and a plain local then fails `m = <NativeExpand>` (measured:
        // E0308 "expected f32, found NativeExpand<f32>" at this attribute; an
        // expression initializer expands to NativeExpand and assigns fine).
        let mut m = f32::cast_from(0) - 3.0e38f32;
        let mut ell = 0.0f32;
        let mut vv = 0i32;
        while vv < val_i {
            out[(row * val + u32::cast_from(vv)) as usize] = 0.0f32;
            vv += 1;
        }
        let kv_tok = (b * n_kv + kvh) * seq;

        let mut j0 = 0u32;
        while j0 < j_stop {
            let mut j1 = j0 + block;
            if j1 > seq {
                j1 = seq;
            }
            let kj = j1 - j0;

            // γ per K-block ([SA] §4.2 Eq. 6): the block's channel mean, which
            // the scores add back below so m/ℓ stay comparable across blocks.
            let mut c = 0i32;
            while c < d_i {
                let mut ssum = 0.0f32;
                let mut jj = j0;
                while jj < j1 {
                    ssum += k[((kv_tok + jj) * d + u32::cast_from(c)) as usize];
                    jj += 1;
                }
                kmean[(row * d + u32::cast_from(c)) as usize] = ssum / f32::cast_from(kj);
                c += 1;
            }

            // S = δ_Q δ_K (Q̂K̂ᵀ) + q̃·meanᵀ — the INT8 dot first, then the two
            // scales, then the shift added back (the dequantized-then-unsmoothed
            // score path the `gamma-shift-invariance` scenario is written on).
            let mut jj = j0;
            while jj < j1 {
                let ktok = (kv_tok + jj) * d;
                let mut kpeak = 0.0f32;
                let mut c = 0i32;
                while c < d_i {
                    let g = k[(ktok + u32::cast_from(c)) as usize]
                        - kmean[(row * d + u32::cast_from(c)) as usize];
                    let ag = if g < 0.0f32 { 0.0f32 - g } else { g };
                    if ag > kpeak {
                        kpeak = ag;
                    }
                    c += 1;
                }
                let mut kdelta = 1.0f32;
                if kpeak > 0.0f32 {
                    kdelta = kpeak / 127.0f32;
                }
                let mut acc_i = 0i32;
                let mut shift = 0.0f32;
                c = 0;
                while c < d_i {
                    let km = kmean[(row * d + u32::cast_from(c)) as usize];
                    let qc = qhat[(qbase + u32::cast_from(c)) as usize];
                    let g = k[(ktok + u32::cast_from(c)) as usize] - km;
                    // ψ_K's code on the smoothed key (same half-to-even rint).
                    let kz = g / kdelta;
                    let kfl = kz.floor();
                    let kfr = kz - kfl;
                    let kup = kfl + 1.0f32;
                    let ktie = if (i32::cast_from(kfl) % 2i32) != 0i32 {
                        kup
                    } else {
                        kfl
                    };
                    let krv = if kfr > 0.5f32 {
                        kup
                    } else {
                        if kfr == 0.5f32 { ktie } else { kfl }
                    };
                    let kraw = i32::cast_from(krv);
                    let kpos = if kraw > 127i32 { 127i32 } else { kraw };
                    let kcode = if kpos < -127i32 { -127i32 } else { kpos };
                    acc_i += i32::cast_from(qc) * kcode;
                    shift += (f32::cast_from(qc) * qdelta) * km;
                    c += 1;
                }
                let mut s = f32::cast_from(acc_i) * qdelta * kdelta + shift;
                if causal == 1u32 && jj > i {
                    s = -3.0e38f32;
                }
                scores[(row * block + (jj - j0)) as usize] = s;
                jj += 1;
            }

            // Row maximum over the block, then the online-softmax rescale.
            let mut rowmax = f32::cast_from(0) - 3.0e38f32;
            let mut kk = 0u32;
            while kk < kj {
                let sv = scores[(row * block + kk) as usize];
                if sv > rowmax {
                    rowmax = sv;
                }
                kk += 1;
            }
            let mut m_new = m;
            if rowmax > m_new {
                m_new = rowmax;
            }
            // A row with no admissible score keeps m = −inf, ℓ = 0 and α = 0
            // exactly (the reference's 0/0 contract), which in the sentinel
            // encoding is "the maximum is still the masked sentinel".
            let mut live = 1i32;
            if m_new < -3.0e38f32 * 0.5f32 {
                live = 0i32;
            }
            let mut alpha = 0.0f32;
            if live == 1i32 {
                alpha = (m - m_new).exp();
            }

            // ψ_V per channel within the tile ([SA] §4.3 reason 3).
            let mut vv = 0i32;
            while vv < val_i {
                let mut vpeak = 0.0f32;
                let mut jj = j0;
                while jj < j1 {
                    let x = v[((kv_tok + jj) * val + u32::cast_from(vv)) as usize];
                    let ax = if x < 0.0f32 { 0.0f32 - x } else { x };
                    if ax > vpeak {
                        vpeak = ax;
                    }
                    jj += 1;
                }
                let mut vd = 1.0f32;
                if vpeak > 0.0f32 {
                    vd = vpeak / 127.0f32;
                }
                vscale[(row * val + u32::cast_from(vv)) as usize] = vd;
                vv += 1;
            }

            // ψ_P at the static 1/127 scale, and ℓ from the FULL-precision P̃
            // ([SA] Eq. 5: only the P̂V̂ product is quantized).
            let mut esum = 0.0f32;
            let mut kk = 0u32;
            while kk < kj {
                let mut p = 0.0f32;
                if live == 1i32 {
                    p = (scores[(row * block + kk) as usize] - m_new).exp();
                }
                esum += p;
                // ψ_P at the static 1/127 scale: rint(p·127), clamped to [0, 127].
                let pz = p * 127.0f32;
                let pfl = pz.floor();
                let pfr = pz - pfl;
                let pup = pfl + 1.0f32;
                let ptie = if (i32::cast_from(pfl) % 2i32) != 0i32 {
                    pup
                } else {
                    pfl
                };
                let prv = if pfr > 0.5f32 {
                    pup
                } else {
                    if pfr == 0.5f32 { ptie } else { pfl }
                };
                let praw = i32::cast_from(prv);
                let ppos = if praw > 127i32 { 127i32 } else { praw };
                let pcode = if ppos < 0i32 { 0i32 } else { ppos };
                phat[(row * block + kk) as usize] = i8::cast_from(pcode);
                kk += 1;
            }
            ell = alpha * ell + esum;

            // O ← diag(α)O + ψ⁻¹_{δ_P δ_V}(P̂V̂): the second INT8 dot, i32
            // accumulate, dequantized only after the dot.
            let mut vv = 0i32;
            while vv < val_i {
                let vd = vscale[(row * val + u32::cast_from(vv)) as usize];
                let mut pv = 0i32;
                let mut kk = 0u32;
                while kk < kj {
                    let vx = v[((kv_tok + j0 + kk) * val + u32::cast_from(vv)) as usize];
                    // ψ_V's code at this channel's tile-local scale.
                    let vz = vx / vd;
                    let vfl = vz.floor();
                    let vfr = vz - vfl;
                    let vup = vfl + 1.0f32;
                    let vtie = if (i32::cast_from(vfl) % 2i32) != 0i32 {
                        vup
                    } else {
                        vfl
                    };
                    let vrv = if vfr > 0.5f32 {
                        vup
                    } else {
                        if vfr == 0.5f32 { vtie } else { vfl }
                    };
                    let vraw = i32::cast_from(vrv);
                    let vpos = if vraw > 127i32 { 127i32 } else { vraw };
                    let vcode = if vpos < -127i32 { -127i32 } else { vpos };
                    pv += i32::cast_from(phat[(row * block + kk) as usize]) * vcode;
                    kk += 1;
                }
                let mut prev = 0.0f32;
                if live == 1i32 {
                    prev = out[(row * val + u32::cast_from(vv)) as usize];
                }
                out[(row * val + u32::cast_from(vv)) as usize] =
                    alpha * prev + f32::cast_from(pv) * (1.0f32 / 127.0f32) * vd;
                vv += 1;
            }

            m = m_new;
            j0 += block;
        }

        // O = acc / ℓ (the reference's 0/0 for a fully-masked row is the same
        // NaN, not invented here).
        let mut vv = 0i32;
        while vv < val_i {
            let idx = (row * val + u32::cast_from(vv)) as usize;
            let cur = out[idx];
            out[idx] = cur / ell;
            vv += 1;
        }
    }
}

/// Attention problem in our engine's conventions: `(B, n_heads, S, d)` queries
/// against `(B, n_kv_heads, S, d)` keys/values, GQA groups of
/// `n_heads / n_kv_heads`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub batch: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
    pub val_dim: usize,
}

/// Packed row-major `[b, h, s, d]` dims and strides (2-D planes, no tiling).
fn dims(batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<usize> {
    vec![batch, heads, seq, dim]
}

fn strides(heads: usize, seq: usize, dim: usize) -> Vec<usize> {
    vec![heads * seq * dim, seq * dim, dim, 1]
}

/// Materialize `n_heads` key/value heads from `n_kv_heads` by repeating each
/// kv head `group` times consecutively (the reference's `repeat_interleave`).
fn repeat_heads(
    src: &[f32],
    shape: &AttentionShape,
    group: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; shape.batch * shape.n_heads * seq * dim];
    let row = seq * dim;
    for b in 0..shape.batch {
        for h in 0..shape.n_heads {
            let kv = h / group;
            let src_base = ((b * shape.n_kv_heads + kv) * seq) * dim;
            let dst_base = ((b * shape.n_heads + h) * seq) * dim;
            out[dst_base..dst_base + row].copy_from_slice(&src[src_base..src_base + row]);
        }
    }
    out
}

impl GpuContext {
    /// Tiled attention forward on the device — `O` in `(B, n_heads, seq_q,
    /// val_dim)`, packed row-major, f32.
    ///
    /// Host-side validation happens BEFORE any device allocation, so a
    /// shape/length mismatch is the same typed error the CPU faces raise and
    /// never a truncated or misinterpreted launch.
    pub fn attention(
        &self,
        shape: AttentionShape,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        causal: bool,
    ) -> Result<Vec<f32>, KernelError> {
        let AttentionShape {
            batch,
            n_heads,
            n_kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        } = shape;
        if batch == 0 || n_heads == 0 || n_kv_heads == 0 {
            return crate::core::err("attention requires non-empty batch and head dimensions");
        }
        if n_heads % n_kv_heads != 0 {
            return crate::core::err(format!(
                "n_kv_heads must divide n_heads: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
            ));
        }
        if seq_q == 0 || seq_kv == 0 || head_dim == 0 || val_dim == 0 {
            return crate::core::err("attention requires non-empty sequence and head dimensions");
        }
        let expected_q = batch * n_heads * seq_q * head_dim;
        let expected_k = batch * n_kv_heads * seq_kv * head_dim;
        let expected_v = batch * n_kv_heads * seq_kv * val_dim;
        if query.len() != expected_q {
            return crate::core::err(format!(
                "query must hold {expected_q} values for {shape:?}, got {}",
                query.len()
            ));
        }
        if key.len() != expected_k {
            return crate::core::err(format!(
                "key must hold {expected_k} values for {shape:?}, got {}",
                key.len()
            ));
        }
        if value.len() != expected_v {
            return crate::core::err(format!(
                "value must hold {expected_v} values for {shape:?}, got {}",
                value.len()
            ));
        }
        // Device index arithmetic is u32; the ceiling is a typed validation
        // error, never a silent wrap (mirrors packed_matvec).
        let largest = expected_q.max(expected_k).max(expected_v) as u64;
        if largest > GPU_INDEX_CEILING {
            return crate::core::err(format!(
                "attention inputs exceed the device u32 index ceiling ({GPU_INDEX_CEILING}); \
                 use the CPU reference face"
            ));
        }

        // --- contract adapter: GQA materialization (see the module docstring).
        // Q is passed UNSCALED: the substrate applies 1/√head_dim itself, and
        // pre-scaling it here would square the scale (measured, not assumed).
        let group = n_heads / n_kv_heads;
        let key_rep = repeat_heads(key, &shape, group, seq_kv, head_dim);
        let value_rep = repeat_heads(value, &shape, group, seq_kv, val_dim);

        let client = self.client();
        let q_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(query)),
            dims(batch, n_heads, seq_q, head_dim),
            strides(n_heads, seq_q, head_dim),
            f32::elem_type_native(),
        );
        let k_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&key_rep)),
            dims(batch, n_heads, seq_kv, head_dim),
            strides(n_heads, seq_kv, head_dim),
            f32::elem_type_native(),
        );
        let v_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&value_rep)),
            dims(batch, n_heads, seq_kv, val_dim),
            strides(n_heads, seq_kv, val_dim),
            f32::elem_type_native(),
        );
        let out_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            dims(batch, n_heads, seq_q, val_dim),
            f32::elem_type_native(),
        );

        let global_types = AttentionGlobalTypes::from_single_float_dtype(
            f32::elem_type_native(),
            AttentionGlobalTypes::mask_dtype(client),
        );
        let options = AttentionOptions {
            causal,
            accumulator_precision: AccumulatorPrecision::default(),
        };

        launch_ref(
            Strategy::Unit(BlueprintStrategy::Inferred(())),
            client,
            q_dev.binding(),
            k_dev.binding(),
            v_dev.binding(),
            None,
            out_dev.clone().binding(),
            &global_types,
            options,
        )
        .map_err(|e| KernelError(format!("cubek attention rejected the problem: {e:?}")))?;

        let bytes = client.read_one_unchecked_tensor(out_dev.handle.clone().copy_descriptor(
            out_dev.shape().clone(),
            out_dev.strides().clone(),
            std::mem::size_of::<f32>(),
        ));
        Ok(f32::from_bytes(&bytes).to_vec())
    }

    /// INT8 tiled attention forward on the device — change `qk-sage-attention`,
    /// task 2.3. ([SA] §4 scheme, matching
    /// `quantibbler.kernels.attention_sage` stage for stage; see
    /// [`sage_int8_tile_kernel`].)
    ///
    /// Contract mirrors [`GpuContext::attention`]: same `(B, n_heads, S, d)` /
    /// `(B, n_kv_heads, S, d)` layout, same GQA grouping, same `1/√d` softmax
    /// scale (folded into Q's quantizer, [SA] §4.6), same packed row-major
    /// `(B, n_heads, seq, val_dim)` f32 output, and the same host-side
    /// validation BEFORE any device allocation — so both faces reject the same
    /// inputs with the same typed errors. `block_size` is the tile width of the
    /// scheme's outer/inner loop (the reference's `block_size`).
    ///
    /// Not supported (deliberately, and not silently): an additive attention
    /// bias (`mask`). The reference takes one; the exact device face does not
    /// take one either, and a bias path that never ran is not shipped.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_sage(
        &self,
        shape: AttentionShape,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        causal: bool,
        block_size: usize,
    ) -> Result<Vec<f32>, KernelError> {
        let AttentionShape {
            batch,
            n_heads,
            n_kv_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        } = shape;
        if batch == 0 || n_heads == 0 || n_kv_heads == 0 {
            return crate::core::err("attention requires non-empty batch and head dimensions");
        }
        if n_heads % n_kv_heads != 0 {
            return crate::core::err(format!(
                "n_kv_heads must divide n_heads: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
            ));
        }
        if seq_q == 0 || seq_kv == 0 || head_dim == 0 || val_dim == 0 {
            return crate::core::err("attention requires non-empty sequence and head dimensions");
        }
        // The γ(K) shift is taken over the key block currently in flight, which
        // the scheme defines for self-attention over one cache. A different
        // seq_kv is a different problem, not a variant of this one.
        if seq_q != seq_kv {
            return crate::core::err(format!(
                "sage attention is self-attention over one cache: seq_q={seq_q}, seq_kv={seq_kv}"
            ));
        }
        if block_size == 0 {
            return crate::core::err(
                "sage attention requires a positive block_size ([SA] §4's tile width)",
            );
        }
        let expected_q = batch * n_heads * seq_q * head_dim;
        let expected_k = batch * n_kv_heads * seq_kv * head_dim;
        let expected_v = batch * n_kv_heads * seq_kv * val_dim;
        if query.len() != expected_q {
            return crate::core::err(format!(
                "query must hold {expected_q} values for {shape:?}, got {}",
                query.len()
            ));
        }
        if key.len() != expected_k {
            return crate::core::err(format!(
                "key must hold {expected_k} values for {shape:?}, got {}",
                key.len()
            ));
        }
        if value.len() != expected_v {
            return crate::core::err(format!(
                "value must hold {expected_v} values for {shape:?}, got {}",
                value.len()
            ));
        }
        // Device index arithmetic is u32: check the operands AND every scratch
        // extent the kernel derives from them (a launch must never wrap).
        let rows = (batch as u64) * (n_heads as u64) * (seq_q as u64);
        let mut largest = (expected_q as u64)
            .max(expected_k as u64)
            .max(expected_v as u64)
            .max(rows * head_dim as u64)
            .max(rows * val_dim as u64)
            .max(rows * block_size as u64);
        // The tile's own address arithmetic is `row * block + k` in u32, so the
        // widest per-row stride times the row count has to stay under the
        // ceiling too.
        largest = largest.max(rows * block_size as u64 + block_size as u64);
        if largest > GPU_INDEX_CEILING {
            return crate::core::err(format!(
                "sage attention inputs exceed the device u32 index ceiling ({GPU_INDEX_CEILING}); \
                 use the reference face"
            ));
        }

        let client = self.client();
        let q_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(query)),
            vec![expected_q],
            vec![1],
            f32::elem_type_native(),
        );
        let k_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(key)),
            vec![expected_k],
            vec![1],
            f32::elem_type_native(),
        );
        let v_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f32::as_bytes(value)),
            vec![expected_v],
            vec![1],
            f32::elem_type_native(),
        );
        let out_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * val_dim],
            f32::elem_type_native(),
        );
        // Per-row scratch, each slice written and read by its own thread only.
        let qhat_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * head_dim],
            i8::elem_type_native(),
        );
        let scores_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * block_size],
            f32::elem_type_native(),
        );
        let phat_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * block_size],
            i8::elem_type_native(),
        );
        let vscale_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * val_dim],
            f32::elem_type_native(),
        );
        let kmean_dev = cubecl::std::tensor::TensorHandle::empty(
            client,
            vec![(rows as usize) * head_dim],
            f32::elem_type_native(),
        );

        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
        let threads = 256u32;
        let cubes = (rows.div_ceil(threads as u64)) as u32;
        sage_int8_tile_kernel::launch(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads),
            q_dev.clone().into_arg(),
            k_dev.clone().into_arg(),
            v_dev.clone().into_arg(),
            out_dev.clone().binding().into_tensor_arg(),
            qhat_dev.clone().binding().into_tensor_arg(),
            scores_dev.clone().binding().into_tensor_arg(),
            phat_dev.clone().binding().into_tensor_arg(),
            vscale_dev.clone().binding().into_tensor_arg(),
            kmean_dev.clone().binding().into_tensor_arg(),
            batch as u32,
            n_heads as u32,
            n_kv_heads as u32,
            seq_q as u32,
            head_dim as u32,
            val_dim as u32,
            block_size as u32,
            u32::from(causal),
            inv_sqrt_d,
        );

        let bytes = client.read_one_unchecked_tensor(out_dev.handle.clone().copy_descriptor(
            out_dev.shape().clone(),
            out_dev.strides().clone(),
            std::mem::size_of::<f32>(),
        ));
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}
