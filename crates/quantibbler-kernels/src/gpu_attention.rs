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

use crate::core::KernelError;
use crate::gpu::{GPU_INDEX_CEILING, GpuContext};
use cubecl::prelude::*;
use cubek_attention::forward::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions,
};
use cubek_attention::forward::launch::{BlueprintStrategy, Strategy, launch_ref};

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
}
