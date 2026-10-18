//! CubeCL GPU kernels (opt-in `gpu` feature) — qk-attention tasks 3.1–3.3.
//!
//! ONE arithmetic definition, THREE faces now: the NumPy/Python reference,
//! the pure-Rust CPU rlib ([`crate::core`]), and this CubeCL device face.
//! The GPU kernel mirrors [`crate::core::accumulate_row`] structurally:
//! per-row thread, bits decoded LSB-first within each byte, and a STRICTLY
//! SEQUENTIAL f64 accumulation in element order. Pair accumulation order is
//! therefore element order (k = 0, 1, 2, ...), identical to the CPU rlib;
//! the only sanctioned deviation is potential fma contraction inside the
//! device compiler, which the E-3.1 parity budget (rel 1e-12) exists to
//! absorb. Reordering the loop (tree reduction, unroll-by-N with
//! reassociation) is NOT sanctioned by this budget and must not be
//! introduced without re-proving the budget against the CPU goldens.
//!
//! Backend generalization (N-4): the kernel source is backend-agnostic
//! CubeCL; HIP on gfx1151 is the validated launch target. Correctness is a
//! PER-BACKEND proof obligation — the parity harness runs against real
//! hardware only (QUANTIBBLER_GPU_TESTS=1), never emulation.
//!
//! Fallback semantics (task 3.1, fail-closed at the boundary): device
//! absence or init failure at construction is a TYPED error carrying the
//! validated staging recipe — callers (server dispatch, PyO3) decide
//! policy; the server treats init failure as "stay on the CPU rlib" (soft),
//! because both faces compute the same numbers. Per-call validation errors
//! (shape/length/ceiling mismatches) are hard errors: wrong-shaped GPU work
//! must never silently produce a value. Device-side inputs are validated on
//! the host BEFORE launch (validate + u32 index ceiling), so a launch error
//! indicates a runtime/compiler fault, not bad input.

use crate::core::KernelError;
use cubecl::prelude::*;

/// Typed device-init failure carrying the operational recipe (task 3.1).
/// The probe-validated environment: staged ROCm userspace on LD_LIBRARY_PATH
/// and, when the runtime needs it, the gfx override for Strix Halo.
pub const GPU_INIT_HINT: &str = "no usable HIP/ROCm device; validated recipe: \
stage the ROCm SDK (libamdhip64.so + libhsa-runtime64 + libamd_comgr) and set \
LD_LIBRARY_PATH=<sdk>/lib (e.g. /opt/data/tmp/rocm-sdk/lib), optionally \
HSA_OVERRIDE_GFX_VERSION=11.5.1; the device must expose /dev/kfd and \
/dev/dri/renderD*";

/// Maximum linear element index addressable by the device kernel (u32).
/// Rows x row_words must stay under this; larger shapes are a hard
/// validation error (fall back to the CPU rlib face, which is usize-native).
pub const GPU_INDEX_CEILING: u64 = u32::MAX as u64;

#[cube(launch)]
fn packed_matvec_kernel(
    packed: &Tensor<u8>,
    vec: &Tensor<f64>,
    out: &mut Tensor<f64>,
    n_rows: u32,
    n_bits: u32,
    row_words: u32,
) {
    let row = ABSOLUTE_POS as u32;
    if row < n_rows {
        let base = row * row_words;
        let mut acc = f64::cast_from(0);
        let mut k = 0i32;
        let n_bits_i = i32::cast_from(n_bits);
        while k < n_bits_i {
            let idx = (base + u32::cast_from(k >> 3)) as usize;
            let byte = u32::cast_from(packed[idx]);
            let bit = (byte >> u32::cast_from(k & 7)) & 1;
            // Branchless sign: bit in {0,1} -> s in {-1.0, +1.0}.
            let s = f64::cast_from(bit) * 2.0 - 1.0;
            acc += s * vec[k as usize];
            k += 1;
        }
        out[row as usize] = acc;
    }
}

/// A live CubeCL client on the validated HIP device (gfx1151 or compatible).
pub struct GpuContext {
    client: Client,
}

impl GpuContext {
    /// Probe for the HIP device and initialize the CubeCL client.
    /// Fail-closed: every failure path returns [`KernelError`] with
    /// [`GPU_INIT_HINT`] embedded.
    pub fn init() -> Result<Self, KernelError> {
        let device = cubecl::Device::rocm(0)
            .map_err(|e| KernelError(format!("{GPU_INIT_HINT}; device probe failed: {e:?}")))?;
        Ok(Self {
            client: device.client(),
        })
    }

    /// `unpack(packed) @ vec` on the device — same contract as
    /// [`crate::core::packed_matvec`] (validation, degenerate shapes, error
    /// cases), mirrored bit-decode and sequential f64 accumulation.
    ///
    /// Fixed launch geometry, autotune-free (task 3.3: correctness before
    /// performance): one thread per row, 256 threads per cube, ceil-div
    /// cube count.
    pub fn packed_matvec(
        &self,
        packed: &[u8],
        n_rows: usize,
        n_bits: usize,
        vec: &[f64],
    ) -> Result<Vec<f64>, KernelError> {
        // Host-side validation BEFORE any device allocation: the CPU face's
        // exact rules, so both faces reject the same inputs with the same
        // messages.
        let n_words = crate::core::validate_packed(packed, n_rows, n_bits)?;
        if vec.len() != n_bits {
            return crate::core::err(format!("vec must have length {n_bits}, got {}", vec.len()));
        }
        // Device index arithmetic is u32 (task 3.1 ceiling). Exceeding it is
        // a typed validation error, never a silent wrap.
        let lin = (n_rows as u64) * (n_words as u64);
        if lin > GPU_INDEX_CEILING || (n_rows as u64) > GPU_INDEX_CEILING {
            return crate::core::err(format!(
                "packed shape {n_rows} x {n_words} words exceeds the device u32 index ceiling ({GPU_INDEX_CEILING}); use the CPU rlib face"
            ));
        }

        if lin == 0 {
            // Degenerate guard mirrors core: (0 x anything) or n_bits == 0
            // with zero rows -> zeroed output, no launch.
            return Ok(vec![0.0; n_rows]);
        }

        let client = &self.client;
        // Explicit TensorHandles (the probe's working pattern):
        // create_from_slice returns a raw Handle; into_arg() lives on
        // TensorHandle/ TensorArg, not on Handle.
        let packed_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(packed),
            vec![n_rows * n_words],
            vec![1],
            u8::elem_type_native(),
        );
        let vec_dev = cubecl::std::tensor::TensorHandle::new(
            client.create_from_slice(f64::as_bytes(vec)),
            vec![n_bits],
            vec![1],
            f64::elem_type_native(),
        );
        let out_dev =
            cubecl::std::tensor::TensorHandle::empty(client, vec![n_rows], f64::elem_type_native());

        let cubes = n_rows.div_ceil(256) as u32;
        packed_matvec_kernel::launch(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(256),
            packed_dev.clone().into_arg(),
            vec_dev.clone().into_arg(),
            out_dev.clone().binding().into_tensor_arg(),
            n_rows as u32,
            n_bits as u32,
            n_words as u32,
        );

        let bytes = client.read_one_unchecked_tensor(out_dev.handle.clone().copy_descriptor(
            out_dev.shape().clone(),
            out_dev.strides().clone(),
            std::mem::size_of::<f64>(),
        ));
        Ok(f64::from_bytes(&bytes).to_vec())
    }
}

/// Convenience probe: does a usable HIP device exist right now?
pub fn gpu_available() -> bool {
    GpuContext::init().is_ok()
}
