//! The generation engine (design D5/D6): immutable shared weights behind
//! `Arc` (lock-free reads — weights are never written after load), engine
//! access bounded by a `tokio::sync::Semaphore` sized by `--max-concurrency`,
//! greedy decode with a plain per-request state cache over the
//! kernels-rlib packed primitives.

use std::sync::Arc;

use tokio::sync::Semaphore;

#[cfg(feature = "gpu")]
use quantibbler_kernels::gpu::GpuContext;
#[cfg(feature = "gpu")]
use quantibbler_kernels::gpu_attention::AttentionShape;
use quantibbler_qpk1::{Layer, NormSlot, inventory_block_floats};

/// Which face computes the packed matvecs (qk-attention 3.4). Selected ONCE
/// per process at engine construction — never mid-stream, so a served model's
/// outputs stay deterministic for its whole lifetime (E-2.1). Both faces
/// implement the same arithmetic (E-3.1 parity, hardware-gated), so face
/// selection is a performance/availability policy, not a semantics choice.
pub enum MatvecFace {
    /// Pure-CPU kernels rlib (default; the only face in default builds).
    Cpu,
    /// CubeCL HIP device face. Constructible only with the `gpu` feature.
    #[cfg(feature = "gpu")]
    Gpu(GpuContext),
}

/// The engine's matvec-face decision, carried on `ModelWeights` so every
/// consumer (forward, health reporting) reads ONE source of truth.
pub struct FaceDecision {
    pub face: MatvecFace,
    /// Human-readable selection record (startup log + /health).
    pub chosen: &'static str,
}

impl FaceDecision {
    /// The default-build selection: the CPU face (scalar rlib, or the
    /// SIMD+rayon fast face when `fast-cpu` is compiled in — a build-time
    /// choice, reported honestly in the label).
    pub fn cpu_selected() -> Self {
        let chosen: &'static str = if cfg!(feature = "fast-cpu") {
            "cpu:fast"
        } else {
            "cpu"
        };
        Self {
            face: MatvecFace::Cpu,
            chosen,
        }
    }

    /// GPU dispatch policy (task 3.4, soft-fail): env QUANTIBBLER_GPU=1
    /// requests the HIP face; a failed device init is NOT fatal — the CPU
    /// face computes the same numbers, so we log and fall back. After
    /// startup the decision is frozen: per-call GPU errors are hard errors.
    #[cfg(feature = "gpu")]
    pub fn select_from_env() -> Self {
        if std::env::var("QUANTIBBLER_GPU").is_ok_and(|v| v == "1") {
            match GpuContext::init() {
                Ok(ctx) => {
                    return Self {
                        face: MatvecFace::Gpu(ctx),
                        chosen: "gpu:hip:gfx",
                    };
                }
                Err(e) => {
                    eprintln!(
                        "QUANTIBBLER_GPU=1 but the HIP face is unavailable; staying on the CPU rlib face. ({e})"
                    );
                }
            }
        }
        Self::cpu_selected()
    }

    #[cfg(not(feature = "gpu"))]
    pub fn select_from_env() -> Self {
        Self::cpu_selected()
    }

    /// Face label for /health.
    pub fn describe(&self) -> &'static str {
        self.chosen
    }
}

/// Which face computes attention (qk-flash-attention 3.2). Selected ONCE per
/// process at engine construction — never mid-stream, so a served model's
/// outputs stay deterministic for its whole lifetime — using the same
/// frozen-decision pattern as [`FaceDecision`] for matvecs. Both faces compute
/// the same attention inside the parity budget (spec delta `qk-server`,
/// "Serving parity across attention faces"), so face selection is a
/// performance/availability policy, not a semantics choice.
pub enum AttentionFace {
    /// The engine's in-line scalar CPU routine (default; the only face in
    /// default builds).
    CpuScalar,
    /// CubeCL tiled device face, behind the `gpu` feature + the env opt-in.
    #[cfg(feature = "gpu")]
    GpuTiled(GpuContext),
    /// CubeCL INT8 tiled device face (qk-sage-attention 3.2): behind the `gpu`
    /// feature + the `QUANTIBBLER_ATTN=sage` opt-in AND the startup quality gate
    /// (`model::sage_face_from_env`), which reads the model directory's shipped
    /// gate record. It is the one face whose selection depends on a MEASURED
    /// record rather than on the build alone — and unlike the other two, it is
    /// lossy: it approximates the exact attention inside [SA] §4.3's gate.
    #[cfg(feature = "gpu")]
    GpuSage(GpuContext),
}

/// The INT8 face's tile width ([SA] §4's block loop; the reference's
/// `block_size`). The kernel is one thread per query row, so this is the
/// γ/online-softmax granularity — the number of key positions whose shared
/// mean is removed by one γ — not a thread-block shape. Only the `gpu` face
/// consumes it: without that feature there is no INT8 path to size.
#[cfg(feature = "gpu")]
pub const SAGE_BLOCK_SIZE: usize = 64;

/// The engine's attention-face decision, carried on `ModelWeights` so every
/// consumer (forward, health reporting) reads ONE source of truth.
pub struct AttentionFaceDecision {
    pub face: AttentionFace,
    /// Human-readable selection record (`/health`) — the same field the matvec
    /// decision carries.
    pub chosen: &'static str,
}

impl AttentionFaceDecision {
    /// The default-build selection: the in-line scalar CPU routine.
    pub fn cpu_scalar_selected() -> Self {
        Self {
            face: AttentionFace::CpuScalar,
            chosen: "cpu:scalar",
        }
    }

    /// GPU dispatch policy (task 3.2, soft-fail): env `QUANTIBBLER_ATTN=1`
    /// requests the tiled device face; a failed device init is NOT fatal — the
    /// scalar face computes the same attention inside the parity budget, so we
    /// log and fall back rather than abort startup (spec delta `qk-server`,
    /// "Soft fallback on init failure"). After startup the decision is
    /// frozen: per-call device errors are hard errors.
    #[cfg(feature = "gpu")]
    pub fn select_from_env() -> Self {
        if std::env::var("QUANTIBBLER_ATTN").is_ok_and(|v| v == "1") {
            match GpuContext::init_at(attention_device_index()) {
                Ok(ctx) => {
                    return Self {
                        face: AttentionFace::GpuTiled(ctx),
                        chosen: "gpu:tiled",
                    };
                }
                Err(e) => {
                    eprintln!(
                        "QUANTIBBLER_ATTN=1 but the tiled attention face is unavailable; \
                         staying on the scalar CPU face. ({e})"
                    );
                }
            }
        }
        Self::cpu_scalar_selected()
    }

    #[cfg(not(feature = "gpu"))]
    pub fn select_from_env() -> Self {
        Self::cpu_scalar_selected()
    }

    /// The `gpu:sage` decision (qk-sage-attention 3.2). The CALLER has already
    /// admitted the model's gate record (`model::sage_face_from_env`), so this
    /// only constructs the device milieu: a failed device init is NOT fatal —
    /// the exact scalar face computes the same attention inside the parity
    /// budget, so we log and fall back — while QUALITY is never the fallback
    /// trigger: an out-of-bounds record aborts startup long before this runs
    /// (spec delta `qk-server`, "Soft fallback on device failure only").
    #[cfg(feature = "gpu")]
    pub fn sage_selected() -> Self {
        match GpuContext::init_at(attention_device_index()) {
            Ok(ctx) => Self {
                face: AttentionFace::GpuSage(ctx),
                chosen: "gpu:sage",
            },
            Err(e) => {
                eprintln!(
                    "the gate admitted the gpu:sage face but the INT8 device face is unavailable; \
                     staying on the scalar CPU face. ({e})"
                );
                Self::cpu_scalar_selected()
            }
        }
    }

    /// Without the `gpu` feature there is no device face to reach: the opt-in is
    /// inert exactly as `QUANTIBBLER_ATTN=1` is (the gate has already run and
    /// admitted the model's record — see `model::sage_face_from_env`).
    #[cfg(not(feature = "gpu"))]
    pub fn sage_selected() -> Self {
        Self::cpu_scalar_selected()
    }

    /// Face label for /health.
    pub fn describe(&self) -> &'static str {
        self.chosen
    }
}

/// The device index the tiled face probes. Production is index 0; test builds
/// may override it so the soft-fallback branch is constructible on a machine
/// WITH a usable device: an out-of-range index is the same deterministic init
/// failure the kernels crate's `fail_closed_construction_without_a_usable_device`
/// test uses. `#[cfg(test)]`-gated on purpose — no shipped binary carries it.
#[cfg(feature = "gpu")]
fn attention_device_index() -> usize {
    #[cfg(test)]
    if let Some(index) = std::env::var("QUANTIBBLER_ATTN_TEST_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
    {
        return index;
    }
    0
}

/// Error surface of the engine (maps to the 500 envelope at the API edge).
#[derive(Debug)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
use std::sync::atomic::AtomicBool;
#[cfg(test)]
use std::sync::atomic::Ordering;

/// Bitplane layout conversion between the two pinned layouts.
///
/// * The qpk1 CONTAINER packs one continuous bit stream per matrix
///   (element `(i, j)` at stream bit `i*r + j`, LSB-first within bytes;
///   qk-representation SF-3.1).
/// * The kernels rlib packs PER-ROW bitplanes (each row starts a new byte;
///   row width `ceil(r/8)` bytes).
///
/// They coincide exactly when `r % 8 == 0`; otherwise the server repacks
/// once at load so request-time math never touches layout conversion.
pub fn container_to_rows(packed: &[u8], n_rows: usize, r: usize) -> Vec<u8> {
    if r % 8 == 0 {
        return packed.to_vec();
    }
    let row_words = r.div_ceil(8);
    let mut out = vec![0u8; n_rows * row_words];
    for i in 0..n_rows {
        for j in 0..r {
            let bit_idx = i * r + j;
            if (packed[bit_idx >> 3] >> (bit_idx & 7)) & 1 == 1 {
                out[i * row_words + (j >> 3)] |= 1 << (j & 7);
            }
        }
    }
    out
}

/// Build the per-row bitplanes of `Vb^T` (r rows x din bits) straight from
/// the container bitplane of `Vb` (din x r, continuous bits).
pub fn container_to_rows_transposed(packed: &[u8], din: usize, r: usize) -> Vec<u8> {
    let row_words = din.div_ceil(8);
    let mut out = vec![0u8; r * row_words];
    for j in 0..din {
        for k in 0..r {
            let bit_idx = j * r + k;
            if (packed[bit_idx >> 3] >> (bit_idx & 7)) & 1 == 1 {
                out[k * row_words + (j >> 3)] |= 1 << (j & 7);
            }
        }
    }
    out
}

fn row_words(r: usize) -> usize {
    r.div_ceil(8)
}

/// The CPU face's matvec: the scalar rlib by default, the SIMD+rayon fast
/// face when `fast-cpu` is compiled in (task 3.5). One shim so `apply`
/// carries no cfg noise; outputs of the two differ only within the
/// backward-error budget crystallized in the kernels crate.
fn cpu_matvec(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    vec: &[f64],
) -> Result<Vec<f64>, quantibbler_kernels::core::KernelError> {
    #[cfg(feature = "fast-cpu")]
    return quantibbler_kernels::fast::packed_matvec(packed, n_rows, n_bits, vec);
    #[cfg(not(feature = "fast-cpu"))]
    quantibbler_kernels::core::packed_matvec(packed, n_rows, n_bits, vec)
}

/// Per-layer serving form of one factorized linear layer.
#[derive(Debug)]
pub struct PackedLayer {
    /// Layer name from the manifest (diagnostics only).
    pub name: String,
    pub dout: usize,
    pub din: usize,
    pub r: usize,
    /// s1 as float32 (serving accumulation dtype; see the design note).
    pub s1_f32: Vec<f32>,
    /// s2 as float32.
    pub s2_f32: Vec<f32>,
    /// Bias as float32, exactly `dout` entries. A version-1 stream carries no
    /// bias vector and normalizes to `dout` zeros — the declared legacy rule,
    /// so `apply` needs no version branch.
    pub bias_f32: Vec<f32>,
    /// Ub in kernels per-row bitplane layout: `dout * ceil(r/8)` bytes.
    pub ub_rows: Vec<u8>,
    /// Vb in kernels per-row bitplane layout: `din * ceil(r/8)` bytes.
    /// Kept for diagnostics and future per-row consumers; the decode path
    /// itself uses the transposed `vb_t_rows`.
    #[allow(dead_code)]
    pub vb_rows: Vec<u8>,
    /// Vb TRANSPOSED (r rows x din bits), per-row bitplane layout:
    /// `r * ceil(din/8)` bytes. Needed because the kernels matvec is
    /// row-oriented and the factorized product contracts `Vb^T z` over the
    /// din axis; the transpose is built once at load.
    pub vb_t_rows: Vec<u8>,
}

impl PackedLayer {
    pub fn from_layer(name: String, layer: &Layer) -> Result<Self, EngineError> {
        let expected_ub = layer.dout * row_words(layer.r);
        let expected_vb = layer.din * row_words(layer.r);
        let expected_vb_t = layer.r * row_words(layer.din);
        let ub_rows = container_to_rows(&layer.ub_packed, layer.dout, layer.r);
        let vb_rows = container_to_rows(&layer.vb_packed, layer.din, layer.r);
        let vb_t_rows = container_to_rows_transposed(&layer.vb_packed, layer.din, layer.r);
        if ub_rows.len() != expected_ub
            || vb_rows.len() != expected_vb
            || vb_t_rows.len() != expected_vb_t
        {
            return Err(EngineError(format!(
                "layer {name}: bitplane length mismatch after repack"
            )));
        }
        if !layer.bias.is_empty() && layer.bias.len() != layer.dout {
            return Err(EngineError(format!(
                "layer {name}: bias has {} entries; expected dout={}",
                layer.bias.len(),
                layer.dout
            )));
        }
        let bias_f32 = if layer.bias.is_empty() {
            vec![0.0f32; layer.dout] // legacy v1: the zero map
        } else {
            layer.bias.iter().map(|&v| v as f32).collect()
        };
        Ok(Self {
            name,
            dout: layer.dout,
            din: layer.din,
            r: layer.r,
            s1_f32: layer.s1.iter().map(|&v| v as f32).collect(),
            s2_f32: layer.s2.iter().map(|&v| v as f32).collect(),
            bias_f32,
            ub_rows,
            vb_rows,
            vb_t_rows,
        })
    }

    /// `y = diag(s1) @ (Ub @ (Vb^T @ (diag(s2) @ x))) + bias` — the
    /// factorized linear transform `x -> W x + b` with
    /// `W = diag(s1) (Ub Vb^T) diag(s2)` (qk-representation Eq. 1) and the
    /// per-linear bias the container carries. A legacy version-1 stream
    /// contributes an all-zero bias, so the map is the one it always was.
    ///
    /// Bitplane products go through the selected matvec face (task 3.4):
    /// the kernels rlib by default, or the CubeCL HIP face under
    /// `gpu` + QUANTIBBLER_GPU=1 — both float64 accumulation in
    /// row-element order (E-3.1 hardware parity); scales and I/O are f32.
    fn apply(&self, face: &MatvecFace, x: &[f32]) -> Result<Vec<f32>, EngineError> {
        if x.len() != self.din {
            return Err(EngineError(format!(
                "layer {} expects input dim {}, got {}",
                self.name,
                self.din,
                x.len()
            )));
        }
        let mut z = vec![0.0f64; self.din];
        for (j, zj) in z.iter_mut().enumerate() {
            *zj = f64::from(self.s2_f32[j] * x[j]);
        }
        // t = Vb^T z  (length r; contraction over the din axis), then
        // u = Ub t  (length dout; contraction over the rank axis).
        let t = match face {
            MatvecFace::Cpu => cpu_matvec(&self.vb_t_rows, self.r, self.din, &z)
                .map_err(|e| EngineError(format!("layer {}: Vb^T matvec: {e}", self.name)))?,
            #[cfg(feature = "gpu")]
            MatvecFace::Gpu(ctx) => ctx
                .packed_matvec(&self.vb_t_rows, self.r, self.din, &z)
                .map_err(|e| EngineError(format!("layer {}: Vb^T GPU matvec: {e}", self.name)))?,
        };
        let u = match face {
            MatvecFace::Cpu => cpu_matvec(&self.ub_rows, self.dout, self.r, &t)
                .map_err(|e| EngineError(format!("layer {}: Ub matvec: {e}", self.name)))?,
            #[cfg(feature = "gpu")]
            MatvecFace::Gpu(ctx) => ctx
                .packed_matvec(&self.ub_rows, self.dout, self.r, &t)
                .map_err(|e| EngineError(format!("layer {}: Ub GPU matvec: {e}", self.name)))?,
        };
        Ok(u.iter()
            .zip(&self.s1_f32)
            .zip(&self.bias_f32)
            .map(|((&u_i, &s1_i), &b_i)| (u_i as f32) * s1_i + b_i)
            .collect())
    }
}

/// Resolved per-block view: layer INDICES into `ModelWeights::layers` and
/// ABSOLUTE norm offsets into `ModelWeights::norm_weights`, derived from
/// the declared inventory at load (E-2.3: presence in the inventory, not
/// any model family, decides which norm points run). Plain data, so
/// `ModelWeights` stays self-contained behind `Arc`.
struct BlockRefs {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
    gate: usize,
    up: usize,
    down: usize,
    /// Absolute sidecar float offset of the block-input norm (required).
    input_norm_at: usize,
    /// Absolute sidecar float offset of the post-attention (pre-MLP)
    /// norm (required).
    post_attention_norm_at: usize,
    /// Absolute offsets of the optional slots; `None` = not declared,
    /// not applied (E-2.3). `post_feedforward` resolves today so an
    /// inventory declaring it loads; its APPLICATION is the Gemma-class
    /// sandwich semantics (E-2.0), consumed when a Gemma-class export
    /// exists to gate it.
    q_norm_at: Option<usize>,
    k_norm_at: Option<usize>,
    #[allow(dead_code)]
    post_feedforward_norm_at: Option<usize>,
}

impl BlockRefs {
    /// Resolve one block's projections by manifest name (HF module paths,
    /// e.g. `layers.3.inner.self_attn.q_proj`) and its norm offsets by
    /// walking the declared inventory. Fail-closed: a missing projection
    /// or a required slot absent from the inventory is a LOAD error.
    fn resolve(
        index: &std::collections::HashMap<String, usize>,
        norm_slots: &[NormSlot],
        block: usize,
        d_model: usize,
        head_dim: usize,
    ) -> Result<Self, EngineError> {
        let find = |suffix: &str| -> Result<usize, EngineError> {
            let name = format!("layers.{block}.inner.{suffix}");
            index
                .get(&name)
                .copied()
                .ok_or_else(|| EngineError(format!("block {block}: missing packed layer {name:?}")))
        };
        let q = find("self_attn.q_proj")?;
        let k = find("self_attn.k_proj")?;
        let v = find("self_attn.v_proj")?;
        let o = find("self_attn.o_proj")?;
        let gate = find("mlp.gate_proj")?;
        let up = find("mlp.up_proj")?;
        let down = find("mlp.down_proj")?;

        let mut input_norm_at = None;
        let mut post_attention_norm_at = None;
        let mut q_norm_at = None;
        let mut k_norm_at = None;
        let mut post_feedforward_norm_at = None;
        let mut at = block * inventory_block_floats(norm_slots, d_model, head_dim);
        for slot in norm_slots {
            match slot {
                NormSlot::Input => input_norm_at = Some(at),
                NormSlot::PostAttention => post_attention_norm_at = Some(at),
                NormSlot::QNorm => q_norm_at = Some(at),
                NormSlot::KNorm => k_norm_at = Some(at),
                NormSlot::PostFeedforward => post_feedforward_norm_at = Some(at),
            }
            at += slot.width(d_model, head_dim);
        }
        let missing = |name: &str| {
            EngineError(format!(
                "block {block}: required norm slot {name:?} not in the \
                 declared inventory {norm_slots:?}"
            ))
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
            input_norm_at: input_norm_at.ok_or_else(|| missing("input"))?,
            post_attention_norm_at: post_attention_norm_at
                .ok_or_else(|| missing("post_attention"))?,
            q_norm_at,
            k_norm_at,
            post_feedforward_norm_at,
        })
    }
}

/// Immutable after load (design D5/IA2); shared as `Arc`.
pub struct ModelWeights {
    pub layers: Vec<PackedLayer>,
    /// `vocab_size x d_model` fp32 embedding (row-major).
    pub embedding: Vec<f32>,
    /// `vocab_size x d_model` fp32 lm_head (row-major); `None` = weight-tied
    /// to `embedding`.
    pub lm_head: Option<Vec<f32>>,
    pub d_model: usize,
    pub vocab_size: usize,
    /// Validated attention geometry (qk-attention N-2). The norm sidecar
    /// below is laid out by the DECLARED inventory (`norm_slots`, qk-attention
    /// N-1.0): per block, each slot's vector in inventory order
    /// (`input`/`post_attention`/`post_feedforward` at `d_model` floats,
    /// `q_norm`/`k_norm` at `head_dim`), then the model-final norm
    /// (`d_model`). Every count is derived from the inventory — no model
    /// family's formula lives in the engine.
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Declared per-block norm inventory, in sidecar order. Read by the
    /// parity test's declared-inventory cross-check (E-2.3); the forward
    /// itself consumes the RESOLVED per-block offsets (`blocks`), not
    /// this list.
    #[allow(dead_code)]
    pub norm_slots: Vec<NormSlot>,
    pub norm_weights: Vec<f32>,
    /// Absolute sidecar offset of the model-final norm (last `d_model`
    /// floats).
    final_norm_at: usize,
    /// Per-block projection indices + norm offsets, resolved once at
    /// load from the manifest names + inventory (E-2.3).
    blocks: Vec<BlockRefs>,
    /// Frozen matvec-face decision (qk-attention 3.4): CPU rlib by
    /// default; HIP device face under `gpu` + QUANTIBBLER_GPU=1 with
    /// successful init. Selected once, never swapped mid-stream.
    matvec: FaceDecision,
    /// Frozen attention-face decision (qk-flash-attention 3.2): the in-line
    /// scalar routine by default; the tiled device face under `gpu` +
    /// QUANTIBBLER_ATTN=1 with successful init (a failed init soft-falls-back
    /// to scalar). Selected once, never swapped mid-stream.
    attention: AttentionFaceDecision,
}

fn f32ify(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

impl ModelWeights {
    /// Build weights from parsed qpk1 layers + sidecar matrices + the
    /// validated transformer contract (qk-attention N-1/N-2).
    #[allow(clippy::too_many_arguments)]
    pub fn from_layers(
        layers: Vec<(String, Layer)>,
        embedding: Vec<f64>,
        lm_head: Option<Vec<f64>>,
        d_model: usize,
        vocab_size: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rope_theta: f64,
        norm_slots: Vec<NormSlot>,
        norm_weights: Vec<f64>,
    ) -> Result<Self, EngineError> {
        if embedding.len() != d_model * vocab_size {
            return Err(EngineError(format!(
                "embedding is {} floats; expected {vocab_size}x{d_model} = {}",
                embedding.len(),
                d_model * vocab_size
            )));
        }
        if let Some(head) = &lm_head {
            if head.len() != d_model * vocab_size {
                return Err(EngineError(format!(
                    "lm_head is {} floats; expected {vocab_size}x{d_model} = {}",
                    head.len(),
                    d_model * vocab_size
                )));
            }
        }
        if layers.is_empty() {
            return Err(EngineError("model has no qpk1 layers".into()));
        }
        let packed = layers
            .into_iter()
            .map(|(name, layer)| PackedLayer::from_layer(name, &layer))
            .collect::<Result<Vec<_>, _>>()?;
        // Sidecar length is INVENTORY-derived (qk-attention N-1.0): the
        // engine, like the loader, knows no model family's formula.
        let per_block = inventory_block_floats(&norm_slots, d_model, head_dim);
        let expected_norm_floats = per_block
            .checked_mul(n_layers)
            .and_then(|p| p.checked_add(d_model))
            .ok_or_else(|| EngineError("norm sidecar count overflows".into()))?;
        if norm_weights.len() != expected_norm_floats {
            return Err(EngineError(format!(
                "norm sidecar is {} floats; inventory {:?} at d_model {d_model}, \
                 head_dim {head_dim} over {n_layers} layers requires {expected_norm_floats}",
                norm_weights.len(),
                norm_slots.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )));
        }
        let norm_weights_f32 = f32ify(norm_weights);
        let final_norm_at = norm_weights_f32
            .len()
            .checked_sub(d_model)
            .ok_or_else(|| EngineError("norm sidecar shorter than d_model".into()))?;
        let index: std::collections::HashMap<String, usize> = packed
            .iter()
            .enumerate()
            .map(|(i, layer)| (layer.name.clone(), i))
            .collect();
        let blocks = (0..n_layers)
            .map(|b| BlockRefs::resolve(&index, &norm_slots, b, d_model, head_dim))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers: packed,
            embedding: f32ify(embedding),
            lm_head: lm_head.map(f32ify),
            d_model,
            vocab_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            norm_slots,
            norm_weights: norm_weights_f32,
            final_norm_at,
            blocks,
            // Face selection happens once, here (task 3.4): default builds
            // are unconditionally CPU; `gpu` builds consult
            // QUANTIBBLER_GPU=1 and soft-fall-back on init failure.
            matvec: FaceDecision::select_from_env(),
            // Attention face selection happens once too, at the same
            // construction point (task 3.2): default builds are the in-line
            // scalar routine; `gpu` builds consult QUANTIBBLER_ATTN=1 and
            // soft-fall-back to scalar on init failure.
            attention: AttentionFaceDecision::select_from_env(),
        })
    }

    /// The frozen matvec-face label (startup log + /health reporting).
    pub fn matvec_face(&self) -> &'static str {
        self.matvec.describe()
    }

    /// The frozen attention-face label (startup log + /health reporting).
    pub fn attention_face(&self) -> &'static str {
        self.attention.describe()
    }

    /// Install a face decision the constructor cannot compute — today only the
    /// `gpu:sage` decision, whose startup gate needs the MODEL DIRECTORY (the
    /// record lives there) rather than just the config (qk-sage-attention 3.2).
    /// Called once, by `model::load`, before the weights are shared.
    pub fn set_attention_face(&mut self, decision: AttentionFaceDecision) {
        self.attention = decision;
    }

    /// Hidden vector for one token: `embedding[token]`.
    pub fn embed(&self, token: u32) -> Result<Vec<f32>, EngineError> {
        let t = token as usize;
        if t >= self.vocab_size {
            return Err(EngineError(format!(
                "token id {token} is outside the vocabulary (size {})",
                self.vocab_size
            )));
        }
        Ok(self.embedding[t * self.d_model..(t + 1) * self.d_model].to_vec())
    }

    /// Logits for one position: transformer forward (qk-attention E-2.2).
    /// One decode step over the session's KV cache.
    pub fn forward(&self, x: &[f32], cache: &mut KvCache) -> Result<Vec<f32>, EngineError> {
        let logits_row = self.transformer(x, cache)?;
        let head = self.lm_head.as_ref().unwrap_or(&self.embedding);
        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let row = &head[v * self.d_model..(v + 1) * self.d_model];
            let mut acc = 0.0f32;
            for t in 0..self.d_model {
                acc += logits_row[t] * row[t];
            }
            *logit = acc;
        }
        Ok(logits)
    }

    /// Prefill over a token prefix: transformer forward for ALL positions,
    /// logits flattened `[seq x vocab]` (E-2.2). Fills the KV cache as it
    /// goes; a subsequent single-token forward over the same cache must
    /// reproduce the last row bit-identically (teacher-forced invariance,
    /// E-2.1's determinism gate).
    #[allow(dead_code)]
    pub fn forward_prefill(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, EngineError> {
        let mut all = Vec::with_capacity(tokens.len() * self.vocab_size);
        let head = self.lm_head.as_ref().unwrap_or(&self.embedding);
        for &tok in tokens {
            let h = self.embed(tok)?;
            let row = self.transformer(&h, cache)?;
            for v in 0..self.vocab_size {
                let mut acc = 0.0f32;
                let row_v = &head[v * self.d_model..(v + 1) * self.d_model];
                for t in 0..self.d_model {
                    acc += row[t] * row_v[t];
                }
                all.push(acc);
            }
        }
        Ok(all)
    }

    /// The transformer trunk for ONE position: all blocks over the KV
    /// cache, final norm, hidden out. Norm application points follow the
    /// DECLARED inventory (E-2.0/E-2.3); packed matvecs keep the kernels
    /// rlib's f64 accumulation semantics.
    fn transformer(&self, x: &[f32], cache: &mut KvCache) -> Result<Vec<f32>, EngineError> {
        if x.len() != self.d_model {
            return Err(EngineError(format!(
                "transformer input is {} floats; d_model is {}",
                x.len(),
                self.d_model
            )));
        }
        let pos = cache.len(0);
        let eps = self.rms_norm_eps;
        let norm_w = |at: usize, len: usize| &self.norm_weights[at..at + len];
        let mut h = x.to_vec();
        let mut scratch = Vec::with_capacity(self.d_model);

        for (b, block) in self.blocks.iter().enumerate() {
            debug_assert_eq!(
                cache.len(b),
                pos,
                "KV cache blocks must advance in lockstep"
            );
            // --- attention sublayer ---
            rms_norm_into(
                &h,
                norm_w(block.input_norm_at, self.d_model),
                eps,
                &mut scratch,
            );
            let normed = scratch.clone();
            let apply_proj = |layer_idx: usize, input: &[f32]| -> Result<Vec<f32>, EngineError> {
                self.layers[layer_idx].apply(&self.matvec.face, input)
            };
            let q = apply_proj(block.q, &normed)?;
            let k = apply_proj(block.k, &normed)?;
            let v = apply_proj(block.v, &normed)?;
            let (mut qh, mut kh) = (q, k);
            if let (Some(at), Some(kw_at)) = (block.q_norm_at, block.k_norm_at) {
                per_head_rms_norm(&mut qh, self.head_dim, norm_w(at, self.head_dim), eps);
                per_head_rms_norm(&mut kh, self.head_dim, norm_w(kw_at, self.head_dim), eps);
            }
            rope_rotate(&mut qh, self.head_dim, pos, self.rope_theta);
            rope_rotate(&mut kh, self.head_dim, pos, self.rope_theta);
            cache.push(b, &kh, &v);

            // GQA causal attention over the cache. `pos` is the query
            // position (cache held pos+1 rows incl. this step after push).
            let n_groups = self.n_heads / self.n_kv_heads;
            let attn_out = self.gqa_attention(&qh, b, cache, n_groups, pos)?;
            let mut attn = self.layers[block.o].apply(&self.matvec.face, &attn_out)?;
            for (hh, &r) in attn.iter_mut().zip(&h) {
                *hh += r;
            }
            h = attn;

            // --- MLP sublayer ---
            rms_norm_into(
                &h,
                norm_w(block.post_attention_norm_at, self.d_model),
                eps,
                &mut scratch,
            );
            let normed = scratch.clone();
            let gate = self.layers[block.gate].apply(&self.matvec.face, &normed)?;
            let up = self.layers[block.up].apply(&self.matvec.face, &normed)?;
            let hidden = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| g * u / (1.0 + (-g).exp()))
                .collect::<Vec<f32>>();
            let mut mlp = self.layers[block.down].apply(&self.matvec.face, &hidden)?;
            for (hh, &r) in mlp.iter_mut().zip(&h) {
                *hh += r;
            }
            h = mlp;
        }

        rms_norm_into(
            &h,
            norm_w(self.final_norm_at, self.d_model),
            eps,
            &mut scratch,
        );
        Ok(scratch)
    }

    /// GQA causal attention for one query position, on the SELECTED face
    /// (task 3.2): the in-line scalar routine by default, the tiled device
    /// face under `gpu` + `QUANTIBBLER_ATTN=1`. Both solve the same problem —
    /// the device call reproduces the conventions `gqa_attention_scalar`
    /// documents.
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(
        &self,
        qh: &[f32],
        block: usize,
        cache: &KvCache,
        n_groups: usize,
        pos: usize,
    ) -> Result<Vec<f32>, EngineError> {
        match &self.attention.face {
            AttentionFace::CpuScalar => self.gqa_attention_scalar(qh, block, cache, n_groups, pos),
            #[cfg(feature = "gpu")]
            AttentionFace::GpuTiled(ctx) => self.gqa_attention_tiled(ctx, qh, block, cache, pos),
            #[cfg(feature = "gpu")]
            AttentionFace::GpuSage(ctx) => self.gqa_attention_sage(ctx, qh, block, cache, pos),
        }
    }

    /// One decode step on the INT8 face (qk-sage-attention 3.2). `n_groups` is
    /// not a parameter: the device wrapper materializes GQA groups itself (the
    /// same `repeat_interleave` the tiled call relies on).
    ///
    /// The shipped INT8 kernel is self-attention over ONE cache — it requires
    /// `seq_q == seq_kv`, because γ's shift is taken over the key block in
    /// flight (the reference's own crystallization entry, task 1.1–1.2). The
    /// decode cache already holds exactly this query's causal window (`0..=pos`,
    /// pushed before this call), so the query goes into **row 0** of a
    /// `pos + 1`-row buffer padded with zeros and `causal = false`: row 0 attends
    /// to every key present, which IS the causal window for the last cached
    /// position. Only row 0 of each head is read; the padding rows are waste
    /// (`block_size`-independent, O(seq_kv²) per step) accepted as the price of
    /// the shipped kernel contract rather than a silent reinterpretation of it.
    #[cfg(feature = "gpu")]
    fn gqa_attention_sage(
        &self,
        ctx: &GpuContext,
        qh: &[f32],
        block: usize,
        cache: &KvCache,
        pos: usize,
    ) -> Result<Vec<f32>, EngineError> {
        debug_assert_eq!(cache.len(block), pos + 1, "current step must be cached");
        let head_dim = self.head_dim;
        let seq = pos + 1;
        let mut padded_q = vec![0.0f32; self.n_heads * seq * head_dim];
        for head in 0..self.n_heads {
            let dst = head * seq * head_dim;
            padded_q[dst..dst + head_dim]
                .copy_from_slice(&qh[head * head_dim..(head + 1) * head_dim]);
        }
        let shape = AttentionShape {
            batch: 1,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            seq_q: seq,
            seq_kv: seq,
            head_dim,
            val_dim: head_dim,
        };
        // K/V go over in the device contract's (1, n_kv_heads, seq_kv,
        // head_dim) plane — head-major, hence the gather out of the row-major
        // cache (`KvCache::keys_head_major`). `attention_sage` mirrors
        // `GpuContext::attention`'s layout, whose `repeat_heads` indexes the
        // source as `((b * n_kv_heads + kv) * seq) * dim`, and the INT8 kernel
        // reads the same plane (`kv_tok = (b * n_kv + kvh) * seq`). Handing it
        // the raw row-major stack is the O(1) transposition task 3.3 measured
        // on the sibling face.
        let keys = cache.keys_head_major(block, self.n_kv_heads, self.head_dim);
        let vals = cache.vals_head_major(block, self.n_kv_heads, self.head_dim);
        let out = ctx
            .attention_sage(shape, &padded_q, &keys, &vals, false, SAGE_BLOCK_SIZE)
            .map_err(|e| EngineError(format!("block {block}: sage attention: {e}")))?;
        let mut attn = vec![0.0f32; self.n_heads * head_dim];
        for head in 0..self.n_heads {
            let src = head * seq * head_dim; // row 0 of this head's plane
            attn[head * head_dim..(head + 1) * head_dim].copy_from_slice(&out[src..src + head_dim]);
        }
        Ok(attn)
    }

    /// One decode step on the tiled device face: ONE query position against the
    /// KV cache, in the device contract's `(B, H, S, d)` layout. `seq_q = 1`
    /// with `causal = true` attends over `0..=pos` (the substrate's causality is
    /// `j + seq_q > i + seq_kv`, which is exactly the scalar routine's window),
    /// and `n_kv_heads` carries the GQA grouping — the substrate's missing
    /// first-class GQA is the kernels wrapper's host-side materialization, not a
    /// convention change here. `val_dim == head_dim`: `v_proj` emits
    /// `n_kv_heads * head_dim`.
    #[cfg(feature = "gpu")]
    fn gqa_attention_tiled(
        &self,
        ctx: &GpuContext,
        qh: &[f32],
        block: usize,
        cache: &KvCache,
        pos: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let shape = AttentionShape {
            batch: 1,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            seq_q: 1,
            seq_kv: pos + 1,
            head_dim: self.head_dim,
            val_dim: self.head_dim,
        };
        // K/V go over in the device contract's (1, n_kv_heads, seq_kv,
        // head_dim) plane — head-major, hence the gather out of the
        // row-major cache (`KvCache::keys_head_major`). The query needs no
        // re-order: at `seq_q = 1`, `(1, n_heads, 1, head_dim)` and
        // `(n_heads, head_dim)` are the same bytes.
        let keys = cache.keys_head_major(block, self.n_kv_heads, self.head_dim);
        let vals = cache.vals_head_major(block, self.n_kv_heads, self.head_dim);
        ctx.attention(shape, qh, &keys, &vals, true)
            .map_err(|e| EngineError(format!("block {block}: tiled attention: {e}")))
    }

    /// GQA causal attention for one query position: per KV group, the
    /// query head attends over positions `0..=pos`; scores are
    /// dot(query, key)/sqrt(head_dim) in f32, softmax over the causal
    /// window, output the value mixture.
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention_scalar(
        &self,
        qh: &[f32],
        block: usize,
        cache: &KvCache,
        n_groups: usize,
        pos: usize,
    ) -> Result<Vec<f32>, EngineError> {
        debug_assert_eq!(cache.len(block), pos + 1, "current step must be cached");
        let mut out = vec![0.0f32; self.n_heads * self.head_dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for head in 0..self.n_heads {
            let kv_head = head / n_groups;
            let q = &qh[head * self.head_dim..(head + 1) * self.head_dim];
            // scores for positions 0..=pos (causal: the last is this step)
            let mut scores: Vec<f32> = (0..=pos)
                .map(|p| {
                    // GQA: this head attends to its group's kv_head slice.
                    let k = &cache.key(block, p)
                        [kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    let mut dot = 0.0f32;
                    for t in 0..self.head_dim {
                        dot += q[t] * k[t];
                    }
                    dot * scale
                })
                .collect();
            let maxs = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxs).exp();
                denom += *s;
            }
            let oh = &mut out[head * self.head_dim..(head + 1) * self.head_dim];
            for (p, score) in scores.iter().enumerate() {
                let w = score / denom;
                if w == 0.0 {
                    continue;
                }
                let val =
                    &cache.val(block, p)[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                for t in 0..self.head_dim {
                    oh[t] += w * val[t];
                }
            }
        }
        Ok(out)
    }

    /// Superseded packed stack forward (kept until 2.2/2.3 replace the
    /// call sites; the parity test switches to `forward`). Dead until then.
    #[allow(dead_code)]
    pub fn forward_packed_legacy(&self, x: &[f32]) -> Result<Vec<f32>, EngineError> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.apply(&self.matvec.face, &h)?;
        }
        let head = self.lm_head.as_ref().unwrap_or(&self.embedding);
        if h.len() != self.d_model {
            return Err(EngineError(format!(
                "layer stack produced dim {}, lm_head expects {}",
                h.len(),
                self.d_model
            )));
        }
        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let row = &head[v * self.d_model..(v + 1) * self.d_model];
            let mut acc = 0.0f32;
            for t in 0..self.d_model {
                acc += h[t] * row[t];
            }
            *logit = acc;
        }
        Ok(logits)
    }
}

/// RMSNorm into `out`: `x / sqrt(mean(x^2) + eps) * weight`, f32 end to
/// end (spec E-2.0: norm + residual accumulation is f32).
fn rms_norm_into(x: &[f32], weight: &[f32], eps: f64, out: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), weight.len(), "norm width must match weight");
    let ss = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps as f32).sqrt();
    out.clear();
    out.extend(x.iter().zip(weight).map(|(&v, &w)| v * inv * w));
}

/// Per-head RMSNorm over `head_dim` slices (Qwen3 q/k-norm: ONE weight
/// vector broadcast across every head; each head normalizes over its own
/// head_dim slice), in place.
fn per_head_rms_norm(v: &mut [f32], head_dim: usize, weight: &[f32], eps: f64) {
    let mut buf = Vec::with_capacity(head_dim);
    for head in v.chunks_mut(head_dim) {
        rms_norm_into(head, weight, eps, &mut buf);
        head.copy_from_slice(&buf);
    }
}

/// RoPE rotation, HF rotate_half convention (Llama/Qwen line): the head
/// dim splits into halves and pairs are `(i, i + half)` —
/// `x'_i = x_i cos_i - x_{i+half} sin_i`,
/// `x'_{i+half} = x_{i+half} cos_i + x_i sin_i`, with
/// `inv_freq_i = theta^(-2i/head_dim)`. Rotates every `head_dim` slice of
/// `v` (Q: n_heads slices, K: n_kv_heads) by the position's angle, in
/// place, f32.
fn rope_rotate(v: &mut [f32], head_dim: usize, pos: usize, theta: f64) {
    let half = head_dim / 2;
    let n_heads = v.len() / head_dim;
    for head in 0..n_heads {
        let base = head * head_dim;
        for i in 0..half {
            let inv_freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
            let (s, c) = (pos as f64 * inv_freq).sin_cos();
            let (s, c) = (s as f32, c as f32);
            let a = v[base + i];
            let b = v[base + half + i];
            v[base + i] = a * c - b * s;
            v[base + half + i] = b * c + a * s;
        }
    }
}

/// Per-block, append-only KV cache: `k[block]` holds `pos_used[block]`
/// (public: the parity test drives prefill/step directly).
/// rows of `kv_dim` (= n_kv_heads * head_dim) f32, one row per position.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(n_blocks: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_blocks],
            v: vec![Vec::new(); n_blocks],
            kv_dim,
        }
    }

    fn push(&mut self, block: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);
        self.k[block].extend_from_slice(k);
        self.v[block].extend_from_slice(v);
    }

    /// Valid positions stored for `block`.
    fn len(&self, block: usize) -> usize {
        self.k[block].len() / self.kv_dim
    }

    fn key(&self, block: usize, pos: usize) -> &[f32] {
        &self.k[block][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }

    fn val(&self, block: usize, pos: usize) -> &[f32] {
        &self.v[block][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }

    /// The block's cached key stack in the DEVICE contract's layout
    /// `(1, n_kv_heads, seq_kv, head_dim)` (see `AttentionShape`): head-major,
    /// so it is built by a host-side gather out of the row-major cache. The
    /// gather is not optional — the layouts differ whenever `n_kv_heads > 1`,
    /// which the fixture model (2 kv heads) exhibits.
    ///
    /// The raw row-major stack (`(seq_kv, n_kv_heads, head_dim)`, each
    /// position's `kv_dim` row contiguous) is what `key` / `val` and the scalar
    /// routine read, and handing THAT to the device is an O(1) wrong-compute:
    /// measured on `gqa_attention_tiled` (1.634708e-1, task 3.3) and on
    /// `gqa_attention_sage` (task 3.4). The two orders coincide only at
    /// `n_kv_heads == 1`. Every device consumer takes this accessor instead.
    #[cfg(feature = "gpu")]
    fn keys_head_major(&self, block: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        self.head_major(&self.k[block], block, n_kv_heads, head_dim)
    }

    /// Values, same layout contract as `KvCache::keys_head_major`.
    #[cfg(feature = "gpu")]
    fn vals_head_major(&self, block: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        self.head_major(&self.v[block], block, n_kv_heads, head_dim)
    }

    /// The shared gather behind the two accessors above: `(seq_kv, kv_dim)`
    /// row-major in, `(n_kv_heads, seq_kv, head_dim)` head-major out.
    #[cfg(feature = "gpu")]
    fn head_major(
        &self,
        flat: &[f32],
        block: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let seq = self.len(block);
        debug_assert_eq!(flat.len(), seq * n_kv_heads * head_dim);
        let mut out = vec![0.0f32; seq * n_kv_heads * head_dim];
        for head in 0..n_kv_heads {
            for pos in 0..seq {
                let src = pos * self.kv_dim + head * head_dim;
                let dst = (head * seq + pos) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&flat[src..src + head_dim]);
            }
        }
        out
    }
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token (or a stop sequence) ended the sequence.
    Stop,
    /// `max_tokens` reached.
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Result of one greedy generation.
#[derive(Debug)]
pub struct GenerationOutput {
    pub prompt_tokens: usize,
    pub completion_tokens: Vec<u32>,
    pub finish: FinishReason,
}

/// Per-request decode state (design D6's plain per-request state; held for
/// the request's duration, dropped with it).
pub struct DecodeSession {
    /// Per-block KV cache, prefilled with the prompt in `decode_start`.
    cache: KvCache,
    out: Vec<u32>,
    max_tokens: usize,
    /// All stop sequences; the session terminates when the decoded text
    /// contains any of them.
    stop: Option<Vec<String>>,
    finish: Option<FinishReason>,
}

impl DecodeSession {
    pub fn finish(&self) -> Option<FinishReason> {
        self.finish
    }
}

/// The concurrency-bounded engine: shared weights + permit budget.
pub struct Engine {
    pub weights: Arc<ModelWeights>,
    pub tokenizer: tokenizers::Tokenizer,
    pub eos_token_id: u32,
    /// BOS is not prepended in v0.1 (prompts are taken as given); carried
    /// for the config contract and future scheduler use.
    #[allow(dead_code)]
    pub bos_token_id: u32,
    permits: Arc<Semaphore>,
    /// Test-only fault injection for the SR-7.3 envelope scenario (the
    /// `#[cfg(test)]` gate keeps it out of the shipped binary).
    #[cfg(test)]
    pub fail_generate: Arc<AtomicBool>,
}

impl Engine {
    pub fn new(model: crate::model::LoadedModel, max_concurrency: usize) -> Self {
        Self {
            weights: model.weights,
            tokenizer: model.tokenizer,
            eos_token_id: model.eos_token_id,
            bos_token_id: model.bos_token_id,
            permits: Arc::new(Semaphore::new(max_concurrency)),
            #[cfg(test)]
            fail_generate: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Acquire one generation permit (backpressures when busy).
    pub async fn acquire(&self) -> tokio::sync::OwnedSemaphorePermit {
        Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .expect("engine semaphore never closed")
    }

    /// Available permits (health reporting).
    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    /// Tokenize a prompt (token counting uses the same tokenizer).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| EngineError(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Start a decode session over an already-tokenized prompt.
    pub fn decode_start(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        stop: Option<&[String]>,
    ) -> Result<DecodeSession, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError("prompt tokenized to zero tokens".into()));
        }
        // Prefill the whole prompt through the real transformer (E-2.2):
        // trunk over every prompt position, head projection only for the
        // last (the position whose argmax produces the first token).
        let mut cache = KvCache::new(
            self.weights.n_layers,
            self.weights.n_kv_heads * self.weights.head_dim,
        );
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            let h = self.weights.embed(tok)?;
            self.weights.transformer(&h, &mut cache)?;
        }
        let h_last = self
            .weights
            .embed(*prompt_tokens.last().expect("non-empty"))?;
        let logits = self.weights.forward(&h_last, &mut cache)?;
        let mut session = DecodeSession {
            cache,
            out: Vec::new(),
            max_tokens,
            stop: stop.map(<[String]>::to_vec),
            finish: None,
        };
        self.push_token(&mut session, crate::sampler::greedy(&logits))?;
        Ok(session)
    }

    /// Apply EOS/stop/max-token semantics to one produced token; returns
    /// whether the session consumed it (false = finished).
    fn push_token(&self, session: &mut DecodeSession, next: u32) -> Result<bool, EngineError> {
        if next == self.eos_token_id {
            // EOS is the terminal control token: it terminates the
            // generation and is NOT part of the completion (OpenAI
            // semantics; usage counts exclude it).
            session.finish = Some(FinishReason::Stop);
            return Ok(false);
        }
        session.out.push(next);
        if let Some(stops) = &session.stop {
            let text = self
                .tokenizer
                .decode(&session.out, false)
                .map_err(|e| EngineError(format!("decode: {e}")))?;
            if stops.iter().any(|s| text.contains(s.as_str())) {
                // The completing token stays out of the returned completion;
                // the text-side trim to pre-stop content happens once at the
                // API boundary (earliest-stop cut), not per token here.
                session.finish = Some(FinishReason::Stop);
                return Ok(false);
            }
        }
        if session.out.len() >= session.max_tokens {
            session.finish = Some(FinishReason::Length);
            return Ok(false);
        }
        Ok(true)
    }

    /// One greedy decode step (argmax over logits, design D6). Returns the
    /// newly produced token, or `None` when the session has finished (EOS,
    /// stop sequence, or `max_tokens`).
    pub fn decode_step(&self, session: &mut DecodeSession) -> Result<Option<u32>, EngineError> {
        if session.finish.is_some() {
            return Ok(None);
        }
        // One transformer step: embed the last produced token, run the
        // trunk over the KV cache, argmax the logits.
        let last = *session.out.last().expect("non-empty session");
        let h = self.weights.embed(last)?;
        let logits = self.weights.forward(&h, &mut session.cache)?;
        let next = crate::sampler::greedy(&logits);
        if self.push_token(session, next)? {
            Ok(Some(next))
        } else {
            Ok(None)
        }
    }

    /// Full greedy generation (design D6): runs a session to completion.
    /// The permit proves engine access was acquired (design D5); the
    /// per-request state lives in the session, dropped with the request.
    pub fn generate(
        &self,
        _permit: &tokio::sync::OwnedSemaphorePermit,
        prompt_tokens: &[u32],
        max_tokens: usize,
        stop_sequence: Option<&[String]>,
    ) -> Result<GenerationOutput, EngineError> {
        #[cfg(test)]
        if self.fail_generate.load(Ordering::SeqCst) {
            return Err(EngineError("injected engine failure (test)".into()));
        }
        let mut session = self.decode_start(prompt_tokens, max_tokens, stop_sequence)?;
        while self.decode_step(&mut session)?.is_some() {}
        Ok(GenerationOutput {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: session.out,
            finish: session.finish.unwrap_or(FinishReason::Length),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantibbler_qpk1::{Layer, ScaleDtype};

    /// r=8 layer (container and row bitplane layouts coincide) with a
    /// hand-checkable product.
    /// Seed-variant of `layer_fixture` (deterministic per-projection
    /// signs/scales) so the seven projections of the transformer fixture
    /// are not identical copies of one matrix.
    fn layer_fixture_with_seed(seed: u64) -> Layer {
        let mut layer = layer_fixture();
        let rot = |v: &[f64], by: u64| {
            v.iter()
                .enumerate()
                .map(|(i, x)| {
                    let k = (i as u64 + by) % v.len() as u64;
                    let s = if (k % 3) == 2 { -1.0 } else { 1.0 };
                    x * s * (1.0 + 0.125 * k as f64)
                })
                .collect()
        };
        layer.s1 = rot(&layer.s1, seed * 2 + 1);
        layer.s2 = rot(&layer.s2, seed * 3 + 2);
        if seed % 2 == 1 {
            layer.ub_packed.iter_mut().for_each(|b| *b = !*b);
        }
        layer
    }

    fn layer_fixture() -> Layer {
        let dout = 4usize;
        let din = 4usize;
        let r = 8usize;
        let ub_signs: Vec<i8> = vec![
            1, -1, 1, 1, -1, 1, -1, -1, //
            1, 1, 1, -1, -1, -1, 1, 1, //
            -1, 1, -1, -1, 1, -1, 1, 1, //
            1, -1, -1, 1, 1, 1, -1, -1,
        ];
        let mut ub = vec![0u8; (dout * r) / 8];
        for (idx, &s) in ub_signs.iter().enumerate() {
            if s == 1 {
                ub[idx >> 3] |= 1 << (idx & 7);
            }
        }
        let vb = vec![0xffu8; (din * r) / 8];
        Layer {
            // A version-1 legacy layer: no bias term, so the dense reference
            // below is the pure factorized product.
            version: quantibbler_qpk1::LEGACY_VERSION,
            dout,
            din,
            r,
            dtype: ScaleDtype::Float16,
            s1: vec![1.0, 0.5, -2.0, 0.25],
            s2: vec![0.25, -0.5, 2.0, 1.0],
            bias: Vec::new(),
            ub_packed: ub,
            vb_packed: vb,
        }
    }

    /// `layer_fixture`'s factors/scales at version 2, plus a known
    /// per-output bias.
    fn bias_layer_fixture() -> Layer {
        let mut layer = layer_fixture();
        layer.version = quantibbler_qpk1::VERSION;
        layer.bias = vec![0.5, -1.5, 2.0, -0.25];
        layer
    }

    #[test]
    fn container_and_row_layouts_coincide_at_r_multiple_of_8() {
        let layer = layer_fixture();
        let packed = PackedLayer::from_layer("l0".into(), &layer).expect("layer ok");
        assert_eq!(packed.ub_rows, layer.ub_packed);
        assert_eq!(packed.vb_rows, layer.vb_packed);
    }

    #[test]
    fn container_to_rows_matches_reference_at_padded_rank() {
        // r = 5, 2 rows: container stream is 10 bits over 2 bytes;
        // row layout wants ceil(5/8)=1 byte per row.
        let signs = [[1, -1, 1, -1, 1], [-1, -1, 1, 1, -1]];
        let mut container = [0u8; 2];
        for (i, row) in signs.iter().enumerate() {
            for (j, &s) in row.iter().enumerate() {
                if s == 1 {
                    container[(i * 5 + j) >> 3] |= 1 << ((i * 5 + j) & 7);
                }
            }
        }
        let rows = container_to_rows(&container, 2, 5);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], 0b0001_0101); // bits 0,2,4
        assert_eq!(rows[1], 0b0000_1100); // bits 2,3
    }

    #[test]
    fn layer_apply_matches_dense_reference() {
        // W = diag(s1) Ub Vb^T diag(s2); x -> W x, checked against a dense
        // f64 computation of the same product.
        let layer = layer_fixture();
        let packed = PackedLayer::from_layer("l0".into(), &layer).expect("layer ok");
        let x = [0.5f32, -1.25, 2.0, 0.0];
        let mut y = vec![0f64; 4];
        // Vb is all-+1, so (Vb^T z)_k = sum_j z_j for every rank index k;
        // then u_i = sum_k ub[i][k] * (Vb^T z)_k, y_i = s1_i * u_i.
        let z: Vec<f64> = (0..4).map(|j| layer.s2[j] * f64::from(x[j])).collect();
        let vt_z: f64 = z.iter().sum();
        for (i, (y_i, s1_i)) in y.iter_mut().zip(layer.s1.iter()).enumerate() {
            let u_i: f64 = (0..8)
                .map(|k| {
                    let bit = (layer.ub_packed[(i * 8 + k) >> 3] >> ((i * 8 + k) & 7)) & 1;
                    let s = if bit == 1 { 1.0 } else { -1.0 };
                    s * vt_z
                })
                .sum();
            *y_i = u_i * s1_i;
        }
        let got = packed.apply(&MatvecFace::Cpu, &x).expect("apply ok");
        for (g, expected) in got.iter().zip(&y) {
            assert!(
                (f64::from(*g) - expected).abs() < 1e-4,
                "got {g}, expected {expected}"
            );
        }
    }

    /// `apply` is `W x + b`: the parsed bias is added per output feature, and
    /// a biasless (legacy) layer adds exactly zeros.
    #[test]
    fn layer_apply_adds_the_parsed_bias() {
        let x = [0.5f32, -1.25, 2.0, 0.0];
        let legacy = PackedLayer::from_layer("l0".into(), &layer_fixture()).expect("layer ok");
        let biased = PackedLayer::from_layer("l0".into(), &bias_layer_fixture()).expect("layer ok");
        assert_eq!(legacy.bias_f32, vec![0.0f32; 4], "legacy reads as zeros");
        assert_eq!(biased.bias_f32, vec![0.5f32, -1.5, 2.0, -0.25]);

        let y0 = legacy.apply(&MatvecFace::Cpu, &x).expect("apply ok");
        let y1 = biased.apply(&MatvecFace::Cpu, &x).expect("apply ok");
        assert_eq!(y0.len(), y1.len());
        for ((a, b), &bias) in y0.iter().zip(&y1).zip(&biased.bias_f32) {
            // The ONLY difference between the two layers is the bias vector.
            assert!(
                (f64::from(*b) - (f64::from(*a) + f64::from(bias))).abs() < 1e-6,
                "bias not applied: {a} + {bias} != {b}"
            );
        }
        // A wrong-width bias vector is a load-time error, not a silent
        // truncation or a panic inside the hot loop.
        let mut bad = bias_layer_fixture();
        bad.bias.pop();
        assert!(PackedLayer::from_layer("l0".into(), &bad).is_err());
    }

    #[test]
    fn argmax_first_max_tiebreak() {
        assert_eq!(crate::sampler::greedy(&[0.1, 3.0, 3.0, -1.0]), 1);
        assert_eq!(crate::sampler::greedy(&[-5.0, -6.0]), 0);
    }

    /// A full one-block transformer fixture: the layer fixture's (4,4,8)
    /// matrices under the engine's projection names, so `from_layers`
    /// resolves a real block (E-2.3 path).
    fn transformer_fixture_weights(
        embedding: Vec<f64>,
        vocab: usize,
        d_model: usize,
    ) -> ModelWeights {
        let suffixes = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ];
        let layers: Vec<(String, Layer)> = suffixes
            .iter()
            .enumerate()
            .map(|(i, suffix)| {
                (
                    format!("layers.0.inner.{suffix}"),
                    layer_fixture_with_seed(i as u64),
                )
            })
            .collect();
        ModelWeights::from_layers(
            layers,
            embedding,
            None,
            d_model,
            vocab,
            1,
            2,
            2,
            2,
            1e-6,
            10_000.0,
            vec![
                quantibbler_qpk1::NormSlot::Input,
                quantibbler_qpk1::NormSlot::PostAttention,
            ],
            vec![0.0; 3 * d_model],
        )
        .expect("transformer fixture resolves")
    }

    fn fixture_embedding(vocab: usize, d_model: usize) -> Vec<f64> {
        let mut embedding = vec![0f64; vocab * d_model];
        for (i, cell) in embedding.iter_mut().enumerate() {
            *cell = ((i % 5) as f64 - 2.0) * 0.5;
        }
        embedding
    }

    #[tokio::test]
    async fn engine_permit_bounds_and_generates() {
        let d_model = 4usize;
        let vocab = 32usize;
        let weights = Arc::new(transformer_fixture_weights(
            fixture_embedding(vocab, d_model),
            vocab,
            d_model,
        ));
        let model = crate::model::LoadedModel {
            id: "t".into(),
            weights: Arc::clone(&weights),
            tokenizer: test_tokenizer(),
            eos_token_id: u32::MAX, // unreachable: bound the run by max_tokens
            bos_token_id: 1,
            vocab_size: vocab,
            d_model,
        };
        let engine = Engine::new(model, 2);
        let permit = engine.acquire().await;
        assert_eq!(engine.available_permits(), 1);
        let out = engine
            .generate(&permit, &[3, 4, 5], 3, None)
            .expect("generation ok");
        assert_eq!(out.prompt_tokens, 3);
        assert!(out.completion_tokens.len() <= 3);
        // Determinism (E-2.1): the same prompt twice must produce the
        // identical token stream.
        let permit2 = engine.acquire().await;
        let out2 = engine
            .generate(&permit2, &[3, 4, 5], 3, None)
            .expect("generation ok");
        assert_eq!(out.completion_tokens, out2.completion_tokens);
        drop(permit);
        drop(permit2);
        assert_eq!(engine.available_permits(), 2);
    }

    #[tokio::test]
    async fn generate_stops_at_eos() {
        let d_model = 4usize;
        let vocab = 32usize;
        let weights = Arc::new(transformer_fixture_weights(
            fixture_embedding(vocab, d_model),
            vocab,
            d_model,
        ));
        // Self-consistent probe: the first greedy token from the fixture
        // (via the real prefill) becomes EOS, so the SAME run must
        // terminate with Stop and an empty completion (OpenAI semantics).
        let mut cache = KvCache::new(weights.n_layers, weights.n_kv_heads * weights.head_dim);
        let logits = weights
            .forward_prefill(&[10], &mut cache)
            .expect("prefill ok");
        let eos = crate::sampler::greedy(&logits[..vocab]);
        let model = crate::model::LoadedModel {
            id: "t".into(),
            weights,
            tokenizer: test_tokenizer(),
            eos_token_id: eos,
            bos_token_id: 1,
            vocab_size: vocab,
            d_model,
        };
        let engine = Engine::new(model, 1);
        let permit = engine.acquire().await;
        let out = engine.generate(&permit, &[10], 50, None).expect("gen ok");
        assert_eq!(out.finish, FinishReason::Stop);
        // EOS terminates the generation but is NOT part of the completion
        // (OpenAI semantics; usage counts exclude it).
        assert_eq!(out.completion_tokens, Vec::<u32>::new());
    }

    fn test_tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/server-model/tokenizer.json"),
        )
        .expect("fixture tokenizer loads")
    }
}
