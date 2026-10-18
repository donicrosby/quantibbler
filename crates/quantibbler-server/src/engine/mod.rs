//! The generation engine (design D5/D6): immutable shared weights behind
//! `Arc` (lock-free reads — weights are never written after load), engine
//! access bounded by a `tokio::sync::Semaphore` sized by `--max-concurrency`,
//! greedy decode with a plain per-request state cache over the
//! kernels-rlib packed primitives.

use std::sync::Arc;

use tokio::sync::Semaphore;

use quantibbler_kernels::core;
use quantibbler_qpk1::Layer;

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
        Ok(Self {
            name,
            dout: layer.dout,
            din: layer.din,
            r: layer.r,
            s1_f32: layer.s1.iter().map(|&v| v as f32).collect(),
            s2_f32: layer.s2.iter().map(|&v| v as f32).collect(),
            ub_rows,
            vb_rows,
            vb_t_rows,
        })
    }

    /// `y = diag(s1) @ (Ub @ (Vb^T @ (diag(s2) @ x)))` — the factorized
    /// linear transform `x -> W x` with
    /// `W = diag(s1) (Ub Vb^T) diag(s2)` (qk-representation Eq. 1).
    ///
    /// Bitplane products go through the kernels rlib (`packed_matvec`,
    /// float64 accumulation in row-element order); scales and I/O are f32.
    fn apply(&self, x: &[f32]) -> Result<Vec<f32>, EngineError> {
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
        let t = core::packed_matvec(&self.vb_t_rows, self.r, self.din, &z)
            .map_err(|e| EngineError(format!("layer {}: Vb^T matvec: {e}", self.name)))?;
        let u = core::packed_matvec(&self.ub_rows, self.dout, self.r, &t)
            .map_err(|e| EngineError(format!("layer {}: Ub matvec: {e}", self.name)))?;
        Ok(u.iter()
            .zip(&self.s1_f32)
            .map(|(&u_i, &s1_i)| (u_i as f32) * s1_i)
            .collect())
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
    /// Validated attention geometry (qk-attention N-2) + norm sidecar,
    /// laid out per block (input_layernorm, post_attention_layernorm) then
    /// the model-final norm — `(2 * n_layers + 1) * d_model` floats.
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub norm_weights: Vec<f32>,
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
        if norm_weights.len() != (2 * n_layers + 1) * d_model {
            return Err(EngineError(format!(
                "norm sidecar is {} floats; expected (2 * {n_layers} + 1) * {d_model} = {}",
                norm_weights.len(),
                (2 * n_layers + 1) * d_model
            )));
        }
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
            norm_weights: f32ify(norm_weights),
        })
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
    ///
    /// TODO(qk-attention 2.2/2.3): RMSNorm -> per-block QKVO/MLP over the
    /// packed factorized projections with RoPE, GQA causal attention over
    /// the KV cache, SwiGLU, residuals, final norm -> LM head. Unwired:
    /// until 2.2/2.3 land this fails closed instead of silently serving the
    /// wrong forward semantics.
    pub fn forward(&self, _x: &[f32]) -> Result<Vec<f32>, EngineError> {
        let _ = (
            &self.layers,
            &self.norm_weights,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.rms_norm_eps,
            self.rope_theta,
        );
        unimplemented!("transformer forward lands with qk-attention tasks 2.2/2.3 (spec E-2.2)")
    }

    /// Prefill over a token prefix: transformer forward for ALL positions,
    /// logits flattened `[seq x vocab]` (qk-attention E-2.2 vehicle; the
    /// 2.3 KV-cache decode steps must preserve per-prefix invariance).
    ///
    /// TODO(qk-attention 2.2/2.3): wire the real transformer semantics.
    /// Dead in the bin until then; the parity test is the only caller.
    #[allow(dead_code)]
    pub fn forward_prefill(&self, _tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let _ = (&self.layers, &self.norm_weights);
        unimplemented!("transformer prefill lands with qk-attention tasks 2.2/2.3 (spec E-2.2)")
    }

    /// Superseded packed stack forward (kept until 2.2/2.3 replace the
    /// call sites; the parity test switches to `forward`). Dead until then.
    #[allow(dead_code)]
    pub fn forward_packed_legacy(&self, x: &[f32]) -> Result<Vec<f32>, EngineError> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.apply(&h)?;
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
    h: Vec<f32>,
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
        let h = self
            .weights
            .embed(*prompt_tokens.last().expect("non-empty"))?;
        Ok(DecodeSession {
            h,
            out: Vec::new(),
            max_tokens,
            stop: stop.map(<[String]>::to_vec),
            finish: None,
        })
    }

    /// One greedy decode step (argmax over logits, design D6). Returns the
    /// newly produced token, or `None` when the session has finished (EOS,
    /// stop sequence, or `max_tokens`).
    pub fn decode_step(&self, session: &mut DecodeSession) -> Result<Option<u32>, EngineError> {
        if session.finish.is_some() || session.out.len() >= session.max_tokens {
            session.finish.get_or_insert(FinishReason::Length);
            return Ok(None);
        }
        let logits = self.weights.forward(&session.h)?;
        let next = crate::sampler::greedy(&logits);
        if next == self.eos_token_id {
            // EOS is the terminal control token: it terminates the
            // generation and is NOT part of the completion (OpenAI
            // semantics; usage counts exclude it).
            session.finish = Some(FinishReason::Stop);
            return Ok(None);
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
                return Ok(None);
            }
        }
        session.h = self.weights.embed(next)?;
        Ok(Some(next))
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
            dout,
            din,
            r,
            dtype: ScaleDtype::Float16,
            s1: vec![1.0, 0.5, -2.0, 0.25],
            s2: vec![0.25, -0.5, 2.0, 1.0],
            ub_packed: ub,
            vb_packed: vb,
        }
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
        let got = packed.apply(&x).expect("apply ok");
        for (g, expected) in got.iter().zip(&y) {
            assert!(
                (f64::from(*g) - expected).abs() < 1e-4,
                "got {g}, expected {expected}"
            );
        }
    }

    #[test]
    fn argmax_first_max_tiebreak() {
        assert_eq!(crate::sampler::greedy(&[0.1, 3.0, 3.0, -1.0]), 1);
        assert_eq!(crate::sampler::greedy(&[-5.0, -6.0]), 0);
    }

    #[tokio::test]
    #[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
    async fn engine_permit_bounds_and_generates() {
        let d_model = 4usize;
        let vocab = 32usize;
        let mut embedding = vec![0f64; vocab * d_model];
        for (i, cell) in embedding.iter_mut().enumerate() {
            *cell = ((i % 5) as f64 - 2.0) * 0.5;
        }
        let weights = ModelWeights::from_layers(
            vec![("l0".into(), layer_fixture())],
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
            vec![0.0; 3 * d_model],
        )
        .unwrap();
        let model = crate::model::LoadedModel {
            id: "t".into(),
            weights: Arc::new(weights),
            tokenizer: test_tokenizer(),
            eos_token_id: 2,
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
        // Greedy chain from embedding[5] with this fixture: 2 then EOS(2)…
        // eos=2 fires when the argmax picks 2; assert the invariant bounds
        // instead of the exact chain (layout-specific).
        assert!(out.completion_tokens.len() <= 3);
        drop(permit);
        assert_eq!(engine.available_permits(), 2);
    }

    #[tokio::test]
    #[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
    async fn generate_stops_at_eos() {
        // Probe shows the fixture's greedy chain reaches token 2 at step 1;
        // configure eos = 2 so the SAME run must terminate with Stop.
        let d_model = 4usize;
        let vocab = 32usize;
        let mut embedding = vec![0f64; vocab * d_model];
        for (i, cell) in embedding.iter_mut().enumerate() {
            *cell = ((i % 5) as f64 - 2.0) * 0.5;
        }
        let weights = ModelWeights::from_layers(
            vec![("l0".into(), layer_fixture())],
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
            vec![0.0; 3 * d_model],
        )
        .unwrap();
        let model = crate::model::LoadedModel {
            id: "t".into(),
            weights: Arc::new(weights),
            tokenizer: test_tokenizer(),
            eos_token_id: 2,
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
