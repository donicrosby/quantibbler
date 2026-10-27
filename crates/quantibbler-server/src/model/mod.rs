//! Model-directory acquisition (qk-server design D3): the qpk1 crate owns
//! the container contract (stream parse + hardened directory load — path
//! containment, stat-before-read, checked arithmetic, fail-closed
//! config); this module does the serving-side composition only: tokenizer
//! load, layer conversion to the engine's working layout, and the
//! LoadedModel handle. Fail-closed end to end: any problem aborts startup
//! with a diagnostic naming the failing file; the listener never starts
//! on a partial model.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use quantibbler_qpk1::loader;
use quantibbler_qpk1::{Blob, RemainderDtype};
use serde::Deserialize;

use crate::engine::AttentionFaceDecision;
use crate::engine::ModelWeights;

/// The attention-face opt-in (`QUANTIBBLER_ATTN`): `=1` is the exact tiled
/// device face (qk-flash-attention 3.2), `=sage` the gate-verified quantized
/// face (qk-sage-attention 3.2). Both are frozen at load.
pub const ATTENTION_OPTIN: &str = "QUANTIBBLER_ATTN";

/// The `gpu:sage` opt-in value.
pub const SAGE_OPTIN: &str = "sage";

/// The gate record's file name inside the model directory (spec: "shipped with
/// the model directory").
pub const ATTENTION_GATE_RECORD: &str = "attention_gate.json";

/// The gate record's schema tag.
pub const ATTENTION_GATE_SCHEMA: &str = "quantibbler.attention_gate/1";

/// [SA] §4.3's gate clauses, adopted verbatim by the change's spec delta
/// ("Cosine Sim ≥ 0.999 and Relative L1 ≤ 0.02 vs the exact face"). The gate
/// enforces THESE constants, never the record's own declared clauses: a record
/// that declares looser bounds is refused rather than trusted.
pub const GATE_COSINE_MIN: f64 = 0.999;
pub const GATE_RELATIVE_L1_MAX: f64 = 0.02;

/// How far the record's declared `worst` may sit from the aggregate of its own
/// per-call metrics. The tool computes both from the same f64 values, so this
/// only absorbs JSON round-off; a real disagreement is a doctored summary.
const GATE_AGGREGATE_TOL: f64 = 1e-12;

/// Why the `gpu:sage` face was refused at startup. Typed so the refusal class
/// is a value, not a string match — and every message names the gate and the
/// record, because a silently degraded model is worse than a failed start
/// (spec delta `qk-server`, "Gate fail refuses the face").
#[derive(Debug)]
pub enum AttentionGateError {
    /// No gate record in the model directory.
    Missing { record: PathBuf },
    /// The record exists but cannot be read.
    Unreadable { record: PathBuf, detail: String },
    /// Not JSON, not this schema, missing fields, a non-finite metric, or a
    /// clause set that contradicts the gate's own.
    Malformed { record: PathBuf, detail: String },
    /// The record was generated for a different model.
    ModelMismatch {
        record: PathBuf,
        served: String,
        claimed: String,
    },
    /// A measured metric is outside the clause it must satisfy.
    OutOfBounds {
        record: PathBuf,
        clause: &'static str,
        measured: f64,
        required: f64,
    },
}

impl std::fmt::Display for AttentionGateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { record } => write!(
                f,
                "attention gate: no gate record at {} (the gpu:sage face is opt-in only for models \
                 whose shipped record passes)",
                record.display()
            ),
            Self::Unreadable { record, detail } => write!(
                f,
                "attention gate: cannot read the gate record {}: {detail}",
                record.display()
            ),
            Self::Malformed { record, detail } => write!(
                f,
                "attention gate: the gate record {} is malformed: {detail}",
                record.display()
            ),
            Self::ModelMismatch {
                record,
                served,
                claimed,
            } => write!(
                f,
                "attention gate: the gate record {} was generated for model_id {claimed:?}, but \
                 the served model is {served:?}",
                record.display()
            ),
            Self::OutOfBounds {
                record,
                clause,
                measured,
                required,
            } => write!(
                f,
                "attention gate: the gate record {} fails its {clause} clause ({measured:.9} vs \
                 required {required:.9})",
                record.display()
            ),
        }
    }
}

impl std::error::Error for AttentionGateError {}

/// [SA] §4.3 metrics as the gate record carries them.
#[derive(Debug, Clone, Deserialize)]
pub struct AttentionGateMetrics {
    pub cosine: f64,
    pub relative_l1: f64,
    pub rmse: f64,
}

/// One captured attention call's metrics.
#[derive(Debug, Clone, Deserialize)]
pub struct AttentionGateCall {
    pub case: usize,
    #[serde(default)]
    pub block_size: usize,
    pub cosine: f64,
    pub relative_l1: f64,
    pub rmse: f64,
}

/// The clauses the record was generated under (audited against the gate's own).
#[derive(Debug, Clone, Deserialize)]
pub struct AttentionGateClauses {
    pub cosine_min: f64,
    pub relative_l1_max: f64,
}

/// The eval-set PPL clause. `None` on the record means no eval-set PPL was
/// measured at that gate run; the gate then admits the face on the [SA] §4.3
/// metrics alone and says so (see [`attention_gate_check`]).
#[derive(Debug, Clone, Deserialize)]
pub struct AttentionGatePpl {
    pub eval_set: String,
    pub delta: f64,
    pub bound: f64,
}

/// The shipped gate record, parsed.
#[derive(Debug, Clone, Deserialize)]
pub struct AttentionGateRecord {
    pub schema: String,
    pub model_id: String,
    #[serde(default)]
    pub activations_source: String,
    #[serde(default)]
    pub generated_at: String,
    #[serde(default)]
    pub code_commit: String,
    pub clauses: AttentionGateClauses,
    pub calls: Vec<AttentionGateCall>,
    pub worst: AttentionGateMetrics,
    #[serde(default)]
    pub ppl: Option<AttentionGatePpl>,
}

fn gate_refusal<T>(record: &Path, detail: impl std::fmt::Display) -> Result<T, AttentionGateError> {
    Err(AttentionGateError::Malformed {
        record: record.to_path_buf(),
        detail: detail.to_string(),
    })
}

/// Load and check a model directory's gate record (spec delta `qk-server`,
/// "Sage attention face with measured quality gate").
///
/// Fail-closed by construction: a missing, unreadable, malformed, foreign or
/// out-of-bounds record is a typed refusal, and the gate enforces its OWN
/// clause constants rather than the record's declared ones (a record cannot
/// widen its own gate). The one clause that is not required to be present is
/// the eval-set PPL delta: its only producer is task 4.2's real-model eval run,
/// so a record that carries none is admitted with the clause named in the
/// startup log — never silently, and never with a number the tree cannot
/// measure.
pub fn attention_gate_check(
    model_dir: &Path,
    served_model_id: &str,
) -> Result<AttentionGateRecord, AttentionGateError> {
    let record = model_dir.join(ATTENTION_GATE_RECORD);
    if !record.is_file() {
        return Err(AttentionGateError::Missing { record });
    }
    let raw = std::fs::read_to_string(&record).map_err(|e| AttentionGateError::Unreadable {
        record: record.clone(),
        detail: e.to_string(),
    })?;
    let parsed: AttentionGateRecord =
        serde_json::from_str(&raw).map_err(|e| AttentionGateError::Malformed {
            record: record.clone(),
            detail: e.to_string(),
        })?;
    if parsed.schema != ATTENTION_GATE_SCHEMA {
        return gate_refusal(
            &record,
            format!(
                "schema is {:?}, expected {ATTENTION_GATE_SCHEMA:?}",
                parsed.schema
            ),
        );
    }
    if parsed.model_id != served_model_id {
        return Err(AttentionGateError::ModelMismatch {
            record,
            served: served_model_id.to_string(),
            claimed: parsed.model_id,
        });
    }
    // The declared clause set must BE the gate's: a record generated under
    // looser bounds is stale (or forged), not admissible.
    if (parsed.clauses.cosine_min - GATE_COSINE_MIN).abs() > f64::EPSILON {
        return gate_refusal(
            &record,
            format!(
                "declares cosine_min={}, the gate requires {GATE_COSINE_MIN}",
                parsed.clauses.cosine_min
            ),
        );
    }
    if (parsed.clauses.relative_l1_max - GATE_RELATIVE_L1_MAX).abs() > f64::EPSILON {
        return gate_refusal(
            &record,
            format!(
                "declares relative_l1_max={}, the gate requires {GATE_RELATIVE_L1_MAX}",
                parsed.clauses.relative_l1_max
            ),
        );
    }
    if parsed.calls.is_empty() {
        return gate_refusal(&record, "records no captured attention calls");
    }
    // Per-call (the record's per-layer metrics), then their aggregate: the two
    // must agree, so a record cannot pass a blanket "worst" over per-call
    // numbers that fail on their own.
    for (index, call) in parsed.calls.iter().enumerate() {
        if call.case != index {
            return gate_refusal(
                &record,
                format!("call {index} is labelled case {}", call.case),
            );
        }
        if call.block_size == 0 {
            return gate_refusal(&record, format!("call {index} has block_size 0"));
        }
        for (clause, measured) in [
            ("cosine", call.cosine),
            ("relative_l1", call.relative_l1),
            ("rmse", call.rmse),
        ] {
            if !measured.is_finite() {
                return gate_refusal(
                    &record,
                    format!("call {index} {clause} is not finite: {measured}"),
                );
            }
        }
        if call.cosine < GATE_COSINE_MIN {
            return Err(AttentionGateError::OutOfBounds {
                record,
                clause: "cosine",
                measured: call.cosine,
                required: GATE_COSINE_MIN,
            });
        }
        if call.relative_l1 > GATE_RELATIVE_L1_MAX {
            return Err(AttentionGateError::OutOfBounds {
                record,
                clause: "relative_l1",
                measured: call.relative_l1,
                required: GATE_RELATIVE_L1_MAX,
            });
        }
    }
    let aggregate = |f: fn(&AttentionGateCall) -> f64, keep: fn(f64, f64) -> f64, init: f64| {
        parsed.calls.iter().map(f).fold(init, keep)
    };
    let computed = [
        (
            "cosine",
            parsed.worst.cosine,
            aggregate(|c| c.cosine, f64::min, f64::INFINITY),
        ),
        (
            "relative_l1",
            parsed.worst.relative_l1,
            aggregate(|c| c.relative_l1, f64::max, f64::NEG_INFINITY),
        ),
        (
            "rmse",
            parsed.worst.rmse,
            aggregate(|c| c.rmse, f64::max, f64::NEG_INFINITY),
        ),
    ];
    for (clause, declared, computed) in computed {
        if !declared.is_finite() {
            return gate_refusal(&record, format!("worst {clause} is not finite: {declared}"));
        }
        if (declared - computed).abs() > GATE_AGGREGATE_TOL {
            return gate_refusal(
                &record,
                format!("worst {clause} is {declared}, but the calls aggregate to {computed}"),
            );
        }
    }
    // The PPL clause, when the gate run declared one: the measured delta must be
    // inside the bound recorded at that run. Absent ⇒ admitted on the metrics,
    // with the gap named.
    if let Some(ppl) = &parsed.ppl {
        if !ppl.delta.is_finite() || !ppl.bound.is_finite() || ppl.bound <= 0.0 {
            return gate_refusal(
                &record,
                format!(
                    "ppl clause is not measurable: delta={}, bound={}",
                    ppl.delta, ppl.bound
                ),
            );
        }
        if ppl.delta > ppl.bound {
            return Err(AttentionGateError::OutOfBounds {
                record,
                clause: "ppl_delta",
                measured: ppl.delta,
                required: ppl.bound,
            });
        }
    }
    Ok(parsed)
}

/// The attention-face decision the `gpu:sage` opt-in asks for, or `None` when
/// the opt-in does not ask for it (then the ordinary
/// [`AttentionFaceDecision::select_from_env`] decision stands).
///
/// The gate runs BEFORE any device is touched, so a refused model never gets a
/// device context — and the refusal is the same in a build without the `gpu`
/// feature, where the face cannot exist at all.
pub fn sage_face_from_env(
    model_dir: &Path,
    served_model_id: &str,
) -> Result<Option<AttentionFaceDecision>, AttentionGateError> {
    if std::env::var(ATTENTION_OPTIN).as_deref() != Ok(SAGE_OPTIN) {
        return Ok(None);
    }
    let record = attention_gate_check(model_dir, served_model_id)?;
    // The record's provenance is logged, never gated on: the clauses above are
    // the gate; the date, commit and activations source are what a reader needs
    // to audit the run that produced them.
    tracing::info!(
        model_id = served_model_id,
        record = %model_dir.join(ATTENTION_GATE_RECORD).display(),
        calls = record.calls.len(),
        worst_cosine = record.worst.cosine,
        worst_relative_l1 = record.worst.relative_l1,
        activations_source = %record.activations_source,
        generated_at = %record.generated_at,
        code_commit = %record.code_commit,
        "attention gate admitted the gpu:sage face"
    );
    match &record.ppl {
        Some(ppl) => tracing::info!(
            model_id = served_model_id,
            eval_set = %ppl.eval_set,
            ppl_delta = ppl.delta,
            ppl_bound = ppl.bound,
            "attention gate: eval-set PPL clause"
        ),
        None => tracing::warn!(
            model_id = served_model_id,
            "attention gate: the record declares no eval-set PPL delta (task 4.2's real-model eval \
             run is its only producer); the face is admitted on the [SA] §4.3 metrics alone"
        ),
    }
    Ok(Some(AttentionFaceDecision::sage_selected()))
}

/// The loaded model: immutable weights plus the metadata the API needs.
pub struct LoadedModel {
    /// The served model id (`manifest.toml`'s `model_id`).
    pub id: String,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: tokenizers::Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    /// Recorded config hyperparameters (informational in v0.1; the weights
    /// carry their own dims).
    pub vocab_size: usize,
    pub d_model: usize,
    /// The byte source the blob loader used (BS-1); `None` for a directory
    /// source, which is never mapped.
    pub blob_load_method: Option<BlobLoadMethod>,
    /// The model's own Jinja2 chat template, read from the artifact's
    /// `tokenizer.json` `chat_template` field (BS-5). `None` = the artifact
    /// declares none, and `/v1/chat/completions` refuses (400) rather than
    /// guessing a prompt format.
    pub chat_template: Option<String>,
}

/// Build the tokenizer from an artifact `tokenizer.json` AND take the model's
/// chat template out of the same document (BS-5, design S5).
///
/// The document serves both purposes: it is the BPE definition the `tokenizers`
/// crate consumes, and it carries the model's Jinja2 `chat_template` (the crate
/// exposes no arbitrary metadata field).
///
/// The key must NOT reach the crate: `tokenizers` 0.23.x rejects any unknown
/// top-level key outright — measured, a real HF `tokenizer.json` plus one extra
/// key fails to parse at all ("expected `,` or `}`") — so the template is read
/// here and removed from the document the tokenizer is built from. Only a
/// document that actually declares the key is re-serialized; an artifact
/// without one is handed over byte-for-byte. A blank or non-string declaration
/// loads fine and yields `None`, so the chat endpoint refuses (BS-5.2) rather
/// than inventing a prompt format.
pub fn tokenizer_from_json(
    bytes: &[u8],
    label: &str,
) -> Result<(tokenizers::Tokenizer, Option<String>), String> {
    let mut parsed: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("{label}: {e}"))?;
    let declared = {
        let object = parsed
            .as_object_mut()
            .ok_or_else(|| format!("{label}: the tokenizer document is not a JSON object"))?;
        object.remove("chat_template")
    };
    let template = declared
        .as_ref()
        .and_then(serde_json::Value::as_str)
        .filter(|t| !t.trim().is_empty())
        .map(str::to_string);
    let tokenizer = if declared.is_some() {
        let stripped = serde_json::to_vec(&parsed).map_err(|e| format!("{label}: {e}"))?;
        tokenizers::Tokenizer::from_bytes(&stripped).map_err(|e| format!("{label}: {e}"))?
    } else {
        tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| format!("{label}: {e}"))?
    };
    Ok((tokenizer, template))
}

/// The byte source the blob loader used (BS-1): mmap where the platform and
/// mount permit it, `std::fs::read` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobLoadMethod {
    Mmap,
    Read,
}

impl BlobLoadMethod {
    /// The label the startup log carries (`load_method: "mmap"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mmap => "mmap",
            Self::Read => "read",
        }
    }
}

/// Test-only hatch (BS-1.3): forces the read fallback so both byte sources can
/// be compared on ONE fixture. Read by the loader; set only from
/// `tests/common/mod.rs` — there is no production story for a forced fallback.
pub const FORCE_READ_FALLBACK_ENV: &str = "QKB_TEST_FORCE_READ_FALLBACK";

/// The engine's serving precision, resolved from the artifact's DECLARED
/// remainder dtype (BS-2). Resolved once at load and stored on `ModelWeights`;
/// never inferred from a byte size, extension, or default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingDtype {
    Fp32,
    Bf16,
}

impl ServingDtype {
    /// The `GET /v1/models` `dtype` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }
}

impl From<RemainderDtype> for ServingDtype {
    fn from(declared: RemainderDtype) -> Self {
        match declared {
            RemainderDtype::Float32 => Self::Fp32,
            RemainderDtype::Bf16 => Self::Bf16,
        }
    }
}

/// The bytes a blob was read from, plus the access path that produced them.
/// One `Deref` into `[u8]` so the two sources are interchangeable downstream.
enum BlobBytes {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for BlobBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Owned(bytes) => bytes,
        }
    }
}

/// Load a qpk1 model artifact — a directory (legacy) or a single-file blob —
/// or fail with a diagnostic naming the file (qk-artifact-blob BL-1). The
/// returned handle carries the blob byte source as a VALUE (not only a log
/// line), so BS-1.1/BS-1.3 can be asserted without scraping stderr.
pub fn load(model_path: &Path) -> Result<LoadedModel, String> {
    let meta =
        std::fs::metadata(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?;
    let (dir, method) = if meta.is_dir() {
        (loader::load(model_path)?, None)
    } else if meta.is_file() {
        let (dir, method) = load_blob(model_path)?;
        (dir, Some(method))
    } else {
        return Err(format!(
            "model path {} is neither a directory nor a regular file",
            model_path.display()
        ));
    };
    compose(model_path, dir, method)
}

/// The blob load path (BS-1): validate the metadata prefix from the open file
/// (no map requested yet — a blob whose header/table breaks a rule, including
/// the 4096-alignment assert, is refused before any mmap call), then pick the
/// byte source, then run the payload checks on those bytes. The composition
/// itself is `quantibbler_qpk1::loader::build_from_blob` — one implementation
/// of the section rules, shared with the crate's own mmap path.
pub fn load_blob(artifact: &Path) -> Result<(loader::ModelDirectory, BlobLoadMethod), String> {
    let file = std::fs::File::open(artifact).map_err(|e| format!("{}: {e}", artifact.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", artifact.display()))?
        .len();
    if len == 0 {
        return Err(format!(
            "{}: an empty file is not a qpk1 blob",
            artifact.display()
        ));
    }
    let blob = {
        let mut reader = std::io::BufReader::new(&file);
        Blob::parse(&mut reader).map_err(|e| format!("{}: {e}", artifact.display()))?
    };
    let (bytes, method, fallback_reason) = blob_bytes(artifact, &file)?;
    blob.validate_payload(&bytes)
        .map_err(|e| format!("{}: {e}", artifact.display()))?;
    let dir = loader::build_from_blob(artifact, &blob, &bytes)?;
    match (method, fallback_reason) {
        (BlobLoadMethod::Mmap, _) => tracing::info!(
            artifact = %artifact.display(),
            load_method = BlobLoadMethod::Mmap.as_str(),
            "loaded blob via mmap"
        ),
        (BlobLoadMethod::Read, reason) => tracing::warn!(
            artifact = %artifact.display(),
            load_method = BlobLoadMethod::Read.as_str(),
            reason = reason.as_deref().unwrap_or("unspecified"),
            "blob load fell back to std::fs::read"
        ),
    }
    Ok((dir, method))
}

/// Choose the blob's byte source, returning the fallback reason when the read
/// path was taken (the WARN the spec requires). mmap failure is not fatal: the
/// loaded model is identical either way.
///
/// The read itself is NOT an `expect`: the artifact was opened above, but the
/// path can be swapped for a directory, made unreadable, or removed between
/// that open and this read, so a failure refuses loudly naming the artifact
/// (SR-1.2's diagnostic contract) instead of panicking without it.
fn blob_bytes(
    artifact: &Path,
    file: &std::fs::File,
) -> Result<(BlobBytes, BlobLoadMethod, Option<String>), String> {
    if std::env::var_os(FORCE_READ_FALLBACK_ENV).is_some() {
        let reason = format!("forced by {FORCE_READ_FALLBACK_ENV} (test hatch)");
        let read = read_whole(artifact)?;
        return Ok((BlobBytes::Owned(read), BlobLoadMethod::Read, Some(reason)));
    }
    // SAFETY: a read-only map of a file opened read-only; the map lives no
    // longer than `file`, and `validate_payload` bounds-checks every section
    // range against the map's own length before any slice is taken from it.
    match unsafe { memmap2::Mmap::map(file) } {
        Ok(map) => Ok((BlobBytes::Mapped(map), BlobLoadMethod::Mmap, None)),
        Err(e) => {
            let reason = format!("mmap failed: {e}");
            let read = read_whole(artifact)?;
            Ok((BlobBytes::Owned(read), BlobLoadMethod::Read, Some(reason)))
        }
    }
}

/// The fallback byte source's own fail-closed refusal: read the whole artifact,
/// naming it and the OS reason on failure.
fn read_whole(artifact: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(artifact).map_err(|e| format!("{}: {e}", artifact.display()))
}

/// The serving-side composition shared by both sources: tokenizer, the
/// `gpu:sage` startup gate, the weights, and the resolved serving dtype.
fn compose(
    model_path: &Path,
    dir: loader::ModelDirectory,
    method: Option<BlobLoadMethod>,
) -> Result<LoadedModel, String> {
    // Side files (the gate record) live beside the artifact: the model
    // directory for a directory source, the blob's own directory for a
    // single-file source.
    let artifact_dir = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(model_path)
    };

    // The tokenizer arrives differently per source: a blob CARRIES it as a
    // required section (design B9), a directory names it beside the model.
    // Either way the SAME document also carries the model's chat template
    // (BS-5), so both are taken out of it in one pass (see
    // `tokenizer_from_json` for why the key must leave the tokenizer's view).
    let (tokenizer, chat_template) = match &dir.tokenizer_json {
        Some(bytes) => tokenizer_from_json(
            bytes,
            &format!("{}: tokenizer.json section", model_path.display()),
        )?,
        None => {
            let tokenizer_path = model_path.join("tokenizer.json");
            let bytes = std::fs::read(&tokenizer_path)
                .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;
            tokenizer_from_json(&bytes, &tokenizer_path.display().to_string())?
        }
    };

    // The `gpu:sage` startup gate (qk-sage-attention 3.2): checked HERE, before
    // any device is touched and before the weights exist, so a model whose
    // record is missing/stale/out of bounds aborts startup with a typed refusal
    // instead of serving degraded attention (spec delta `qk-server`, "Gate fail
    // refuses the face"). `Ok(None)` = the opt-in does not ask for sage and the
    // ordinary face decision stands.
    let sage = sage_face_from_env(artifact_dir, &dir.model_id).map_err(|e| e.to_string())?;

    // BS-2: the DECLARATION selects the serving dtype. A directory never
    // selects Bf16 — native bf16 serving is reachable only through the blob
    // loader, because directory loading follows qk-artifact-dtype RD-1/RD-2
    // (a bf16-declared directory widens; BS-2.5). A blob follows the
    // `remainder_dtype` its parser already required and cross-checked against
    // every section tag.
    let serving_dtype = match method {
        Some(_) => ServingDtype::from(dir.remainder_dtype),
        None => ServingDtype::Fp32,
    };

    let mut weights = ModelWeights::from_layers(
        dir.layers,
        dir.embedding,
        dir.lm_head,
        dir.d_model,
        dir.vocab_size,
        dir.n_layers,
        dir.n_heads,
        dir.n_kv_heads,
        dir.head_dim,
        dir.rms_norm_eps,
        dir.rope_theta,
        dir.norm_slots.clone(),
        dir.norm_weights,
        serving_dtype,
    )
    .map_err(|e| format!("{}: {e}", model_path.display()))?;
    // The gate admitted the face: install the frozen decision over the ordinary
    // opt-in one (which for `=sage` is the exact scalar fallback root).
    if let Some(decision) = sage {
        weights.set_attention_face(decision);
    }
    Ok(LoadedModel {
        id: dir.model_id,
        weights: Arc::new(weights),
        tokenizer,
        eos_token_id: dir.eos_token_id,
        bos_token_id: dir.bos_token_id,
        vocab_size: dir.vocab_size,
        d_model: dir.d_model,
        blob_load_method: method,
        chat_template,
    })
}
