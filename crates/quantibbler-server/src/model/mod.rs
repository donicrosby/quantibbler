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
}

/// Load a qpk1 model artifact — a directory (legacy) or a single-file blob —
/// or fail with a diagnostic naming the file (qk-artifact-blob BL-1).
pub fn load(model_dir: &Path) -> Result<LoadedModel, String> {
    let dir = loader::ModelDirectory::load(model_dir)?;
    // Side files (the gate record) live beside the artifact: the model
    // directory for a directory source, the blob's own directory for a
    // single-file source.
    let artifact_dir = if model_dir.is_dir() {
        model_dir
    } else {
        model_dir.parent().unwrap_or(model_dir)
    };

    // The tokenizer arrives differently per source: a blob CARRIES it as a
    // required section (design B9), a directory names it beside the model.
    let tokenizer = match &dir.tokenizer_json {
        Some(bytes) => tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("{}: tokenizer.json section: {e}", model_dir.display()))?,
        None => {
            let tokenizer_path = model_dir.join("tokenizer.json");
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?
        }
    };

    // The `gpu:sage` startup gate (qk-sage-attention 3.2): checked HERE, before
    // any device is touched and before the weights exist, so a model whose
    // record is missing/stale/out of bounds aborts startup with a typed refusal
    // instead of serving degraded attention (spec delta `qk-server`, "Gate fail
    // refuses the face"). `Ok(None)` = the opt-in does not ask for sage and the
    // ordinary face decision stands.
    let sage = sage_face_from_env(artifact_dir, &dir.model_id).map_err(|e| e.to_string())?;

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
    )
    .map_err(|e| format!("{}: {e}", model_dir.display()))?;
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
    })
}
