//! qpk1 model-directory loader (qk-server design D3, hardened).
//!
//! Loads `<model-dir>` end to end: `manifest.toml`, per-layer qpk1 streams
//! via the fail-closed reader, `config.toml`, and the remainder sidecar
//! matrices (embedding / lm_head / norms). Every rule below is a hard
//! rejection — no partial model is ever returned.
//!
//! Hardening rules (added 2026-10-17 after independent review):
//!
//! 1. **Path containment.** Manifest-controlled relative paths
//!    (`[layer_files]`, `[sidecar]`) are validated lexically: empty,
//!    absolute, `..`-containing, and component-prefixed paths are
//!    rejected. `Path::join` with an absolute right side would otherwise
//!    discard the model dir entirely (arbitrary file read).
//! 2. **Stat-before-read.** Every manifest-controlled file is
//!    `symlink_metadata`-checked first: it must be a regular file (not a
//!    symlink — the model dir is read-only trusted content, and links are
//!    an escape hatch) and its length must fit the class cap (layers) or
//!    the exact expected size (sidecar) BEFORE any bytes are read. No
//!    unbounded read of `/dev/zero`-class specials, FIFOs, or huge files.
//! 3. **Checked arithmetic.** `vocab_size * d_model` and all byte-offset
//!    products are `checked_mul`-guarded, and both dims are capped, so a
//!    crafted config can neither wrap the expected-length equality nor
//!    drive multi-GB allocations at request time.
//! 4. **Fail-closed `weight_tying`.** The field is validated against the
//!    documented forms (`unset` or `"lm_head=<path>@<dtype>"`; the tied
//!    sentinel `"lm_head=e1@<dtype>"` reuses the embedding sidecar, an untied
//!    head names its own matrix); unknown values abort startup instead of
//!    being silently ignored.
//!
//! 5. **Declared remainder dtype.** The `[sidecar]` block declares the dtype
//!    its three files hold in an optional `dtype` key (`fp32` / `bf16`); the
//!    key's ABSENCE is the fp32 reading, which is the meaning every
//!    pre-existing artifact already has. The declaration alone selects the
//!    itemsize — a byte length is never used to infer a dtype. An unknown
//!    token, a `weight_tying` suffix that disagrees with the key, and a file
//!    whose length is not the declared element count times the declared
//!    itemsize are all hard rejections naming the offending declaration.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::blob::{Blob, BlobSection, BlobTag};
use crate::{Layer, RemainderDtype};

/// Hard cap on one qpk1 layer stream (bytes). A legitimate layer for a
/// 100B-class model at r=256 is tens of MB; 4 GiB leaves two orders of
/// magnitude of headroom while still refusing multi-GB pathological reads.
const MAX_LAYER_BYTES: u64 = 4 << 30;
/// Hard cap on one sidecar remainder matrix file (bytes). At vocab*d_model
/// for any v0.1-served model this is orders of magnitude above need; same
/// rationale as MAX_LAYER_BYTES.
const MAX_SIDECAR_BYTES: u64 = 4 << 30;
/// Cap on either model dimension (config.toml). Bounds the embedding
/// product far below usize wraparound and below allocation-dangerous
/// sizes; real models sit at ~1e5 vocab, ~1e4 d_model.
const MAX_DIM: usize = 1 << 26;
/// Cap on declared block count (config.toml). Bounds the norm-sidecar
/// product (per-block inventory floats * n_layers + d_model) through
/// `checked_mul`; real frontier models top out around two hundred blocks.
const MAX_LAYERS: usize = 1 << 12;

/// A fully loaded and validated model directory.
#[derive(Debug)]
pub struct ModelDirectory {
    /// The served model id (`manifest.toml`'s `model_id`).
    pub model_id: String,
    /// Layers in serving order.
    pub layers: Vec<(String, Layer)>,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub vocab_size: usize,
    pub d_model: usize,
    /// Validated attention geometry (qk-attention N-2).
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Declared norm inventory (config.toml `norm_slots`), in file order.
    pub norm_slots: Vec<NormSlot>,
    /// The dtype the remainder sidecars declared — informational: the loaded
    /// vectors below are always widened to f64, so a consumer never branches
    /// on it, but a reader of this struct can tell an fp32 artifact from bf16.
    pub remainder_dtype: RemainderDtype,
    /// Embedding, `vocab_size x d_model`, read at its declared dtype and
    /// widened to f64.
    pub embedding: Vec<f64>,
    /// lm_head when not weight-tied, same layout.
    pub lm_head: Option<Vec<f64>>,
    /// Norm sidecar, `norm_slots.total_floats() * n_layers + d_model`
    /// floats read at its declared dtype and widened to f64: per block, every
    /// declared slot's vector in inventory order; then the model-final norm. Layout is fully
    /// derived from the declared inventory — no architecture is
    /// hard-coded (qk-attention N-1, amended).
    pub norm_weights: Vec<f64>,
    /// The `tokenizer.json` bytes when the source was a BLOB (the section is
    /// required there, design B9); `None` for a directory source, where the
    /// server reads `tokenizer.json` beside the model.
    pub tokenizer_json: Option<Vec<u8>>,
    /// The canonical chat-template sidecar's text when the source was a BLOB
    /// that carried the section; `None` for a directory source (the server
    /// reads the sidecar beside the model) and for a blob without it, or
    /// whose section is blank after trim.
    pub chat_template_jinja: Option<String>,
}

/// One declared norm slot and the width of its vector per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormSlot {
    /// Transformer-block input RMSNorm: `d_model` floats.
    Input,
    /// Post-attention RMSNorm: `d_model` floats.
    PostAttention,
    /// Attention q-norm over `head_dim` floats (one shared vector,
    /// broadcast across heads).
    QNorm,
    /// Attention k-norm over `head_dim` floats.
    KNorm,
    /// Post-feedforward RMSNorm (Gemma-style sandwich): `d_model` floats.
    PostFeedforward,
}

impl NormSlot {
    /// Config.toml spelling; also the inventory's canonical order index.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::PostAttention => "post_attention",
            Self::QNorm => "q_norm",
            Self::KNorm => "k_norm",
            Self::PostFeedforward => "post_feedforward",
        }
    }

    /// Vector width for one block, given the model geometry.
    pub fn width(self, d_model: usize, head_dim: usize) -> usize {
        match self {
            Self::QNorm | Self::KNorm => head_dim,
            _ => d_model,
        }
    }
}

impl std::str::FromStr for NormSlot {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "input" => Ok(Self::Input),
            "post_attention" => Ok(Self::PostAttention),
            "q_norm" => Ok(Self::QNorm),
            "k_norm" => Ok(Self::KNorm),
            "post_feedforward" => Ok(Self::PostFeedforward),
            other => Err(format!(
                "unknown norm slot {other:?}; known: input, post_attention, \
                 q_norm, k_norm, post_feedforward"
            )),
        }
    }
}

/// Per-block float count of a declared inventory.
pub fn inventory_block_floats(slots: &[NormSlot], d_model: usize, head_dim: usize) -> usize {
    slots.iter().map(|s| s.width(d_model, head_dim)).sum()
}

/// Load a qpk1 model directory, or fail with a diagnostic naming the file.
pub fn load(model_dir: &Path) -> Result<ModelDirectory, String> {
    if !model_dir.is_dir() {
        return Err(format!(
            "model dir {} does not exist or is not a directory",
            model_dir.display()
        ));
    }

    let manifest_path = model_dir.join("manifest.toml");
    let manifest_raw = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: Manifest =
        toml::from_str(&manifest_raw).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    if manifest.layers.is_empty() {
        return Err(format!(
            "{}: manifest lists no layers; nothing to serve",
            manifest_path.display()
        ));
    }
    // Every layer name must have exactly one manifest-controlled path.
    for name in &manifest.layers {
        if !manifest.layer_files.contains_key(name) {
            return Err(format!(
                "{}: layer {name:?} has no entry in [layer_files]",
                manifest_path.display()
            ));
        }
    }

    let config_path = model_dir.join("config.toml");
    let config_raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
    let config: ModelConfig =
        toml::from_str(&config_raw).map_err(|e| format!("{}: {e}", config_path.display()))?;
    config.validate(&config_path)?;

    // Layers: containment check, stat-before-read, fail-closed parse, rank
    // gate — in that order, so an escaped or oversized path never reaches
    // the reader.
    let mut layers: Vec<(String, Layer)> = Vec::new();
    for name in &manifest.layers {
        let rel = &manifest.layer_files[name];
        let layer_path = contained_join(model_dir, rel).map_err(|why| {
            format!(
                "{}: layer {name:?} file path {rel:?} rejected: {why}",
                manifest_path.display()
            )
        })?;
        let raw = read_regular_capped(&layer_path, MAX_LAYER_BYTES)?;
        let layer = crate::parse(&raw).map_err(|e| format!("{}: {e}", layer_path.display()))?;
        if layer.r > layer.dout.min(layer.din) {
            return Err(format!(
                "{}: rank r={} exceeds min(dout, din)={}",
                layer_path.display(),
                layer.r,
                layer.dout.min(layer.din)
            ));
        }
        layers.push((name.clone(), layer));
    }

    // Sidecars: same containment discipline on the manifest-controlled
    // paths, plus the dtype the [sidecar] block DECLARES; the matrices are
    // exact-size by construction, so the stat length must equal the declared
    // element count * the declared itemsize before any read.
    let sidecar = manifest.sidecar.ok_or_else(|| {
        format!(
            "{}: missing [sidecar] block (embedding/lm_head/norms weights)",
            manifest_path.display()
        )
    })?;
    let remainder_dtype = sidecar.declared_dtype(&manifest_path)?;
    // `weight_tying`'s own suffix is a SECOND declaration of the same bytes.
    // A remainder set cannot hold two widths, and honouring either one would
    // make the other a lie in the shipped manifest — so disagreement is a
    // refusal, not a precedence rule.
    if let Some(tying_dtype) = config.tying_dtype(&config_path)? {
        if tying_dtype != remainder_dtype {
            return Err(format!(
                "{}: weight_tying declares @{} but the manifest [sidecar] block declares \
                 dtype = \"{}\"; a remainder set cannot hold two dtypes",
                config_path.display(),
                tying_dtype.as_str(),
                remainder_dtype.as_str()
            ));
        }
    }
    let expected_floats = config
        .vocab_size
        .checked_mul(config.d_model)
        .ok_or_else(|| format!("{}: vocab_size * d_model overflows", config_path.display()))?;
    let emb_path = resolve_sidecar(model_dir, &manifest_path, "embedding", &sidecar.embedding)?;
    let head_path = resolve_sidecar(model_dir, &manifest_path, "lm_head", &sidecar.lm_head)?;
    let embedding =
        read_remainder_exact(&emb_path, expected_floats, remainder_dtype, &config_path)?;
    let lm_head = if head_path == emb_path {
        None // weight-tied
    } else {
        Some(read_remainder_exact(
            &head_path,
            expected_floats,
            remainder_dtype,
            &config_path,
        )?)
    };

    // Norm sidecar (qk-attention N-1, amended): per block the input
    // layernorm + post-attention layernorm (d_model each) and the
    // attention q/k norms (head_dim each, one shared vector per
    // projection), plus the model-final norm.
    let norms_path = resolve_sidecar(model_dir, &manifest_path, "norms", &sidecar.norms)?;
    let slots: Vec<NormSlot> = config
        .norm_slots
        .iter()
        .map(|s| s.parse::<NormSlot>().expect("validated inventory"))
        .collect();
    let per_block = inventory_block_floats(&slots, config.d_model, config.head_dim);
    let norm_floats = per_block
        .checked_mul(config.n_layers)
        .and_then(|p| p.checked_add(config.d_model))
        .ok_or_else(|| {
            format!(
                "{}: norm sidecar count overflows (inventory {:?})",
                config_path.display(),
                config.norm_slots
            )
        })?;
    let norm_weights =
        read_remainder_exact(&norms_path, norm_floats, remainder_dtype, &config_path)?;

    Ok(ModelDirectory {
        model_id: manifest.model_id,
        layers,
        eos_token_id: config.eos_token_id,
        bos_token_id: config.bos_token_id,
        vocab_size: config.vocab_size,
        d_model: config.d_model,
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        norm_slots: slots,
        remainder_dtype,
        embedding,
        lm_head,
        norm_weights,
        tokenizer_json: None,
        chat_template_jinja: None,
    })
}

impl ModelDirectory {
    /// Load a model from EITHER source (BL-1, design B7): a directory path runs
    /// the legacy directory loader, a regular file is detected by its `QKPB`
    /// magic and runs the blob loader, and a regular file without the magic is
    /// refused — never a directory attempt, never a partial parse.
    pub fn load(path: &Path) -> Result<ModelDirectory, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.is_dir() {
            return load(path);
        }
        if !meta.is_file() {
            return Err(format!(
                "model path {} is neither a directory nor a regular file",
                path.display()
            ));
        }
        if detect_blob_magic(path)? {
            load_blob(path)
        } else {
            Err(format!(
                "model file {} is not a qpk1 blob: its first four bytes are not {:?}. A regular \
                 file is only ever read as a blob; the legacy directory format is read from a \
                 directory path",
                path.display(),
                crate::BLOB_MAGIC
            ))
        }
    }
}

/// Does the file begin with the blob magic? A file shorter than four bytes
/// answers `false` (it cannot be a blob).
fn detect_blob_magic(path: &Path) -> Result<bool, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(magic == crate::BLOB_MAGIC),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Load a single-file blob in two phases (BL-4.2 ordering): first read and
/// validate the metadata prefix from the open file with pread — magic, header,
/// header padding, and section table, including the 4096-alignment assert —
/// and only after that passes map the file (BL-4) and run the payload checks
/// on the map. A misaligned blob is refused by name before any mmap call.
/// Every refusal names the file.
pub fn load_blob(path: &Path) -> Result<ModelDirectory, String> {
    load_blob_with(path, |file| unsafe { memmap2::Mmap::map(file) })
}

/// The body of [`load_blob`] with an injectable mapper. The mapper is called
/// only after the metadata prefix passes every rule — in particular BL-4.2's
/// 4096-alignment assert — so a misaligned blob refuses with zero map calls.
/// Tests inject a probe that counts calls to prove the ordering.
fn load_blob_with<F>(path: &Path, map: F) -> Result<ModelDirectory, String>
where
    F: FnOnce(&std::fs::File) -> std::io::Result<memmap2::Mmap>,
{
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if len == 0 {
        return Err(format!(
            "{}: an empty file is not a qpk1 blob",
            path.display()
        ));
    }
    // Phase 1: read and validate the metadata prefix from the file. This is a
    // pread/read of the magic, header JSON, header padding, and section table;
    // no mmap has been requested yet. BL-4.2's alignment assert runs here.
    let mut reader = std::io::BufReader::new(&file);
    let blob = Blob::parse(&mut reader).map_err(|e| format!("{}: {e}", path.display()))?;
    drop(reader);
    // Phase 2: metadata passed; map the file and run the payload checks on the
    // map. A misaligned blob never reaches this line.
    // SAFETY: the map is read-only and lives no longer than `file`. The blob
    // is read-only trusted content, and every section access below is bounds
    // checked against the map's own length by `Blob::validate_payload`'s range
    // validation, so a concurrent truncation cannot produce an out-of-range
    // slice through the parsed offsets.
    let map = map(&file).map_err(|e| format!("{}: mmap failed: {e}", path.display()))?;
    let bytes: &[u8] = &map;
    blob.validate_payload(bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    build_from_blob(path, &blob, bytes)
}

/// Every section-against-header validation of BL-2/BL-3, then the
/// `ModelDirectory` composition. Public so a caller that owns its own byte
/// source (the server's mmap-with-read-fallback loader) reuses this single
/// validation/composition instead of re-deriving it: `blob` must be the
/// result of `Blob::parse` on `bytes`, and `bytes` must already have passed
/// `Blob::validate_payload` — the caller composes the phases, this fn owns
/// the rules.
pub fn build_from_blob(path: &Path, blob: &Blob, bytes: &[u8]) -> Result<ModelDirectory, String> {
    let header = &blob.header;
    // The header's geometry is validated by the SAME rules as config.toml
    // (qk-attention N-2), so a blob cannot declare a shape a directory could
    // not.
    let config = ModelConfig {
        vocab_size: header.vocab_size,
        d_model: header.d_model,
        eos_token_id: header.eos_token_id,
        bos_token_id: header.bos_token_id,
        n_layers: header.n_layers,
        n_heads: header.n_heads,
        n_kv_heads: header.n_kv_heads,
        head_dim: header.head_dim,
        rms_norm_eps: header.rms_norm_eps,
        rope_theta: header.rope_theta,
        norm_slots: header.norm_slots.clone(),
        weight_tying: header.weight_tying.clone(),
    };
    config.validate(path)?;

    let remainder_dtype = header.remainder_dtype;
    // `weight_tying`'s suffix is the SECOND declaration of the same bytes
    // (BL-3.1/BL-3.2); disagreement names both.
    if let Some(declared) = header.tying_dtype() {
        let tying = declared.map_err(|e| format!("{}: {e}", path.display()))?;
        if tying != remainder_dtype {
            return Err(format!(
                "{}: weight_tying declares @{} but the header declares remainder_dtype \
                 \"{}\"; a remainder set cannot hold two dtypes",
                path.display(),
                tying.as_str(),
                remainder_dtype.as_str()
            ));
        }
    }

    // Layers: required section, stream tag agreement, fail-closed parse, rank.
    let mut layers: Vec<(String, Layer)> = Vec::new();
    for name in &header.layers {
        let section = blob.section(name).ok_or_else(|| {
            format!(
                "{}: blob is missing the required layer section {name:?}",
                path.display()
            )
        })?;
        if section.tag.stream_version().is_none() {
            return Err(format!(
                "{}: layer section {name:?} is tagged {}; a layer section must carry a qpk1 \
                 stream (tag 0 qpk1_v1 or 1 qpk1_v2)",
                path.display(),
                section.tag.as_str()
            ));
        }
        let raw = blob.slice(bytes, section);
        let layer =
            crate::parse(raw).map_err(|e| format!("{}: section {name:?}: {e}", path.display()))?;
        if section.tag.stream_version() != Some(layer.version) {
            return Err(format!(
                "{}: layer section {name:?} is tagged {} but its stream header declares version {}",
                path.display(),
                section.tag.as_str(),
                layer.version
            ));
        }
        if layer.r > layer.dout.min(layer.din) {
            return Err(format!(
                "{}: section {name:?}: rank r={} exceeds min(dout, din)={}",
                path.display(),
                layer.r,
                layer.dout.min(layer.din)
            ));
        }
        layers.push((name.clone(), layer));
    }

    let embedding_floats = config
        .vocab_size
        .checked_mul(config.d_model)
        .ok_or_else(|| format!("{}: vocab_size * d_model overflows", path.display()))?;
    let embedding_section = blob.section("embedding").ok_or_else(|| {
        format!(
            "{}: blob is missing the required section \"embedding\"",
            path.display()
        )
    })?;
    let embedding = read_remainder_section(
        path,
        blob,
        bytes,
        embedding_section,
        embedding_floats,
        remainder_dtype,
    )?;
    // The head is keyed on the SECTION's presence (design B2): present means
    // an untied head with its own bytes; absent means the head IS the
    // embedding. A header that declares an untied head must ship the section.
    let lm_head = match blob.section("lm_head") {
        Some(section) => Some(read_remainder_section(
            path,
            blob,
            bytes,
            section,
            embedding_floats,
            remainder_dtype,
        )?),
        None if header.head_is_untied() => {
            return Err(format!(
                "{}: blob is missing the required section \"lm_head\" (the header declares an \
                 untied head, so the head is not the embedding)",
                path.display()
            ));
        }
        None => None,
    };

    let slots: Vec<NormSlot> = config
        .norm_slots
        .iter()
        .map(|s| s.parse::<NormSlot>().expect("validated inventory"))
        .collect();
    let per_block = inventory_block_floats(&slots, config.d_model, config.head_dim);
    let norm_floats = per_block
        .checked_mul(config.n_layers)
        .and_then(|p| p.checked_add(config.d_model))
        .ok_or_else(|| {
            format!(
                "{}: norm section count overflows (inventory {:?})",
                path.display(),
                config.norm_slots
            )
        })?;
    let norms_section = blob.section("norms").ok_or_else(|| {
        format!(
            "{}: blob is missing the required section \"norms\"",
            path.display()
        )
    })?;
    let norm_weights = read_remainder_section(
        path,
        blob,
        bytes,
        norms_section,
        norm_floats,
        remainder_dtype,
    )?;

    // tokenizer.json is required (BL-2.2/B9): the server tokenizes immediately
    // after load, so a blob without it would die at first tokenize.
    let tokenizer_section = blob.section("tokenizer.json").ok_or_else(|| {
        format!(
            "{}: blob is missing the required section \"tokenizer.json\"",
            path.display()
        )
    })?;
    if tokenizer_section.tag != BlobTag::Raw {
        return Err(format!(
            "{}: section \"tokenizer.json\" is tagged {}; tokenizer sections are raw (tag 4)",
            path.display(),
            tokenizer_section.tag.as_str()
        ));
    }
    let tokenizer_json = Some(blob.slice(bytes, tokenizer_section).to_vec());

    // The canonical chat-template sidecar, carried as its own raw section
    // (a model's template is a text file, not a `tokenizer.*` blob entry).
    let chat_template_jinja = read_chat_template_section(path, blob, bytes)?;

    Ok(ModelDirectory {
        model_id: header.model_id.clone(),
        layers,
        eos_token_id: config.eos_token_id,
        bos_token_id: config.bos_token_id,
        vocab_size: config.vocab_size,
        d_model: config.d_model,
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        norm_slots: slots,
        remainder_dtype,
        embedding,
        lm_head,
        norm_weights,
        tokenizer_json,
        chat_template_jinja,
    })
}

/// The optional canonical chat-template sidecar: a raw section decoded as
/// UTF-8. Absent, or blank after trim, reads as "the artifact declares no
/// template" (`None`); a section tagged anything but raw, or payload that is
/// not UTF-8, is refused naming the section.
fn read_chat_template_section(
    path: &Path,
    blob: &Blob,
    bytes: &[u8],
) -> Result<Option<String>, String> {
    let Some(section) = blob.section(crate::CHAT_TEMPLATE_FILE) else {
        return Ok(None);
    };
    if section.tag != BlobTag::Raw {
        return Err(format!(
            "{}: section {:?} is tagged {}; the chat-template sidecar is raw (tag 4)",
            path.display(),
            section.name,
            section.tag.as_str()
        ));
    }
    let raw = blob.slice(bytes, section);
    let text = std::str::from_utf8(raw).map_err(|e| {
        format!(
            "{}: section {:?} is not valid UTF-8: {e}",
            path.display(),
            section.name
        )
    })?;
    Ok((!text.trim().is_empty()).then(|| text.to_string()))
}

/// One remainder section: the tag must equal the header's `remainder_dtype`
/// (BL-3.3), the length must equal the inventory-derived float count times the
/// declared itemsize (BL-2.3), and the payload widens to f64.
fn read_remainder_section(
    path: &Path,
    blob: &Blob,
    bytes: &[u8],
    section: &BlobSection,
    floats: usize,
    dtype: RemainderDtype,
) -> Result<Vec<f64>, String> {
    let tag_dtype = section.tag.remainder_dtype().ok_or_else(|| {
        format!(
            "{}: remainder section {:?} is tagged {}; a remainder section must be fp32_le (tag 2) \
             or bf16_le (tag 3)",
            path.display(),
            section.name,
            section.tag.as_str()
        )
    })?;
    if tag_dtype != dtype {
        return Err(format!(
            "{}: remainder section {:?} is tagged {} but the header declares remainder_dtype \
             \"{}\"",
            path.display(),
            section.name,
            section.tag.as_str(),
            dtype.as_str()
        ));
    }
    let expected_bytes = floats.checked_mul(dtype.itemsize()).ok_or_else(|| {
        format!(
            "{}: remainder section {:?} byte size overflows ({floats} elements @{})",
            path.display(),
            section.name,
            dtype.as_str()
        )
    })?;
    let raw = blob.slice(bytes, section);
    if raw.len() != expected_bytes {
        return Err(format!(
            "{}: remainder section {:?} is {} bytes; expected exactly {expected_bytes} bytes (a \
             {floats}-element remainder declared @{})",
            path.display(),
            section.name,
            raw.len(),
            dtype.as_str()
        ));
    }
    Ok(raw
        .chunks_exact(dtype.itemsize())
        .map(|p| dtype.widen_le(p))
        .collect())
}

/// Lexical containment check + join: `rel` must be a non-empty relative
/// path with no `..` components, no root/prefix, and (on Windows) no
/// drive-prefix trickery. The join can therefore never escape `base`,
/// regardless of `Path::join`'s absolute-path-replacement semantics.
fn contained_join(base: &Path, rel: &str) -> Result<PathBuf, String> {
    if rel.is_empty() {
        return Err("path is empty".into());
    }
    let p = Path::new(rel);
    let mut depth: usize = 0;
    for comp in p.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => {
                return Err("absolute paths are not allowed".into());
            }
            Component::ParentDir => {
                return Err("'..' components are not allowed".into());
            }
            Component::CurDir => { /* skip explicit no-ops */ }
            Component::Normal(_) => depth += 1,
        }
    }
    if depth == 0 {
        return Err("path names no file".into());
    }
    Ok(base.join(p))
}

/// `symlink_metadata` gate: regular file, size within `cap`. Returns the
/// stat length. Never follows symlinks and never reads the file.
fn stat_regular_capped(path: &Path, cap: u64, what: &str) -> Result<u64, String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!(
            "{}: not a regular file (symlinks and specials are rejected); {what} must be a plain file inside the model directory",
            path.display()
        ));
    }
    let len = meta.len();
    if len > cap {
        return Err(format!(
            "{}: {len} bytes exceeds the {what} cap of {cap} bytes",
            path.display()
        ));
    }
    Ok(len)
}

/// Read a manifest-controlled file after the regular-file + cap gate.
fn read_regular_capped(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    stat_regular_capped(path, cap, "layer stream")?;
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Resolve one `[sidecar]` entry to a path inside the model directory. The
/// entry is a plain relative path: the dtype is declared ONCE for the whole
/// remainder set (the `[sidecar] dtype` key), never spelled per file, so this
/// is the containment discipline alone and a rejection names the manifest key.
fn resolve_sidecar(
    model_dir: &Path,
    manifest_path: &Path,
    key: &str,
    raw: &str,
) -> Result<PathBuf, String> {
    contained_join(model_dir, raw).map_err(|why| {
        format!(
            "{}: [sidecar] {key} = {raw:?} rejected: {why}",
            manifest_path.display()
        )
    })
}

/// Read a remainder sidecar of EXACTLY `floats * dtype.itemsize()` bytes at
/// the DECLARED dtype: the stat length must match before the read (the read is
/// then exact by stat). A length that contradicts the declaration is refused —
/// a file is never re-interpreted at another dtype to make it fit.
fn read_remainder_exact(
    path: &Path,
    floats: usize,
    dtype: RemainderDtype,
    config_path: &Path,
) -> Result<Vec<f64>, String> {
    let expected_bytes = floats.checked_mul(dtype.itemsize()).ok_or_else(|| {
        format!(
            "{}: remainder byte size overflows ({floats} elements at {} bytes declared @{})",
            config_path.display(),
            dtype.itemsize(),
            dtype.as_str()
        )
    })?;
    let len = stat_regular_capped(path, MAX_SIDECAR_BYTES, "sidecar matrix")?;
    if len != expected_bytes as u64 {
        return Err(format!(
            "{}: {len} bytes; expected exactly {expected_bytes} bytes (a {floats}-element \
             remainder declared @{})",
            path.display(),
            dtype.as_str()
        ));
    }
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(raw
        .chunks_exact(dtype.itemsize())
        .map(|p| dtype.widen_le(p))
        .collect())
}

#[derive(Debug, Deserialize)]
struct Manifest {
    model_id: String,
    /// Serving order of layer names.
    layers: Vec<String>,
    #[serde(rename = "layer_files")]
    layer_files: HashMap<String, String>,
    #[serde(default)]
    sidecar: Option<Sidecar>,
}

#[derive(Debug, Deserialize)]
struct Sidecar {
    /// Declared storage dtype of the three remainder files below: `fp32` or
    /// `bf16`. ABSENT is the fp32 reading — the byte contract every artifact
    /// written before this key existed already has — so a pre-change manifest
    /// keeps its exact meaning. A present-but-unknown token fails the load: a
    /// reader never guesses a width from anything but this declaration.
    #[serde(default)]
    dtype: Option<String>,
    /// Relative path of the embedding matrix, `vocab_size x d_model`,
    /// row-major, LE at the declared dtype.
    embedding: String,
    /// Same layout for the lm_head matrix; when its resolved path equals
    /// `embedding`'s the head is weight-tied.
    lm_head: String,
    /// The norm-weights sidecar (qk-attention N-1): inventory-derived floats,
    /// row-major, LE at the declared dtype.
    norms: String,
}

impl Sidecar {
    /// The remainder dtype this block declares, or the fp32 legacy reading.
    fn declared_dtype(&self, manifest_path: &Path) -> Result<RemainderDtype, String> {
        match self.dtype.as_deref() {
            None => Ok(RemainderDtype::Float32),
            Some(token) => RemainderDtype::from_token(token).ok_or_else(|| {
                format!(
                    "{}: [sidecar] dtype = {token:?} is not a declared remainder dtype \
                     (fp32 or bf16); an absent key reads as fp32",
                    manifest_path.display()
                )
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    vocab_size: usize,
    d_model: usize,
    eos_token_id: u32,
    bos_token_id: u32,
    /// Attention geometry (qk-attention N-2): all required, all validated.
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    /// Declared per-block norm inventory (qk-attention N-1, amended): the
    /// architecture states which norm vectors exist, in file order. The
    /// sidecar byte length is derived from this — no architecture is
    /// hard-coded here.
    norm_slots: Vec<String>,
    /// Documented forms (qk-server C5): unset, or `"lm_head=<path>@<dtype>"`
    /// with a declared dtype (weight-tied: `"lm_head=e1@fp32"`). Anything
    /// else — including an undeclared dtype token — is a config error.
    #[serde(default)]
    weight_tying: Option<String>,
}

impl ModelConfig {
    fn validate(&self, config_path: &Path) -> Result<(), String> {
        if self.vocab_size == 0 || self.vocab_size > MAX_DIM {
            return Err(format!(
                "{}: vocab_size {} outside 1..={MAX_DIM}",
                config_path.display(),
                self.vocab_size
            ));
        }
        if self.d_model == 0 || self.d_model > MAX_DIM {
            return Err(format!(
                "{}: d_model {} outside 1..={MAX_DIM}",
                config_path.display(),
                self.d_model
            ));
        }
        // The token ids name embedding rows; anything at or past vocab_size
        // would be unrepresentable in the model itself.
        if u64::from(self.eos_token_id) >= self.vocab_size as u64 {
            return Err(format!(
                "{}: eos_token_id {} >= vocab_size {}",
                config_path.display(),
                self.eos_token_id,
                self.vocab_size
            ));
        }
        if u64::from(self.bos_token_id) >= self.vocab_size as u64 {
            return Err(format!(
                "{}: bos_token_id {} >= vocab_size {}",
                config_path.display(),
                self.bos_token_id,
                self.vocab_size
            ));
        }
        // Attention geometry (qk-attention N-2): every field required
        // (missing fields fail the TOML deserialize), every relationship
        // validated before anything is loaded.
        if self.n_layers == 0 || self.n_layers > MAX_LAYERS {
            return Err(format!(
                "{}: n_layers {} outside 1..={MAX_LAYERS}",
                config_path.display(),
                self.n_layers
            ));
        }
        if self.n_heads == 0 || self.n_heads > MAX_DIM {
            return Err(format!(
                "{}: n_heads {} outside 1..={MAX_DIM}",
                config_path.display(),
                self.n_heads
            ));
        }
        if self.n_kv_heads == 0 {
            return Err(format!(
                "{}: n_kv_heads must be at least 1 (got 0)",
                config_path.display()
            ));
        }
        if self.head_dim == 0 {
            return Err(format!(
                "{}: head_dim must be at least 1 (got 0)",
                config_path.display()
            ));
        }
        // GQA consistency: KV projection width must equal the residual
        // stream, and every query head must share exactly one group.
        let kv_width = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| format!("{}: n_kv_heads * head_dim overflows", config_path.display()))?;
        if kv_width != self.d_model {
            return Err(format!(
                "{}: n_kv_heads * head_dim = {kv_width} != d_model {} (GQA width mismatch)",
                config_path.display(),
                self.d_model
            ));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(format!(
                "{}: n_heads {} is not a multiple of n_kv_heads {} (GQA group mismatch)",
                config_path.display(),
                self.n_heads,
                self.n_kv_heads
            ));
        }
        if !(self.rms_norm_eps > 0.0 && self.rms_norm_eps.is_finite()) {
            return Err(format!(
                "{}: rms_norm_eps must be finite and > 0 (got {})",
                config_path.display(),
                self.rms_norm_eps
            ));
        }
        if !(self.rope_theta > 0.0 && self.rope_theta.is_finite()) {
            return Err(format!(
                "{}: rope_theta must be finite and > 0 (got {})",
                config_path.display(),
                self.rope_theta
            ));
        }
        // Norm inventory (N-1): non-empty, every name known, no repeats.
        // The ENGINE decides which slots its forward consumes; the loader
        // only guarantees the file matches the declaration.
        if self.norm_slots.is_empty() {
            return Err(format!(
                "{}: norm_slots must declare at least one slot",
                config_path.display()
            ));
        }
        for s in &self.norm_slots {
            if let Err(why) = s.parse::<NormSlot>() {
                return Err(format!("{}: {why}", config_path.display()));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for s in &self.norm_slots {
            if !seen.insert(s.as_str()) {
                return Err(format!(
                    "{}: duplicate norm slot {s:?}",
                    config_path.display()
                ));
            }
        }
        match self.weight_tying.as_deref() {
            None => {}
            Some(wt) => {
                // Documented forms: "lm_head=<relative-path>@<dtype>". The
                // tied sentinel "lm_head=e1@<dtype>" reuses the embedding
                // sidecar; any other path names its own head matrix (resolved
                // from the manifest's [sidecar] lm_head, which declares the
                // dtype the bytes actually hold). A dtype token is REQUIRED
                // here — every shipped artifact writes one — and an undeclared
                // token is named rather than defaulted to fp32.
                let declared = wt.strip_prefix("lm_head=").and_then(|rest| {
                    rest.rsplit_once('@').and_then(|(path, token)| {
                        (!path.is_empty() && RemainderDtype::from_token(token).is_some())
                            .then_some((path, token))
                    })
                });
                if declared.is_none() {
                    return Err(format!(
                        "{}: unknown weight_tying {wt:?}; documented forms: \
                         unset or \"lm_head=<path>@<dtype>\" with dtype fp32 or bf16 \
                         (tied: \"lm_head=e1@fp32\")",
                        config_path.display()
                    ));
                }
            }
        }
        Ok(())
    }

    /// The dtype `weight_tying` declares in its own suffix, or `None` when the
    /// field is unset. Its suffix is a SECOND declaration of the same bytes as
    /// the manifest's `[sidecar] dtype`, so `load` compares the two.
    fn tying_dtype(&self, config_path: &Path) -> Result<Option<RemainderDtype>, String> {
        let Some(wt) = self.weight_tying.as_deref() else {
            return Ok(None);
        };
        let token = wt
            .strip_prefix("lm_head=")
            .and_then(|rest| rest.rsplit_once('@'))
            .map(|(_, token)| token)
            .ok_or_else(|| {
                format!(
                    "{}: unknown weight_tying {wt:?}; documented forms: unset or \
                     \"lm_head=<path>@<dtype>\" with dtype fp32 or bf16",
                    config_path.display()
                )
            })?;
        RemainderDtype::from_token(token).map(Some).ok_or_else(|| {
            format!(
                "{}: weight_tying {wt:?} declares an undeclared dtype token {token:?}; \
                     declared dtypes are fp32 and bf16",
                config_path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/server-model")
    }

    fn copy_rec(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).expect("mkdir");
        for entry in std::fs::read_dir(src).expect("read_dir") {
            let entry = entry.expect("dir entry");
            let to = dst.join(entry.file_name());
            if entry.path().is_dir() {
                copy_rec(&entry.path(), &to);
            } else {
                std::fs::copy(entry.path(), &to).expect("copy file");
            }
        }
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "qpk1-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).expect("mkdir");
        path
    }

    /// The committed bf16-remainder twin (produced by `tools/repack_remainder.py`).
    fn bf16_fixture_dir() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/server-model-bf16")
    }

    /// bf16 round-to-nearest-even of an f32 word — the encoding the Python
    /// writer emits (a rounding bias into the low 16 bits before truncating;
    /// the LSB term makes exact ties land even). The loader only ever WIDENS a
    /// declared bf16 word; the tests use this to BUILD the bytes a
    /// declared-bf16 directory holds and to derive the exactly-widened values.
    fn bf16_word(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
    }

    fn bf16_bytes(floats: &[f32]) -> Vec<u8> {
        floats
            .iter()
            .flat_map(|v| bf16_word(*v).to_le_bytes())
            .collect()
    }

    fn f32_floats(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .expect("sidecar bytes")
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect()
    }

    /// The exactly-widened values a bf16-declared sidecar must yield: bf16 is
    /// the high half of the f32 word, so bf16 -> f32 -> f64 invents no bit.
    fn widened_f32_floats(path: &Path) -> Vec<f64> {
        f32_floats(path)
            .iter()
            .map(|v| f64::from(f32::from_bits(u32::from(bf16_word(*v)) << 16)))
            .collect()
    }

    fn rewrite(path: &Path, from: &str, to: &str) {
        let text = std::fs::read_to_string(path).expect("read");
        let swapped = text.replace(from, to);
        assert_ne!(text, swapped, "{from:?} must exist to be swapped");
        std::fs::write(path, swapped).expect("write");
    }

    /// A copy of the committed fixture whose remainder sidecars are re-encoded
    /// at bf16 and DECLARED through the `[sidecar] dtype` key, with
    /// `weight_tying`'s suffix agreeing — the shape a bf16 export produces.
    fn bf16_twin(tag: &str) -> PathBuf {
        let root = tmp_dir(tag);
        copy_rec(&dir(), &root);
        for (src_name, out_name) in [
            ("e1.f32.bin", "e1.bf16.bin"),
            ("norms.f32.bin", "norms.bf16.bin"),
        ] {
            let bytes = bf16_bytes(&f32_floats(&root.join(src_name)));
            std::fs::write(root.join(out_name), bytes).expect("write bf16 sidecar");
            std::fs::remove_file(root.join(src_name)).expect("drop the fp32 sidecar");
        }
        let manifest = root.join("manifest.toml");
        rewrite(&manifest, "[sidecar]", "[sidecar]\ndtype = \"bf16\"");
        rewrite(
            &manifest,
            "embedding = \"e1.f32.bin\"",
            "embedding = \"e1.bf16.bin\"",
        );
        rewrite(
            &manifest,
            "lm_head = \"e1.f32.bin\"",
            "lm_head = \"e1.bf16.bin\"",
        );
        rewrite(
            &manifest,
            "norms = \"norms.f32.bin\"",
            "norms = \"norms.bf16.bin\"",
        );
        rewrite(&root.join("config.toml"), "@fp32", "@bf16");
        root
    }

    // --- RD-1: declaration, agreement, refusals ----------------------------

    #[test]
    fn loads_declared_bf16_sidecar_rd1_1() {
        // `[sidecar] dtype = "bf16"` and a matching `@bf16` weight_tying
        // suffix: the embedding and norms resolve bf16 (the tied head shares
        // the embedding file).
        let root = bf16_twin("rd1-1");
        let m = load(&root).expect("a declared bf16 directory loads");
        assert_eq!(m.remainder_dtype, RemainderDtype::Bf16);
        assert_eq!(m.embedding, widened_f32_floats(&dir().join("e1.f32.bin")));
        assert_eq!(
            m.norm_weights,
            widened_f32_floats(&dir().join("norms.f32.bin"))
        );
        assert!(m.lm_head.is_none(), "the fixture's head is weight-tied");
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn absent_dtype_reads_fp32_rd1_2() {
        // A pre-change-shaped manifest (no `dtype` key) reads fp32 exactly:
        // the values are the raw LE f32s, not a widened reading.
        let m = load(&dir()).expect("the legacy manifest loads");
        assert_eq!(m.remainder_dtype, RemainderDtype::Float32);
        let want: Vec<f64> = std::fs::read(dir().join("e1.f32.bin"))
            .expect("embedding bytes")
            .chunks_exact(4)
            .map(|p| f64::from(f32::from_le_bytes([p[0], p[1], p[2], p[3]])))
            .collect();
        assert_eq!(m.embedding, want);
    }

    #[test]
    fn rejects_suffix_field_mismatch_rd1_3() {
        // Declared bf16, weight_tying says @fp32: two declarations of one byte
        // set that cannot both be true.
        let root = bf16_twin("rd1-3-bf16-field");
        rewrite(&root.join("config.toml"), "@bf16", "@fp32");
        let err = load(&root).expect_err("suffix/key mismatch must refuse");
        assert!(
            err.contains("bf16") && err.contains("fp32"),
            "the refusal must name both declarations: {err}"
        );
        std::fs::remove_dir_all(&root).expect("cleanup");

        // The other direction: declared fp32, weight_tying says @bf16.
        let root = bf16_twin("rd1-3-fp32-field");
        rewrite(
            &root.join("manifest.toml"),
            "dtype = \"bf16\"",
            "dtype = \"fp32\"",
        );
        let err = load(&root).expect_err("suffix/key mismatch must refuse");
        assert!(
            err.contains("bf16") && err.contains("fp32"),
            "the refusal must name both declarations: {err}"
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn rejects_unknown_dtype_token_rd1_4() {
        // The `dtype` key with any value outside fp32/bf16.
        for token in ["f8", "fp16", "b16", ""] {
            let root = bf16_twin("rd1-4-key");
            rewrite(
                &root.join("manifest.toml"),
                "dtype = \"bf16\"",
                &format!("dtype = {token:?}"),
            );
            let err = load(&root).expect_err("an unknown dtype token must refuse");
            assert!(
                err.contains(&format!("{token:?}")),
                "the refusal must name the offending token: {err}"
            );
            std::fs::remove_dir_all(&root).expect("cleanup");
        }

        // The weight_tying suffix with an unknown token, and with NO suffix.
        let config_path = Path::new("config.toml");
        for (bad, needle) in [
            ("lm_head=e1@fp16", "fp16"),
            ("lm_head=e1@b16", "b16"),
            ("lm_head=e1", "unknown weight_tying"),
        ] {
            let cfg = ModelConfig {
                weight_tying: Some(bad.into()),
                ..valid_config()
            };
            let err = cfg
                .validate(config_path)
                .expect_err("an undeclared weight_tying suffix must refuse");
            assert!(err.contains(needle), "{bad}: {err}");
        }
    }

    #[test]
    fn unset_tying_bf16_sidecar_rd1_5() {
        // No weight_tying line at all: the head is untied and read at the
        // declared dtype like the other two remainder files.
        let root = bf16_twin("rd1-5");
        std::fs::copy(root.join("e1.bf16.bin"), root.join("head.bf16.bin")).expect("head copy");
        rewrite(
            &root.join("manifest.toml"),
            "lm_head = \"e1.bf16.bin\"",
            "lm_head = \"head.bf16.bin\"",
        );
        let cfg_path = root.join("config.toml");
        let text = std::fs::read_to_string(&cfg_path).expect("config");
        let stripped: String = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("weight_tying"))
            .map(|l| format!("{l}\n"))
            .collect();
        std::fs::write(&cfg_path, stripped).expect("rewrite config");

        let m = load(&root).expect("an unset weight_tying with a bf16 sidecar must load");
        assert_eq!(m.remainder_dtype, RemainderDtype::Bf16);
        let head = m.lm_head.expect("the untied head is read, not tied");
        assert_eq!(head, widened_f32_floats(&dir().join("e1.f32.bin")));
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    // --- RD-2: reading at the declared itemsize ----------------------------

    #[test]
    fn loads_bf16_fixture_exact_values_rd2_1() {
        let m = load(&bf16_fixture_dir()).expect("the committed bf16 twin loads");
        assert_eq!(m.remainder_dtype, RemainderDtype::Bf16);
        // Half the fp32 byte count on disk: no fp32-sized read can have occurred.
        let bf16_emb = bf16_fixture_dir().join("e1.bf16.bin");
        let fp32_emb = dir().join("e1.f32.bin");
        assert_eq!(
            std::fs::metadata(&bf16_emb).expect("stat bf16").len() * 2,
            std::fs::metadata(&fp32_emb).expect("stat fp32").len()
        );
        // Every loaded value is the exactly-widened source value.
        assert_eq!(m.embedding, widened_f32_floats(&fp32_emb));
        assert_eq!(
            m.norm_weights,
            widened_f32_floats(&dir().join("norms.f32.bin"))
        );
        // And the widening is exact: the low 16 bits of every f32 word are zero.
        for v in m.embedding.iter().chain(m.norm_weights.iter()) {
            assert_eq!(
                (*v as f32).to_bits() & 0xFFFF,
                0,
                "a widened bf16 value must have a zero low mantissa half"
            );
        }
    }

    #[test]
    fn rejects_bf16_declared_wrong_len_rd2_2() {
        // bf16 declared over an fp32-sized file: the stat gate must reject it
        // rather than fall back to fp32 (a silent misparse).
        let root = bf16_twin("rd2-2-long");
        let fp32_len = std::fs::metadata(dir().join("norms.f32.bin"))
            .expect("stat fp32 norms")
            .len();
        std::fs::write(root.join("norms.bf16.bin"), vec![0u8; fp32_len as usize])
            .expect("resize to the fp32 width");
        let err = load(&root).expect_err("a bf16 declaration over fp32 bytes must refuse");
        assert!(err.contains("norms.bf16.bin"), "{err}");
        assert!(err.contains("bf16"), "must name the declared dtype: {err}");
        assert!(
            err.contains(&format!("expected exactly {} bytes", fp32_len / 2)),
            "must name the expected byte count: {err}"
        );
        std::fs::remove_dir_all(&root).expect("cleanup");

        // The reverse: fp32 declared (absent key) over a bf16-sized file.
        let root = tmp_dir("rd2-2-short");
        copy_rec(&dir(), &root);
        let bf16_norms = bf16_bytes(&f32_floats(&dir().join("norms.f32.bin")));
        std::fs::write(root.join("norms.f32.bin"), bf16_norms).expect("resize");
        let err = load(&root).expect_err("an fp32 declaration over bf16 bytes must refuse");
        assert!(err.contains("norms.f32.bin"), "{err}");
        assert!(err.contains("fp32"), "must name the declared dtype: {err}");
        assert!(
            err.contains(&format!("expected exactly {fp32_len} bytes")),
            "must name the expected byte count: {err}"
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn fp32_fixture_unchanged_rd2_3() {
        // The fp32 path's byte math is untouched: the committed pre-change
        // fixture loads to the raw LE f32 values, at 4 bytes per element.
        let m = load(&dir()).expect("the committed fp32 fixture loads");
        assert_eq!(m.remainder_dtype, RemainderDtype::Float32);
        assert_eq!(
            std::fs::metadata(dir().join("e1.f32.bin"))
                .expect("stat")
                .len(),
            m.embedding.len() as u64 * 4
        );
        assert_eq!(
            m.embedding,
            std::fs::read(dir().join("e1.f32.bin"))
                .expect("bytes")
                .chunks_exact(4)
                .map(|p| f64::from(f32::from_le_bytes([p[0], p[1], p[2], p[3]])))
                .collect::<Vec<f64>>()
        );
    }

    #[test]
    fn weight_tying_accepts_only_declared_dtypes() {
        let cfg = |wt: &str| ModelConfig {
            weight_tying: Some(wt.into()),
            ..valid_config()
        };
        let config_path = Path::new("config.toml");
        assert!(cfg("lm_head=e1@fp32").validate(config_path).is_ok());
        assert!(cfg("lm_head=e1@bf16").validate(config_path).is_ok());
        assert!(
            cfg("lm_head=head.bf16.bin@bf16")
                .validate(config_path)
                .is_ok()
        );
        for bad in ["lm_head=e1@fp8", "lm_head=e1", "lm_head=@bf16", "banana"] {
            let err = cfg(bad).validate(config_path).expect_err(bad);
            assert!(err.contains("unknown weight_tying"), "{bad}: {err}");
        }
    }

    #[test]
    fn loads_the_valid_fixture() {
        let m = load(&dir()).expect("fixture model loads");
        assert_eq!(m.model_id, "server-model-fixture");
        // One real transformer block = seven packed projections.
        assert_eq!(m.layers.len(), 7);
        assert_eq!(m.vocab_size, 32);
        assert_eq!(m.d_model, 4);
        assert_eq!(m.embedding.len(), 32 * 4);
        assert!(m.lm_head.is_none(), "fixture is weight-tied");
    }

    #[test]
    fn cross_inventory_sidecar_count_is_rejected() {
        // N-1.3: identical geometry, different declared inventory -> the
        // sidecar is validated against ITS OWN inventory's count. The valid
        // fixture (Qwen3-class inventory, 16-float sidecar) must FAIL with a
        // Llama-class inventory (expects 12 floats), proving the count is
        // inventory-derived, not baked into the loader.
        let tmp = tmp_dir("n13");
        copy_rec(&dir(), &tmp);
        let cfg_path = tmp.join("config.toml");
        let cfg = std::fs::read_to_string(&cfg_path).expect("config");
        let llama_cfg = cfg.replace(
            "norm_slots = [\"input\", \"post_attention\", \"q_norm\", \"k_norm\"]",
            "norm_slots = [\"input\", \"post_attention\"]",
        );
        assert_ne!(cfg, llama_cfg, "inventory line must exist to be swapped");
        std::fs::write(&cfg_path, llama_cfg).expect("rewrite config");
        let err = load(&tmp).expect_err("sidecar must reject against the other inventory");
        // 12 Llama-class floats x 4 bytes = 48: the swap target's count.
        assert!(
            err.contains("48 bytes"),
            "error must name the other inventory's byte count: {err}"
        );
        std::fs::remove_dir_all(&tmp).expect("cleanup");
    }

    #[test]
    fn containment_rejects_escape_and_absolute() {
        let base = Path::new("/models/x");
        for bad in [
            "../etc/shadow",
            "/etc/shadow",
            "a/../../b",
            "..",
            "layers/../../out.qpk1",
            "",
            ".",
        ] {
            assert!(
                contained_join(base, bad).is_err(),
                "{bad:?} must be rejected"
            );
        }
        assert!(contained_join(base, "layers/a.qpk1").is_ok());
    }

    #[test]
    fn config_validation_rejects_bogus_weight_tying() {
        let cfg = valid_config();
        let cfg = ModelConfig {
            weight_tying: Some("bogus".into()),
            ..cfg
        };
        assert!(cfg.validate(Path::new("config.toml")).is_err());
        let cfg = ModelConfig {
            weight_tying: Some("lm_head=e1@fp32".into()),
            ..cfg
        };
        assert!(cfg.validate(Path::new("config.toml")).is_ok());
    }

    /// A geometry-valid config (qk-attention N-2 shape).
    fn valid_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            d_model: 4,
            eos_token_id: 3,
            bos_token_id: 1,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            norm_slots: vec![
                "input".into(),
                "post_attention".into(),
                "q_norm".into(),
                "k_norm".into(),
            ],
            weight_tying: Some("lm_head=e1@fp32".into()),
        }
    }

    #[test]
    fn consistent_geometry_passes() {
        assert!(valid_config().validate(Path::new("config.toml")).is_ok());
    }

    #[test]
    fn geometry_rejects_kv_width_mismatch() {
        // n_kv_heads * head_dim (3 * 2 = 6) != d_model (4).
        let cfg = ModelConfig {
            n_kv_heads: 3,
            ..valid_config()
        };
        let err = cfg
            .validate(Path::new("config.toml"))
            .expect_err("mismatched KV width must fail");
        assert!(err.contains("GQA width mismatch"), "{err}");
    }

    #[test]
    fn geometry_rejects_non_multiple_heads() {
        // 3 query heads over 2 KV heads: no clean GQA grouping.
        let cfg = ModelConfig {
            n_heads: 3,
            ..valid_config()
        };
        let err = cfg
            .validate(Path::new("config.toml"))
            .expect_err("non-multiple heads must fail");
        assert!(err.contains("GQA group mismatch"), "{err}");
    }

    #[test]
    fn inventory_rejects_unknown_empty_and_duplicate() {
        let empty = ModelConfig {
            norm_slots: vec![],
            ..valid_config()
        };
        assert!(empty.validate(Path::new("config.toml")).is_err());
        let unknown = ModelConfig {
            norm_slots: vec!["input".into(), "banana".into()],
            ..valid_config()
        };
        let err = unknown
            .validate(Path::new("config.toml"))
            .expect_err("unknown slot must fail");
        assert!(err.contains("unknown norm slot"), "{err}");
        let dupe = ModelConfig {
            norm_slots: vec!["input".into(), "input".into()],
            ..valid_config()
        };
        let err = dupe
            .validate(Path::new("config.toml"))
            .expect_err("duplicate slot must fail");
        assert!(err.contains("duplicate norm slot"), "{err}");
        // Known alternative inventories load-validate: Llama-class (no
        // q/k norms) and Gemma-style post_feedforward.
        let llama_like = ModelConfig {
            norm_slots: vec!["input".into(), "post_attention".into()],
            ..valid_config()
        };
        assert!(llama_like.validate(Path::new("config.toml")).is_ok());
        let gemma_like = ModelConfig {
            norm_slots: vec![
                "input".into(),
                "post_attention".into(),
                "post_feedforward".into(),
            ],
            ..valid_config()
        };
        assert!(gemma_like.validate(Path::new("config.toml")).is_ok());
    }

    #[test]
    fn geometry_rejects_zero_and_overflow() {
        for (field, cfg) in [
            (
                "n_layers",
                ModelConfig {
                    n_layers: 0,
                    ..valid_config()
                },
            ),
            (
                "n_layers cap",
                ModelConfig {
                    n_layers: MAX_LAYERS + 1,
                    ..valid_config()
                },
            ),
            (
                "n_heads",
                ModelConfig {
                    n_heads: 0,
                    ..valid_config()
                },
            ),
            (
                "n_kv_heads",
                ModelConfig {
                    n_kv_heads: 0,
                    ..valid_config()
                },
            ),
            (
                "head_dim",
                ModelConfig {
                    head_dim: 0,
                    ..valid_config()
                },
            ),
            (
                "rms_norm_eps",
                ModelConfig {
                    rms_norm_eps: 0.0,
                    ..valid_config()
                },
            ),
            (
                "rms_norm_eps neg",
                ModelConfig {
                    rms_norm_eps: -1.0,
                    ..valid_config()
                },
            ),
            (
                "rope_theta",
                ModelConfig {
                    rope_theta: 0.0,
                    ..valid_config()
                },
            ),
            (
                "rope_theta inf",
                ModelConfig {
                    rope_theta: f64::INFINITY,
                    ..valid_config()
                },
            ),
        ] {
            assert!(
                cfg.validate(Path::new("config.toml")).is_err(),
                "{field} must be rejected"
            );
        }
    }

    #[test]
    fn missing_config_field_fails_toml_parse() {
        // N-2.2: any missing geometry field aborts the parse itself, so a
        // crafted directory can never omit the contract.
        let missing = "vocab_size = 32\nd_model = 4\neos_token_id = 3\nbos_token_id = 1\n";
        let parsed: Result<ModelConfig, _> = toml::from_str(missing);
        assert!(parsed.is_err(), "missing geometry must fail the parse");
    }

    #[test]
    fn loads_the_valid_fixture_with_sidecar_and_geometry() {
        let m = load(&dir()).expect("fixture model loads");
        // Inventory [input, post_attention, q_norm, k_norm] at d_model 4,
        // head_dim 2: 4+4+2+2 = 12 floats/block; +4 final = 28.
        assert_eq!(
            m.norm_weights.len(),
            inventory_block_floats(&m.norm_slots, m.d_model, m.head_dim) * m.n_layers + m.d_model
        );
        assert_eq!(
            m.norm_slots,
            vec![
                NormSlot::Input,
                NormSlot::PostAttention,
                NormSlot::QNorm,
                NormSlot::KNorm
            ]
        );
        assert_eq!(m.n_heads, 2);
        assert_eq!(m.n_kv_heads, 2);
        assert_eq!(m.head_dim, 2);
        assert!((m.rms_norm_eps - 1e-6).abs() < 1e-30);
        assert!((m.rope_theta - 10_000.0).abs() < 1e-30);
    }

    fn next_page(x: u64) -> u64 {
        let rem = x % crate::blob::PAGE_SIZE;
        if rem == 0 {
            x
        } else {
            x + (crate::blob::PAGE_SIZE - rem)
        }
    }

    fn find_embedding_offset_pos(blob: &[u8]) -> usize {
        let header_len = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as u64;
        let table_start = next_page(12 + header_len) as usize;
        let count = u32::from_le_bytes(blob[table_start..table_start + 4].try_into().unwrap());
        let mut pos = table_start + 4;
        for _ in 0..count {
            let name_len = blob[pos] as usize;
            let name_start = pos + 1;
            let name = std::str::from_utf8(&blob[name_start..name_start + name_len]).unwrap();
            let offset_pos = name_start + name_len + 1;
            if name == "embedding" {
                return offset_pos;
            }
            pos = offset_pos + 8 + 8;
        }
        panic!("no embedding section in fixture");
    }

    // BL-4.2 ordering: the alignment assert runs on the pread prefix, so a
    // misaligned blob must refuse with ZERO mapper calls. The probe counts
    // calls; a bug that maps first would make the count 1.
    #[test]
    fn misaligned_blob_refused_before_any_mmap() {
        let mut blob = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/server-model.qpk"),
        )
        .expect("committed blob fixture");
        let pos = find_embedding_offset_pos(&blob);
        let offset = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap());
        blob[pos..pos + 8].copy_from_slice(&(offset + 8).to_le_bytes());
        let path = std::env::temp_dir().join(format!(
            "qpk1-misaligned-{}-{}.qpk",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::write(&path, &blob).expect("write misaligned blob");
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let result = load_blob_with(&path, |file| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            unsafe { memmap2::Mmap::map(file) }
        });
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("a misaligned blob must refuse");
        assert!(err.contains("4096"), "names the alignment: {err}");
        assert!(err.contains("embedding"), "names the section: {err}");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "mmap must not be called when alignment fails (BL-4.2 ordering)"
        );
    }
}
