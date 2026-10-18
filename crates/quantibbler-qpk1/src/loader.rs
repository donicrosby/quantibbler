//! qpk1 model-directory loader (qk-server design D3, hardened).
//!
//! Loads `<model-dir>` end to end: `manifest.toml`, per-layer qpk1 streams
//! via the fail-closed reader, `config.toml`, and the fp32 sidecar
//! matrices. Every rule below is a hard rejection — no partial model is
//! ever returned.
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
//!    documented forms (`unset` or `"lm_head=e1@fp32"`); unknown values
//!    abort startup instead of being silently ignored.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::Layer;

/// Hard cap on one qpk1 layer stream (bytes). A legitimate layer for a
/// 100B-class model at r=256 is tens of MB; 4 GiB leaves two orders of
/// magnitude of headroom while still refusing multi-GB pathological reads.
const MAX_LAYER_BYTES: u64 = 4 << 30;
/// Hard cap on one sidecar fp32 matrix file (bytes). At vocab*d_model for
/// any v0.1-served model this is orders of magnitude above need; same
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
    /// fp32 embedding, `vocab_size x d_model`, widened to f64.
    pub embedding: Vec<f64>,
    /// fp32 lm_head when not weight-tied, same layout.
    pub lm_head: Option<Vec<f64>>,
    /// Norm sidecar, `norm_slots.total_floats() * n_layers + d_model`
    /// floats widened to f64: per block, every declared slot's vector in
    /// inventory order; then the model-final norm. Layout is fully
    /// derived from the declared inventory — no architecture is
    /// hard-coded (qk-attention N-1, amended).
    pub norm_weights: Vec<f64>,
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
    // paths; fp32 matrices are exact-size by construction, so the stat
    // length must equal the declared element count * 4 before any read.
    let sidecar = manifest.sidecar.ok_or_else(|| {
        format!(
            "{}: missing [sidecar] block (embedding/lm_head/norms weights)",
            manifest_path.display()
        )
    })?;
    let expected_floats = config
        .vocab_size
        .checked_mul(config.d_model)
        .ok_or_else(|| format!("{}: vocab_size * d_model overflows", config_path.display()))?;
    let expected_bytes = checked_bytes(expected_floats, &config_path)?;
    let emb_path = contained_join(model_dir, &sidecar.embedding).map_err(|why| {
        format!(
            "{}: sidecar embedding path {:?} rejected: {why}",
            manifest_path.display(),
            sidecar.embedding
        )
    })?;
    let head_path = contained_join(model_dir, &sidecar.lm_head).map_err(|why| {
        format!(
            "{}: sidecar lm_head path {:?} rejected: {why}",
            manifest_path.display(),
            sidecar.lm_head
        )
    })?;
    let embedding = read_f32_bin_exact(&emb_path, expected_bytes)?;
    let lm_head = if head_path == emb_path {
        None // weight-tied
    } else {
        Some(read_f32_bin_exact(&head_path, expected_bytes)?)
    };

    // Norm sidecar (qk-attention N-1, amended): per block the input
    // layernorm + post-attention layernorm (d_model each) and the
    // attention q/k norms (head_dim each, one shared vector per
    // projection), plus the model-final norm.
    let norms_path = contained_join(model_dir, &sidecar.norms).map_err(|why| {
        format!(
            "{}: sidecar norms path {:?} rejected: {why}",
            manifest_path.display(),
            sidecar.norms
        )
    })?;
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
    let norm_bytes = checked_bytes(norm_floats, &config_path)?;
    let norm_weights = read_f32_bin_exact(&norms_path, norm_bytes)?;

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
        embedding,
        lm_head,
        norm_weights,
    })
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

/// Read an fp32 sidecar matrix of EXACTLY `expected_bytes`: the stat length
/// must match before the read (the read itself is then exact by stat).
fn read_f32_bin_exact(path: &Path, expected_bytes: usize) -> Result<Vec<f64>, String> {
    let len = stat_regular_capped(path, MAX_SIDECAR_BYTES, "sidecar matrix")?;
    if len != expected_bytes as u64 {
        return Err(format!(
            "{}: {len} bytes; expected exactly {expected_bytes} bytes (a vocab_size x d_model little-endian float32 matrix)",
            path.display()
        ));
    }
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(raw
        .chunks_exact(4)
        .map(|p| f64::from(f32::from_le_bytes([p[0], p[1], p[2], p[3]])))
        .collect())
}

/// `floats * 4` with an explicit overflow diagnostic naming the config.
fn checked_bytes(floats: usize, config_path: &Path) -> Result<usize, String> {
    floats.checked_mul(4).ok_or_else(|| {
        format!(
            "{}: sidecar byte size overflows (vocab_size * d_model * 4)",
            config_path.display()
        )
    })
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
    /// Path (relative to the model dir) of the fp32 embedding matrix,
    /// `vocab_size x d_model`, row-major, little-endian float32.
    embedding: String,
    /// Path of the fp32 lm_head matrix; when equal to `embedding` the head
    /// is weight-tied.
    lm_head: String,
    /// Path of the fp32 norm-weights sidecar (qk-attention N-1):
    /// inventory-derived floats, row-major, LE float32.
    norms: String,
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
    /// Documented forms (qk-server C5): unset, or `"lm_head=e1@fp32"`
    /// (weight-tied fp32 head). Anything else is a config error.
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
            None | Some("lm_head=e1@fp32") => {}
            Some(other) => {
                return Err(format!(
                    "{}: unknown weight_tying {other:?}; documented forms: unset or \"lm_head=e1@fp32\"",
                    config_path.display()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/server-model")
    }

    #[test]
    fn loads_the_valid_fixture() {
        let m = load(&dir()).expect("fixture model loads");
        assert_eq!(m.model_id, "server-model-fixture");
        assert_eq!(m.layers.len(), 1);
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
        let src = dir();
        let tmp = std::env::temp_dir().join(format!(
            "qpk1-n13-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
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
        copy_rec(&src, &tmp);
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
}
