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

/// A fully loaded and validated model directory.
pub struct ModelDirectory {
    /// The served model id (`manifest.toml`'s `model_id`).
    pub model_id: String,
    /// Layers in serving order.
    pub layers: Vec<(String, Layer)>,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub vocab_size: usize,
    pub d_model: usize,
    /// fp32 embedding, `vocab_size x d_model`, widened to f64.
    pub embedding: Vec<f64>,
    /// fp32 lm_head when not weight-tied, same layout.
    pub lm_head: Option<Vec<f64>>,
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

    // Sidecar: same containment discipline on the manifest-controlled
    // paths; fp32 matrices are exact-size by construction, so the stat
    // length must equal `vocab_size * d_model * 4` before any read.
    let sidecar = manifest.sidecar.ok_or_else(|| {
        format!(
            "{}: missing [sidecar] block (embedding/lm_head weights)",
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

    Ok(ModelDirectory {
        model_id: manifest.model_id,
        layers,
        eos_token_id: config.eos_token_id,
        bos_token_id: config.bos_token_id,
        vocab_size: config.vocab_size,
        d_model: config.d_model,
        embedding,
        lm_head,
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
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    vocab_size: usize,
    d_model: usize,
    eos_token_id: u32,
    bos_token_id: u32,
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
        let cfg = ModelConfig {
            vocab_size: 32,
            d_model: 4,
            eos_token_id: 3,
            bos_token_id: 1,
            weight_tying: Some("bogus".into()),
        };
        assert!(cfg.validate(Path::new("config.toml")).is_err());
        let cfg = ModelConfig {
            weight_tying: Some("lm_head=e1@fp32".into()),
            ..cfg
        };
        assert!(cfg.validate(Path::new("config.toml")).is_ok());
    }
}
