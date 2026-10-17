//! Model-directory acquisition (qk-server design D3): manifest scan, qpk1
//! layer load, tokenizer load, config load — fail-closed end to end. Any
//! missing file, qpk1 reader rejection, tokenizer parse failure, or config
//! problem aborts startup with a diagnostic naming the failing file; the
//! listener never starts on a partial model.

pub mod qpk1;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::engine::ModelWeights;

/// The loaded model: immutable weights plus the metadata the API needs.
pub struct LoadedModel {
    /// The served model id (`manifest.toml`'s `model_id`).
    pub id: String,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: tokenizers::Tokenizer,
    /// Config values consumed by the engine (`config.toml`).
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    /// Recorded config hyperparameters (informational in v0.1; the weights
    /// carry their own dims).
    #[allow(dead_code)]
    pub vocab_size: usize,
    #[allow(dead_code)]
    pub d_model: usize,
}

/// Load a qpk1 model directory, or fail with a diagnostic naming the file.
pub fn load(model_dir: &Path) -> Result<LoadedModel, String> {
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

    let config_path = model_dir.join("config.toml");
    let config_raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
    let config: ModelConfig =
        toml::from_str(&config_raw).map_err(|e| format!("{}: {e}", config_path.display()))?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;

    // Every manifest layer must load under the fail-closed qpk1 reader; no
    // partial model survives.
    let mut layers: Vec<(String, qpk1::Layer)> = Vec::new();
    for name in &manifest.layers {
        let rel = manifest.layer_files.get(name).ok_or_else(|| {
            format!(
                "{}: layer {name:?} has no entry in [layer_files]",
                manifest_path.display()
            )
        })?;
        let layer_path = model_dir.join(rel);
        let raw =
            std::fs::read(&layer_path).map_err(|e| format!("{}: {e}", layer_path.display()))?;
        let layer = qpk1::parse(&raw).map_err(|e| format!("{}: {e}", layer_path.display()))?;
        // Reader-side rank feasibility, mirroring the qk-representation
        // deserialize gate (rejects the rank-infeasible fixture class).
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
    if layers.is_empty() {
        return Err(format!(
            "{}: manifest lists no layers; nothing to serve",
            manifest_path.display()
        ));
    }

    // Non-quantized sidecar weights (OQ2 minimally): fp32 embeddings +
    // tied lm_head, byte layout pinned by the manifest's sidecar block.
    let sidecar = manifest.sidecar.as_ref().ok_or_else(|| {
        format!(
            "{}: missing [sidecar] block (embedding/lm_head weights)",
            manifest_path.display()
        )
    })?;
    let emb_path = model_dir.join(&sidecar.embedding);
    let embedding = load_f32_bin(&emb_path, config.vocab_size * config.d_model)?;
    let lm_head_path = model_dir.join(&sidecar.lm_head);
    let lm_head = if lm_head_path == emb_path {
        None // weight-tied
    } else {
        Some(load_f32_bin(
            &lm_head_path,
            config.vocab_size * config.d_model,
        )?)
    };

    let weights = Arc::new(
        ModelWeights::from_layers(
            layers,
            embedding,
            lm_head,
            config.d_model,
            config.vocab_size,
        )
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?,
    );
    Ok(LoadedModel {
        id: manifest.model_id,
        weights,
        tokenizer,
        eos_token_id: config.eos_token_id,
        bos_token_id: config.bos_token_id,
        vocab_size: config.vocab_size,
        d_model: config.d_model,
    })
}

/// Load a raw little-endian float32 binary of exactly `expected_len` floats.
fn load_f32_bin(path: &Path, expected_len: usize) -> Result<Vec<f64>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() != expected_len * 4 {
        return Err(format!(
            "{}: {} bytes; expected {} float32 values ({} bytes)",
            path.display(),
            raw.len(),
            expected_len,
            expected_len * 4
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|p| f64::from(f32::from_le_bytes([p[0], p[1], p[2], p[3]])))
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
    /// Recorded for forward-compatibility; the loader rejects unknown
    /// weight_tying values that do not describe a tied or explicit head.
    #[serde(default)]
    #[allow(dead_code)]
    weight_tying: Option<String>,
}

use serde::Deserialize;
