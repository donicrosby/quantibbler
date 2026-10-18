//! Model-directory acquisition (qk-server design D3): the qpk1 crate owns
//! the container contract (stream parse + hardened directory load — path
//! containment, stat-before-read, checked arithmetic, fail-closed
//! config); this module does the serving-side composition only: tokenizer
//! load, layer conversion to the engine's working layout, and the
//! LoadedModel handle. Fail-closed end to end: any problem aborts startup
//! with a diagnostic naming the failing file; the listener never starts
//! on a partial model.

use std::path::Path;
use std::sync::Arc;

use quantibbler_qpk1::loader;

use crate::engine::ModelWeights;

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

/// Load a qpk1 model directory, or fail with a diagnostic naming the file.
pub fn load(model_dir: &Path) -> Result<LoadedModel, String> {
    let dir = loader::load(model_dir)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;

    let weights = Arc::new(
        ModelWeights::from_layers(
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
        .map_err(|e| format!("{}: {e}", model_dir.join("manifest.toml").display()))?,
    );
    Ok(LoadedModel {
        id: dir.model_id,
        weights,
        tokenizer,
        eos_token_id: dir.eos_token_id,
        bos_token_id: dir.bos_token_id,
        vocab_size: dir.vocab_size,
        d_model: dir.d_model,
    })
}
