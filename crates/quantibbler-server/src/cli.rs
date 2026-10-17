//! CLI (design D1): clap definitions for the `quantibbler-server` binary.

use clap::Parser;

/// Pure-Rust OpenAI-compatible inference server for qpk1 models.
#[derive(Debug, Parser)]
#[command(name = "quantibbler-server", version, about)]
pub struct Cli {
    /// Path to the qpk1 model directory (manifest.toml + tokenizer.json +
    /// config.toml + layers/*.qpk1).
    #[arg(long = "model-dir", required = true)]
    pub model_dir: String,

    /// TCP port to bind.
    #[arg(long = "port", required = true)]
    pub port: u16,

    /// Bind address (default: loopback — plaintext HTTP is a non-goal
    /// boundary; TLS termination is the deployment's job).
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Max concurrent in-flight generations (design D5, IA3).
    #[arg(long = "max-concurrency", default_value_t = 4)]
    pub max_concurrency: usize,

    /// When set, requests must carry `Authorization: Bearer <key>` (D7).
    /// Prefer `--serve-api-key-file` to keep the key out of argv/`ps`.
    #[arg(long = "serve-api-key")]
    pub serve_api_key: Option<String>,

    /// Read the Bearer key from a file (first line); overrides
    /// `--serve-api-key`. Recommended over the inline flag.
    #[arg(long = "serve-api-key-file")]
    pub serve_api_key_file: Option<String>,
}
