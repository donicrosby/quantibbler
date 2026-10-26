//! CLI (design D1): clap definitions for the `quantibbler-server` binary.
//!
//! One artifact selector is required (SR-1.5): `--artifact` names the qpk1
//! model — a single-file `.qpk` blob or a legacy directory, auto-detected by
//! the loader — and `--model-dir` is its deprecated spelling. The group makes
//! "both" and "neither" parse-time usage errors, so no model load is ever
//! attempted under an ambiguous selection.

use clap::{ArgGroup, Parser};

/// Pure-Rust OpenAI-compatible inference server for qpk1 models.
#[derive(Debug, Parser)]
#[command(name = "quantibbler-server", version, about)]
#[command(group(
    ArgGroup::new("source")
        .required(true)
        .multiple(false)
        .args(["artifact", "model_dir"])
))]
pub struct Cli {
    /// Path to the qpk1 model artifact: a single-file `.qpk` blob (loaded via
    /// mmap) or a legacy model directory (`manifest.toml` + `config.toml` +
    /// layers). The format is auto-detected.
    #[arg(long = "artifact")]
    pub artifact: Option<String>,

    /// Deprecated alias for `--artifact` (the legacy directory spelling).
    /// Supplying it beside `--artifact` is refused at parse.
    #[arg(long = "model-dir")]
    pub model_dir: Option<String>,

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

impl Cli {
    /// The resolved artifact path. The clap `source` group guarantees exactly
    /// one selector is present, so this cannot be reached with both or
    /// neither.
    pub fn artifact_path(&self) -> &str {
        self.artifact
            .as_deref()
            .or(self.model_dir.as_deref())
            .expect("clap's `source` group requires --artifact or --model-dir")
    }

    /// True when the legacy spelling was used (the caller emits the
    /// deprecation notice).
    pub fn used_deprecated_model_dir(&self) -> bool {
        self.model_dir.is_some()
    }
}
