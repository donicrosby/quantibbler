//! `quantibbler-server` — pure-Rust OpenAI-compatible inference server for
//! qpk1 models (qk-server capability, v0.1).
//!
//! Thin composition root: parse the CLI, load the model artifact
//! fail-closed, then serve until shutdown. All behavior lives in the
//! `cli` / `model` / `engine` / `api` (incl. `api::chat_template`) /
//! `sampler` modules, which the integration tests include directly
//! (design D1: one binary, no lib target in v0.1).

mod api;
mod cli;
mod engine;
mod model;
mod sampler;

use std::process::ExitCode;

use clap::Parser as _;

fn main() -> ExitCode {
    let cfg = cli::Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    match run(cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            // Fail-closed startup: a load failure never serves traffic and
            // exits non-zero with a diagnostic naming the failing file
            // (SR-1.2).
            eprintln!("quantibbler-server: error: {err}");
            ExitCode::from(1)
        }
    }
}

fn run(cfg: cli::Cli) -> Result<(), api::StartupError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| api::StartupError(format!("tokio runtime: {e}")))?;
    runtime.block_on(api::serve(cfg))
}
