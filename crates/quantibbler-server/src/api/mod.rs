//! API surface (design D1): axum routes, shared state, startup/shutdown.
//! Every error response flows through [`error::ApiError`] (design D7).

pub mod delta;
pub mod error;
pub mod generate;
pub mod health;
pub mod models;
pub mod types;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::routing::{get, post};

use crate::api::generate::check_auth;
use crate::cli::Cli;
use crate::engine::Engine;
use crate::model::LoadedModel;
use error::ApiError;

/// Shared server state. `ready` flips only after the model directory has
/// fully loaded (fail-closed, SR-1.*). The field is private behind
/// `ready()`/`set_ready()`; tests construct via `AppState::new`.
pub struct AppState {
    pub model_id: String,
    pub created_at: u64,
    pub engine: Arc<Engine>,
    /// Bearer key required when `--serve-api-key` is set (D7); `None` =
    /// open server.
    pub api_key: Option<String>,
    ready: AtomicBool,
}

impl AppState {
    /// Assemble state; readiness starts OFF until `set_ready()`.
    pub fn new(model: LoadedModel, max_concurrency: usize, api_key: Option<String>) -> Self {
        Self {
            model_id: model.id.clone(),
            created_at: now_secs(),
            engine: Arc::new(Engine::new(model, max_concurrency)),
            api_key,
            ready: AtomicBool::new(false),
        }
    }

    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    pub fn set_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
    }
}

/// Startup failure (process exits non-zero with this diagnostic, SR-1.2).
#[derive(Debug)]
pub struct StartupError(pub String);

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StartupError {}

/// Resolve the API key from the CLI config: `--serve-api-key-file` wins
/// (first line, trailing newline stripped; the file is read once at
/// startup), falling back to the inline `--serve-api-key` value. Keeping
/// long-lived keys out of argv keeps them out of `ps` and shell history.
fn api_key_for_serve(cfg: &crate::cli::Cli) -> Option<String> {
    if let Some(path) = &cfg.serve_api_key_file {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("--serve-api-key-file {path}: {e}"));
        let key = raw.lines().next().unwrap_or("").trim().to_string();
        if key.is_empty() {
            panic!("--serve-api-key-file {path}: file is empty");
        }
        Some(key)
    } else {
        cfg.serve_api_key.clone()
    }
}

/// Auth middleware for the unauthenticated-by-handler GET surfaces
/// (`/health`, `/v1/models`): when a key is configured, these demand it
/// too — readiness and model-id disclosure are not free (review finding:
/// key presence otherwise leaks via 401 vs 200 on /v1/models).
async fn auth_guard(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, ApiError> {
    check_auth(&state, &headers)?;
    Ok(next.run(request).await)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Enveloped 404 for unknown routes (SR-2.2 probing must see the envelope).
async fn fallback_404() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        message: "Unknown route.".into(),
        err_type: "invalid_request_error",
        param: None,
        code: None,
    }
}

/// Enveloped 405 so no error path leaks a non-envelope body (SR-7.3).
async fn method_not_allowed_405() -> ApiError {
    ApiError {
        status: StatusCode::METHOD_NOT_ALLOWED,
        message: "Method not allowed.".into(),
        err_type: "invalid_request_error",
        param: None,
        code: None,
    }
}

/// Map an axum `Json` rejection onto the OpenAI envelope (SR-7.1 / SR-7.3:
/// body-parse rejections never leak anything else).
pub fn json_rejection_error(rej: JsonRejection) -> ApiError {
    ApiError::bad_request(rej.body_text())
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/v1/models", get(models::list))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_guard,
        ))
        .route("/v1/chat/completions", post(generate::chat_completions))
        .route("/v1/completions", post(generate::completions))
        .method_not_allowed_fallback(method_not_allowed_405)
        .fallback(fallback_404)
        .with_state(state)
}

/// Load-then-serve (design D3/D4): the model loads fully BEFORE the
/// listener binds; readiness flips only after that. Any load failure
/// returns `Err` and the process exits non-zero without ever serving.
pub async fn serve(cfg: Cli) -> Result<(), StartupError> {
    let model_dir = PathBuf::from(&cfg.model_dir);
    let loaded = crate::model::load(&model_dir)
        .map_err(|e| StartupError(format!("model load failed: {e}")))?;
    tracing::info!(
        model_id = %loaded.id,
        vocab_size = loaded.vocab_size,
        d_model = loaded.d_model,
        dir = %model_dir.display(),
        "model loaded"
    );
    let state = Arc::new(AppState::new(
        loaded,
        cfg.max_concurrency,
        api_key_for_serve(&cfg),
    ));
    let router = build_router(Arc::clone(&state));
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| StartupError(format!("failed to bind {addr}: {e}")))?;
    // Ready only now: model fully loaded, listener bound, no traffic served
    // before this point (SR-1.1).
    state.set_ready();
    tracing::info!(%addr, model_id = %state.model_id, "ready");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| StartupError(format!("server error: {e}")))?;
    tracing::info!("shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
