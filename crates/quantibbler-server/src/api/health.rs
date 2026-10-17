//! GET /health (SR-6.1 / SR-6.2): 200 + readiness once serving, 503 while
//! the model has not (yet) loaded.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::api::AppState;

pub async fn health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<serde_json::Value>) {
    let ready = state.ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "ready": ready,
            "model": state.model_id,
            "available_permits": state.engine.available_permits(),
        })),
    )
}
