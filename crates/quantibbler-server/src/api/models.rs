//! GET /v1/models (SR-2.1): OpenAI-shaped list with exactly one entry, the
//! single model loaded at startup. There is no registration surface — no
//! route exists that could add a second model (SR-2.2).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::api::AppState;

pub async fn list(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_id,
                "object": "model",
                "created": state.created_at,
                "owned_by": "quantibbler",
            }
        ],
    }))
}
