//! The single OpenAI error envelope (design D7): every error response in
//! the server renders `{"error": {"message", "type", "param", "code"}}`
//! through this one IntoResponse type — no error path leaks anything else.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub err_type: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            err_type: "invalid_request_error",
            param: None,
            code: None,
        }
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!(
                "The model `{}` does not exist or is not served by this process.",
                model.into()
            ),
            err_type: "invalid_request_error",
            param: Some("model".into()),
            code: Some("model_not_found".into()),
        }
    }

    pub fn not_ready() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "The model is still loading; the server is not ready.".into(),
            err_type: "server_error",
            param: None,
            code: Some("model_not_ready".into()),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            err_type: "server_error",
            param: None,
            code: None,
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "Missing or invalid API key.".into(),
            err_type: "invalid_request_error",
            param: None,
            code: Some("invalid_api_key".into()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.err_type,
                "param": self.param,
                "code": self.code,
            }
        });
        (self.status, Json(body)).into_response()
    }
}
