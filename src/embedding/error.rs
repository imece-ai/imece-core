//! # Embedding Error Types
//!
//! Unified error types for the Voyage-4 Nano embedding subsystem.
//! Follows the same `thiserror` pattern used by [`super::super::memory::error`]
//! and [`super::super::inference::error`].

use thiserror::Error;

// ---------------------------------------------------------------------------
// EmbeddingError
// ---------------------------------------------------------------------------

/// Errors that can occur during embedding model operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// ONNX Runtime session creation or inference failure.
    #[error("ONNX Runtime error: {0}")]
    OrtError(String),

    /// Tokenizer loading or encoding failure.
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    /// Model files not found at the configured path.
    #[error("Model not found at '{path}': {reason}")]
    ModelNotFound { path: String, reason: String },

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Embedding dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Convenience type alias for embedding results.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

// ---------------------------------------------------------------------------
// Foreign Error Conversions
// ---------------------------------------------------------------------------

impl From<ort::Error> for EmbeddingError {
    fn from(e: ort::Error) -> Self {
        EmbeddingError::OrtError(e.to_string())
    }
}
