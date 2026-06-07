//! # Embedding Error Types
//!
//! Unified error types for the pluggable embedding subsystem.
//! Backend-specific errors (ORT, tokenizer, etc.) are wrapped in the
//! [`EmbeddingError::BackendError`] variant, keeping the public API
//! independent of any particular model engine.

use thiserror::Error;

// ---------------------------------------------------------------------------
// EmbeddingError
// ---------------------------------------------------------------------------

/// Errors that can occur during embedding operations.
///
/// Backend-agnostic errors (`ModelNotFound`, `InvalidConfig`, `DimensionMismatch`)
/// are represented directly. Backend-specific errors are encapsulated in
/// [`BackendError`](EmbeddingError::BackendError) with the originating backend
/// name for diagnostics.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Model files not found at the configured path.
    /// This is backend-agnostic: any local backend that loads model files
    /// from disk can produce this error.
    #[error("Model not found at '{path}': {reason}")]
    ModelNotFound {
        /// Path that was checked.
        path: String,
        /// Human-readable explanation.
        reason: String,
    },

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Embedding dimension mismatch (e.g., model output smaller than
    /// the requested MRL truncation dimension).
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension that was requested or expected.
        expected: usize,
        /// The dimension that was actually produced.
        got: usize,
    },

    /// The requested backend type is not available or not compiled in.
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// An error originating from a specific backend implementation.
    ///
    /// The `backend` field identifies which engine produced the error,
    /// and `source` carries the underlying cause for the error chain.
    #[error("[{backend}] {source}")]
    BackendError {
        /// Name of the backend that produced this error (e.g., `"voyage-4-nano"`).
        backend: String,
        /// The underlying backend-specific error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Convenience type alias for embedding results.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;
