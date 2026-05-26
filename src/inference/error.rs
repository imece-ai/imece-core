//! Unified error types for the Inference subsystem.

use thiserror::Error;

/// All recoverable errors within the inference subsystem.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// The llama.cpp backend returned an error.
    #[error("LLM backend error: {0}")]
    BackendError(String),

    /// KV-Cache operation failed.
    #[error("KV-Cache error: {0}")]
    KvCacheError(String),

    /// Rollback target token position is out of bounds.
    #[error("Rollback position {position} out of bounds (cache length: {cache_len})")]
    RollbackOutOfBounds { position: usize, cache_len: usize },

    /// The sandboxed executor returned a failure.
    #[error("Execution error (exit_code={exit_code}): {stderr}")]
    ExecutionFailed { exit_code: i32, stderr: String },

    /// Execution timed out.
    #[error("Execution timed out after {timeout_ms}ms")]
    ExecutionTimeout { timeout_ms: u64 },

    /// Maximum rollback retries exceeded.
    #[error("Max retries exceeded ({max_retries}) for action block")]
    MaxRetriesExceeded { max_retries: usize },

    /// Stop sequence was never found during generation.
    #[error("Generation completed without encountering stop sequence")]
    NoStopSequence,

    /// Model is not loaded.
    #[error("No model loaded — call load_model() first")]
    ModelNotLoaded,
}

/// Convenience alias.
pub type InferenceResult<T> = Result<T, InferenceError>;
