//! Unified error types for the ImeceMemory subsystem.

use thiserror::Error;

/// All recoverable errors within the memory subsystem.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Embedding dimension mismatch between query and stored vectors.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The candidate pool was empty — nothing to evolve.
    #[error("Empty candidate pool: no nodes match the query")]
    EmptyCandidatePool,

    /// LanceDB / persistence layer error.
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Serialization / deserialization failure (Arrow/JSON).
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Convenience alias used throughout the memory subsystem.
pub type MemoryResult<T> = Result<T, MemoryError>;
