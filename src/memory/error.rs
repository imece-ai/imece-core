//! # Memory Error Types
//!
//! Unified error types for the Chain-of-Memory subsystem.
//!
//! All memory operations return [`MemoryResult<T>`] which wraps a
//! `Result<T, MemoryError>`. Error variants cover the main failure modes:
//!
//! - **Dimension mismatches** between query embeddings and stored vectors
//! - **Empty candidate pools** when DMCE has nothing to evolve
//! - **Database errors** from the LanceDB persistence layer
//! - **Serialization errors** during Arrow/JSON encoding
//!
//! ## Example
//!
//! ```rust
//! use imece_core::memory::store::MemoryStore;
//! use imece_core::memory::node::{MemoryNode, Role};
//! use ndarray::Array1;
//!
//! let mut store = MemoryStore::new_in_memory(3).unwrap();
//!
//! // Inserting a node with the wrong dimension returns DimensionMismatch
//! let bad_node = MemoryNode::new(
//!     "wrong dim".into(), Role::User,
//!     Array1::from_vec(vec![0.1, 0.2]),  // dim=2 vs expected 3
//! );
//! assert!(store.insert(&bad_node).is_err());
//! ```

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
