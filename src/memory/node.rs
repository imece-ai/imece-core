//! # Memory Node
//!
//! Atomic unit of the IMECE flat-index memory database.
//!
//! Each node represents a single interaction in an agent's memory,
//! corresponding to the formal definition:
//!
//! ```text
//! m_{i,j} = (x, τ, ρ, e)
//! ```
//!
//! Where:
//!   - `x` : Text payload (the content of the interaction)
//!   - `τ` : Timestamp (Unix Epoch, seconds)
//!   - `ρ` : Role — [`Role::User`], [`Role::Agent`], or [`Role::System`]
//!   - `e` : Embedding vector `e ∈ ℝ^d` (dimension determined by the
//!           embedding model, e.g., 256 for Voyage-4 Nano with MRL truncation)
//!
//! ## Example
//!
//! ```rust
//! use imece_core::memory::node::{MemoryNode, Role};
//! use ndarray::Array1;
//!
//! // Create a user interaction node
//! let embedding = Array1::from_vec(vec![0.1f32; 256]);
//! let node = MemoryNode::new(
//!     "How does Rust prevent data races?".into(),
//!     Role::User,
//!     embedding,
//! );
//!
//! assert_eq!(node.role, Role::User);
//! assert_eq!(node.dim(), 256);
//! assert!(node.timestamp > 0);
//! ```
//!
//! ## Serialization
//!
//! `MemoryNode` derives `Serialize` and `Deserialize` for JSON/TOML
//! persistence. The embedding vector is serialized as a flat `Vec<f32>`
//! array for interoperability.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The origin role of a memory interaction.
///
/// Maps to `ρ` in the specification:
///   - `User`   — Human input
///   - `Agent`  — LLM-generated response
///   - `System` — Internal system observation / tool output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Agent,
    System,
}

impl Role {
    /// Encode role as a single `u8` for compact storage.
    #[inline]
    pub fn as_u8(&self) -> u8 {
        match self {
            Role::User => 0,
            Role::Agent => 1,
            Role::System => 2,
        }
    }

    /// Decode from the compact `u8` representation.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Role::User),
            1 => Some(Role::Agent),
            2 => Some(Role::System),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryNode
// ---------------------------------------------------------------------------

/// A single atomic memory node stored in the flat-index database `M`.
///
/// # Layout
/// ```text
/// m_{i,j} = (x, τ, ρ, e)
/// ```
///
/// This struct is designed to be lightweight and `Clone`-able so it can be
/// freely moved between the candidate pool, chain builder, and storage layer
/// without lifetime entanglement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    // ── Identity ──────────────────────────────────────────────────────────
    /// Unique node identifier (UUID v4).
    pub id: Uuid,

    // ── Payload (x) ───────────────────────────────────────────────────────
    /// Raw text content of the interaction.
    pub text: String,

    // ── Temporal (τ) ──────────────────────────────────────────────────────
    /// Unix Epoch timestamp in seconds when this node was created.
    pub timestamp: u64,

    // ── Role (ρ) ──────────────────────────────────────────────────────────
    /// Origin role of this interaction.
    pub role: Role,

    // ── Embedding (e ∈ ℝ^d) ──────────────────────────────────────────────
    /// Dense embedding vector. Dimension `d` is determined at runtime by the
    /// embedding model (e.g., 384 for `all-MiniLM-L6-v2`).
    ///
    /// Stored as a contiguous `ndarray::Array1<f32>` for zero-copy SIMD
    /// dot-product computation during DMCE scoring.
    #[serde(
        serialize_with = "serialize_embedding",
        deserialize_with = "deserialize_embedding"
    )]
    pub embedding: Array1<f32>,
}

impl MemoryNode {
    /// Create a new memory node with an auto-generated UUID and current
    /// timestamp.
    pub fn new(text: String, role: Role, embedding: Array1<f32>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System clock before UNIX epoch")
            .as_secs();

        Self {
            id: Uuid::new_v4(),
            text,
            timestamp,
            role,
            embedding,
        }
    }

    /// Create a node with an explicit ID and timestamp (for rehydration from
    /// the database).
    pub fn from_raw(
        id: Uuid,
        text: String,
        timestamp: u64,
        role: Role,
        embedding: Array1<f32>,
    ) -> Self {
        Self {
            id,
            text,
            timestamp,
            role,
            embedding,
        }
    }

    /// Dimensionality of the embedding vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }
}

// ---------------------------------------------------------------------------
// Serde helpers for ndarray::Array1<f32>
// ---------------------------------------------------------------------------

fn serialize_embedding<S>(embedding: &Array1<f32>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let slice = embedding.as_slice().expect("embedding not contiguous");
    slice.serialize(serializer)
}

fn deserialize_embedding<'de, D>(deserializer: D) -> Result<Array1<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let vec = Vec::<f32>::deserialize(deserializer)?;
    Ok(Array1::from_vec(vec))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::User, Role::Agent, Role::System] {
            assert_eq!(Role::from_u8(role.as_u8()), Some(role));
        }
        assert_eq!(Role::from_u8(255), None);
    }

    #[test]
    fn test_memory_node_creation() {
        let dim = 384;
        let emb = Array1::from_vec(vec![0.42_f32; dim]);
        let node = MemoryNode::new("Hello, world!".into(), Role::User, emb.clone());

        assert_eq!(node.text, "Hello, world!");
        assert_eq!(node.role, Role::User);
        assert_eq!(node.dim(), dim);
        assert!(node.timestamp > 0);
    }

    #[test]
    fn test_memory_node_serde_roundtrip() {
        let dim = 128;
        let emb = Array1::from_vec(vec![1.0_f32; dim]);
        let node = MemoryNode::new("serialize me".into(), Role::Agent, emb);

        let json = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.id, node.id);
        assert_eq!(restored.text, node.text);
        assert_eq!(restored.role, node.role);
        assert_eq!(restored.dim(), dim);
        assert_eq!(restored.embedding, node.embedding);
    }
}
