//! # Memory Store — In-Memory Flat-Index
//!
//! Flat-index storage layer for memory nodes with brute-force cosine
//! similarity retrieval. This is the baseline storage backend for the
//! Chain-of-Memory subsystem.
//!
//! ## When to Use
//!
//! | Store | Best For | Performance |
//! |-------|----------|-------------|
//! | `MemoryStore` (this module) | ≤10k nodes, no persistence needed | O(n) retrieval, zero startup cost |
//! | [`LanceMemoryStore`](super::lance_store) | >10k nodes, persistence required | ANN indexing, disk-backed |
//!
//! ## Usage
//!
//! ```rust
//! use imece_core::memory::store::MemoryStore;
//! use imece_core::memory::node::{MemoryNode, Role};
//! use ndarray::Array1;
//!
//! let mut store = MemoryStore::new_in_memory(3).unwrap();
//!
//! let node = MemoryNode::new(
//!     "Rust is a systems programming language.".into(),
//!     Role::Agent,
//!     Array1::from_vec(vec![0.9, 0.1, 0.0]),
//! );
//! store.insert(&node).unwrap();
//!
//! // Retrieve Top-K candidates for DMCE
//! let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
//! let results = store.top_k(&query, 5, &[]).unwrap();
//! assert_eq!(results[0].node.text, "Rust is a systems programming language.");
//! ```
//!
//! ## Persistent Backend: LanceDB
//!
//! When node count exceeds the brute-force threshold, swap the retrieval
//! implementation to [`LanceMemoryStore`](super::lance_store) which
//! uses LanceDB's native ANN index without changing the DMCE API.

use ndarray::Array1;

use super::error::{MemoryError, MemoryResult};
use super::node::MemoryNode;

// ---------------------------------------------------------------------------
// Scored Candidate (internal)
// ---------------------------------------------------------------------------

/// A memory node paired with its cosine similarity score against a query.
/// Used internally during Top-K retrieval.
#[derive(Debug, Clone)]
pub struct ScoredNode {
    pub node: MemoryNode,
    /// Cosine similarity ∈ [-1, 1].
    pub score: f32,
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Flat-index database `M = {m_1, m_2, ..., m_n}`.
///
/// Stores all memory nodes and provides Top-K retrieval via cosine similarity
/// for the DMCE candidate pool `P`.
pub struct MemoryStore {
    /// Expected embedding dimensionality. Every inserted node must match.
    dim: usize,

    /// All nodes in insertion order.
    nodes: Vec<MemoryNode>,
}

impl MemoryStore {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create a new in-memory store expecting embeddings of dimension `dim`.
    pub fn new_in_memory(dim: usize) -> MemoryResult<Self> {
        if dim == 0 {
            return Err(MemoryError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            dim,
            nodes: Vec::new(),
        })
    }

    // ── Insertion ─────────────────────────────────────────────────────────

    /// Insert a memory node into the store.
    ///
    /// Returns `DimensionMismatch` if the node's embedding dimension does not
    /// match the store's expected `dim`.
    pub fn insert(&mut self, node: &MemoryNode) -> MemoryResult<()> {
        if node.dim() != self.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dim,
                got: node.dim(),
            });
        }
        self.nodes.push(node.clone());
        Ok(())
    }

    /// Batch insert multiple nodes.
    pub fn insert_batch(&mut self, nodes: &[MemoryNode]) -> MemoryResult<()> {
        for node in nodes {
            self.insert(node)?;
        }
        Ok(())
    }

    // ── Retrieval ─────────────────────────────────────────────────────────

    /// Retrieve the Top-K most similar nodes to `query` via brute-force
    /// cosine similarity.
    ///
    /// This populates candidate pool `P` for the DMCE algorithm.
    ///
    /// # Arguments
    /// * `query` — Query embedding `q ∈ ℝ^d`.
    /// * `top_k` — Maximum number of candidates to return.
    /// * `exclude_ids` — Node IDs to skip (already consumed by the chain).
    pub fn top_k(
        &self,
        query: &Array1<f32>,
        top_k: usize,
        exclude_ids: &[uuid::Uuid],
    ) -> MemoryResult<Vec<ScoredNode>> {
        if query.len() != self.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let mut scored: Vec<ScoredNode> = self
            .nodes
            .iter()
            .filter(|n| !exclude_ids.contains(&n.id))
            .map(|n| ScoredNode {
                score: cosine_similarity(query, &n.embedding),
                node: n.clone(),
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(top_k);
        Ok(scored)
    }

    /// Return all stored nodes (read-only).
    pub fn all_nodes(&self) -> &[MemoryNode] {
        &self.nodes
    }

    /// Number of stored nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the store is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Expected embedding dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Cosine Similarity (ndarray SIMD-friendly)
// ---------------------------------------------------------------------------

/// Compute `cos(a, b) = (a · b) / (‖a‖ · ‖b‖)`.
///
/// Returns `0.0` when either vector has zero magnitude to avoid NaN
/// propagation — a safe default for the DMCE gating score.
#[inline]
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    let denom = norm_a * norm_b;

    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::node::Role;

    fn make_node(text: &str, emb: Vec<f32>) -> MemoryNode {
        MemoryNode::new(text.into(), Role::User, Array1::from_vec(emb))
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let zero = Array1::from_vec(vec![0.0, 0.0]);
        assert_eq!(cosine_similarity(&a, &zero), 0.0);
    }

    #[test]
    fn test_store_insert_and_len() {
        let mut store = MemoryStore::new_in_memory(3).unwrap();
        let node = make_node("test", vec![0.1, 0.2, 0.3]);
        store.insert(&node).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_store_dimension_mismatch() {
        let mut store = MemoryStore::new_in_memory(3).unwrap();
        let bad_node = make_node("wrong dim", vec![0.1, 0.2]); // dim=2 vs expected 3
        assert!(store.insert(&bad_node).is_err());
    }

    #[test]
    fn test_top_k_ordering() {
        let mut store = MemoryStore::new_in_memory(3).unwrap();

        let close = make_node("close", vec![0.9, 0.1, 0.0]);
        let far = make_node("far", vec![0.0, 0.0, 1.0]);
        store.insert(&close).unwrap();
        store.insert(&far).unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = store.top_k(&query, 2, &[]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node.text, "close");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_top_k_excludes() {
        let mut store = MemoryStore::new_in_memory(2).unwrap();

        let n1 = make_node("alpha", vec![1.0, 0.0]);
        let n2 = make_node("beta", vec![0.9, 0.1]);
        store.insert(&n1).unwrap();
        store.insert(&n2).unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0]);
        let results = store.top_k(&query, 10, &[n1.id]).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node.text, "beta");
    }
}
