//! # Memory Subsystem (Module 1 — Chain-of-Memory / CoM)
//!
//! Implements the memory architecture described in:
//! [Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution
//!  for LLM Agents (arXiv:2601.14287v1)](https://arxiv.org/abs/2601.14287v1).
//!
//! ## Architecture Overview
//!
//! The memory subsystem provides a three-layer architecture:
//!
//! 1. **[`node::MemoryNode`]** — Atomic memory unit: `m = (x, τ, ρ, e)` where
//!    `x` is text, `τ` is timestamp, `ρ` is role (user/agent/system), and
//!    `e ∈ ℝ^d` is the embedding vector.
//!
//! 2. **Storage Layer** — Two interchangeable backends:
//!    - [`store::MemoryStore`] — In-memory flat-index with brute-force cosine
//!      similarity. Best for ≤10k nodes on constrained devices.
//!    - [`lance_store::LanceMemoryStore`] — Persistent LanceDB backend with
//!      optional IVF-PQ ANN indexing. Best for larger datasets or when
//!      persistence is required across agent restarts.
//!
//! 3. **[`chain::DmceEngine`]** — Builds semantically coherent memory chains
//!    from the flat-index store using DMCE (Dynamic Memory Chain Evolution)
//!    and APT (Adaptive Path Truncation).
//!
//! ## Usage
//!
//! ```rust
//! use imece_core::memory::store::MemoryStore;
//! use imece_core::memory::node::{MemoryNode, Role};
//! use imece_core::memory::chain::DmceEngine;
//! use ndarray::Array1;
//!
//! // 1. Create a store matching your embedding dimension
//! let mut store = MemoryStore::new_in_memory(3).unwrap();
//!
//! // 2. Insert nodes (embeddings would come from Module 4 in production)
//! let node = MemoryNode::new(
//!     "Rust prevents data races at compile time.".into(),
//!     Role::Agent,
//!     Array1::from_vec(vec![0.9, 0.1, 0.0]),
//! );
//! store.insert(&node).unwrap();
//!
//! // 3. Build a memory chain using DMCE
//! let engine = DmceEngine::new(
//!     0.6,   // β — APT truncation threshold
//!     20,    // Top-K candidate pool size
//!     8,     // Maximum chain length
//! );
//! let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
//! let chain = engine.evolve(&store, &query);
//! // `chain` is now an ordered Vec<MemoryNode> ready for LLM context injection
//! ```
//!
//! ## Submodules
//! - [`node`]: Atomic memory node definition (`m_{i,j} = (x, τ, ρ, e)`).
//! - [`store`]: In-memory flat-index storage with brute-force cosine similarity.
//! - [`lance_store`]: LanceDB-backed persistent vector storage with ANN indexing.
//! - [`chain`]: DMCE algorithm & APT truncation logic.
//! - [`error`]: Unified error types for the memory subsystem.

pub mod chain;
pub mod error;
pub mod lance_store;
pub mod node;
pub mod store;
