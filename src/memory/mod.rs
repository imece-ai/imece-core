//! # Memory Subsystem (Chain-of-Memory / CoM)
//!
//! Implements the memory architecture described in:
//! "Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution
//!  for LLM Agents" (arXiv:2601.14287v1).
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
