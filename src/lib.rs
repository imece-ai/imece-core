//! # IMECE Core Engine
//!
//! Local-First Autonomous Agent Framework.
//! Optimized for extreme low-VRAM (<=8GB) edge devices.
//!
//! ## Modules
//! - `memory`: Chain-of-Memory (CoM) subsystem with DMCE & APT.
//! - `inference`: KV-Cache Rollback & Execution Guided Generation.
//! - `actor`: Asynchronous Multi-Agent Swarm via Tokio MPSC.
//! - `embedding`: Pluggable local embedding engine (EmbeddingBackend trait).

pub mod actor;
pub mod embedding;
pub mod inference;
pub mod memory;
