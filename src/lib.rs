//! # IMECE Core — Local-First Autonomous Agent Runtime
//!
//! A Rust-native runtime for building fully autonomous AI agents that run
//! entirely on-device. Zero API dependency. Zero cloud lock-in. Full
//! data sovereignty. Optimized for edge devices with ≤8 GB VRAM.
//!
//! ## Architecture
//!
//! IMECE Core is organized into four composable modules that form a
//! complete agent pipeline:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`memory`] | Chain-of-Memory (CoM) with DMCE chain evolution & LanceDB persistence |
//! | [`inference`] | KV-Cache "Time Travel" rollback & Execution Guided Generation via llama.cpp |
//! | [`actor`] | Asynchronous multi-agent swarm via Tokio MPSC channels |
//! | [`embedding`] | Pluggable local embedding engine (`EmbeddingBackend` trait, ships with Voyage-4 Nano) |
//!
//! Each module is decoupled and can be used independently. For example,
//! you can run the embedding subsystem without loading the LLM inference
//! backend, or use the actor swarm without the memory chain.
//!
//! ## Feature Flags
//!
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `llama_backend` | `off` | Enables the llama.cpp FFI backend for LLM inference (Module 2) |
//! | `cuda` | `off` | Enables CUDA GPU offloading (requires `llama_backend`) |
//!
//! ## Quickstart
//!
//! ### Memory — Store and retrieve with DMCE
//!
//! ```rust,no_run
//! use imece_core::memory::store::MemoryStore;
//! use imece_core::memory::node::{MemoryNode, Role};
//! use imece_core::memory::chain::DmceEngine;
//! use ndarray::Array1;
//!
//! // Create an in-memory store (dimension must match your embedding model)
//! let mut store = MemoryStore::new_in_memory(256).unwrap();
//!
//! // Insert memory nodes with pre-computed embeddings
//! let embedding = Array1::from_vec(vec![0.0f32; 256]); // from Module 4
//! let node = MemoryNode::new(
//!     "Rust's ownership model prevents data races.".into(),
//!     Role::Agent,
//!     embedding,
//! );
//! store.insert(&node).unwrap();
//!
//! // Build a memory chain using DMCE
//! let engine = DmceEngine::new(0.6, 20, 8);
//! let query = Array1::from_vec(vec![0.0f32; 256]);
//! let chain = engine.evolve(&store, &query);
//! ```
//!
//! ### Embedding — Generate local vectors
//!
//! ```rust,no_run
//! use imece_core::embedding::config::EmbeddingServiceConfig;
//!
//! let config = EmbeddingServiceConfig::default(); // Voyage-4 Nano, 256-d, int8
//! let backend = config.create_backend().unwrap();
//! let embedding = backend.embed_query("What is IMECE?").unwrap();
//! let vector = embedding.to_f32(); // Array1<f32> for memory subsystem
//! ```
//!
//! ## Design Philosophy
//!
//! - **Local-first**: All computation happens on-device. No network calls.
//! - **Pluggable**: Every subsystem exposes a trait (`EmbeddingBackend`,
//!   `KvCacheManager`, `Agent`, `ActionExecutor`) for custom implementations.
//! - **Low-VRAM optimized**: MRL truncation, int8 quantization, APT chain
//!   truncation, and automatic build parallelism throttling.
//! - **Zero Python dependency**: Pure Rust with C FFI only for llama.cpp.
//!
//! ## External Resources
//!
//! - [GitHub Repository](https://github.com/imece-ai/imece-core)
//! - [Examples Repository](https://github.com/imece-ai/imece-examples)
//! - [Chain-of-Memory Paper (arXiv:2601.14287v1)](https://arxiv.org/abs/2601.14287v1)

pub mod actor;
pub mod embedding;
pub mod inference;
pub mod memory;
