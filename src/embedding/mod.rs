//! # Embedding Subsystem (Module 4)
//!
//! Pluggable local embedding engine with support for multiple model
//! backends via the [`EmbeddingBackend`](backend::EmbeddingBackend) trait.
//! Currently ships with the **Voyage-4 Nano** backend
//! ([`VoyageNanoEngine`](engine::VoyageNanoEngine)), running entirely
//! on-device via ONNX Runtime with zero cloud dependencies.
//!
//! ## Architecture
//!
//! The subsystem is organized around a trait-object pattern:
//!
//! 1. **[`EmbeddingBackend`](backend::EmbeddingBackend)** — core trait
//!    that all backends implement.
//! 2. **[`EmbeddingServiceConfig`](config::EmbeddingServiceConfig)** —
//!    type-safe tagged enum for backend-specific configuration, with a
//!    [`create_backend()`](config::EmbeddingServiceConfig::create_backend)
//!    factory method.
//! 3. **[`EmbeddingOutput`](backend::EmbeddingOutput)** — precision-typed
//!    output (Float32 or Int8).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use imece_core::embedding::config::EmbeddingServiceConfig;
//!
//! let config = EmbeddingServiceConfig::default(); // Voyage-4 Nano, 256-d, int8
//! let backend = config.create_backend().unwrap();
//!
//! let embedding = backend.embed_query("What is IMECE?").unwrap();
//! let vector = embedding.to_f32(); // Array1<f32> for memory subsystem
//! ```
//!
//! ## Key Optimizations (Voyage-4 Nano)
//!
//! 1. **Matryoshka Representation Learning (MRL)**:
//!    The model is trained with MRL, allowing the 2048-dimensional output
//!    to be truncated to 256 dimensions with minimal retrieval quality loss.
//!    This reduces vector storage cost by 8× and accelerates similarity search.
//!
//! 2. **Native int8 Quantization (QAT)**:
//!    The model uses quantization-aware training to support native int8
//!    output vectors. This further reduces storage by 4× compared to float32
//!    while preserving retrieval fidelity — a capability specifically trained
//!    into the model, not a post-hoc approximation.
//!
//! ## Submodules
//! - [`backend`]: [`EmbeddingBackend`](backend::EmbeddingBackend) trait,
//!   [`EmbeddingOutput`](backend::EmbeddingOutput), and shared math utilities.
//! - [`config`]: Configuration types and backend factory.
//! - [`engine`]: [`VoyageNanoEngine`](engine::VoyageNanoEngine) — Voyage-4 Nano
//!   ONNX inference backend.
//! - [`error`]: Embedding-specific error types.
//!
//! ## Model Setup
//!
//! Before using the Voyage-4 Nano engine, you must provide the ONNX model
//! and tokenizer. Place the `model.onnx` and `tokenizer.json` files in a
//! directory and set it as `model_dir` in
//! [`VoyageNanoConfig`](config::VoyageNanoConfig).

pub mod backend;
pub mod config;
pub mod engine;
pub mod error;
