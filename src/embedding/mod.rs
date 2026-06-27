//! # Embedding Subsystem (Module 4)
//!
//! A completely pluggable, model-agnostic local embedding engine built
//! around the [`EmbeddingBackend`](backend::EmbeddingBackend) trait.
//!
//! This module allows you to generate dense vector embeddings locally
//! without relying on external APIs. You can implement the trait for
//! any model architecture, precision, or inference backend.
//!
//! ## Pluggable Architecture
//!
//! 1. **[`EmbeddingBackend`](backend::EmbeddingBackend)** — Core trait
//!    that all backends implement.
//! 2. **[`EmbeddingServiceConfig`](config::EmbeddingServiceConfig)** —
//!    Type-safe tagged enum for backend-specific configuration.
//! 3. **[`EmbeddingOutput`](backend::EmbeddingOutput)** — Precision-typed
//!    output supporting both Float32 and Int8.
//!
//! ## Default Shipped Backend: Voyage-4 Nano
//!
//! While the architecture is model-agnostic, IMECE ships with a highly
//! optimized default backend: [`VoyageNanoEngine`](engine::VoyageNanoEngine).
//! This engine runs the Voyage-4 Nano model entirely on-device via ONNX
//! Runtime and features two major optimizations for edge devices:
//!
//! 1. **Matryoshka Representation Learning (MRL)**:
//!    The output can be seamlessly truncated from 2048 down to 256 dimensions
//!    with minimal retrieval quality loss, reducing vector storage by 8×.
//!
//! 2. **Native int8 Quantization (QAT)**:
//!    The model uses quantization-aware training to output native int8
//!    vectors, further reducing storage by 4× compared to float32.
//!
//! Combined, these optimizations provide a 32× reduction in memory footprint
//! while maintaining strong semantic retrieval performance.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use imece_core::embedding::config::EmbeddingServiceConfig;
//!
//! // Using the default shipped backend (Voyage-4 Nano)
//! let config = EmbeddingServiceConfig::default();
//! let backend = config.create_backend().unwrap();
//!
//! let embedding = backend.embed_query("What is IMECE?").unwrap();
//! let vector = embedding.to_f32(); // Array1<f32> for memory subsystem
//! ```
//!
//! ## Submodules
//! - [`backend`]: [`EmbeddingBackend`](backend::EmbeddingBackend) trait,
//!   [`EmbeddingOutput`](backend::EmbeddingOutput), and shared math utilities.
//! - [`config`]: Configuration types and backend factory.
//! - [`engine`]: [`VoyageNanoEngine`](engine::VoyageNanoEngine) — Voyage-4 Nano
//!   ONNX inference backend.
//! - [`error`]: Embedding-specific error types.

pub mod backend;
pub mod config;
pub mod engine;
pub mod error;
