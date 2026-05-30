//! # Embedding Subsystem (Module 4)
//!
//! Local-first embedding engine powered by the **Voyage-4 Nano** model,
//! running entirely on-device via ONNX Runtime. Replaces the previous
//! manual vector provisioning with dynamic text-to-embedding inference.
//!
//! ## Architecture
//!
//! The Voyage-4 Nano model (180M non-embedding + 160M embedding parameters)
//! is exported to ONNX format and loaded by the Rust-native `ort` crate.
//! Tokenization uses the HuggingFace `tokenizers` crate (also Rust-native),
//! ensuring zero Python dependency at runtime.
//!
//! ## Key Optimizations
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
//! - [`config`]: Configuration types (MRL dimension, output precision).
//! - [`engine`]: The `VoyageNanoEngine` ONNX inference pipeline.
//! - [`error`]: Embedding-specific error types.
//!
//! ## Model Setup
//!
//! Before using the engine, you must provide the ONNX model and tokenizer.
//! Place the `model.onnx` and `tokenizer.json` files in a directory and
//! pass its path to `EmbeddingConfig::model_dir`.

pub mod config;
pub mod engine;
pub mod error;
