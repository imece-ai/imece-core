//! # Embedding Configuration
//!
//! Configuration types for the Voyage-4 Nano embedding engine.
//!
//! ## Defaults
//!
//! The default configuration targets the IMECE low-VRAM edge profile:
//! - **MRL Dimension**: 256 (minimum supported, maximum storage efficiency)
//! - **Output Precision**: `Int8` (leveraging quantization-aware training)
//! - **Threads**: 4 (matches typical edge-device core count)
//! - **Max Length**: 512 tokens (sufficient for memory node payloads)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MRL Dimension
// ---------------------------------------------------------------------------

/// Supported Matryoshka Representation Learning (MRL) output dimensions.
///
/// Voyage-4 Nano is trained with MRL to support flexible embedding
/// dimensions with minimal retrieval quality loss. Lower dimensions
/// reduce storage cost and improve similarity computation throughput.
///
/// Supported values: 2048 (full), 1024, 512, 256 (minimum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MrlDimension {
    /// Full 2048-dimensional output (no truncation).
    D2048 = 2048,
    /// 1024-dimensional output.
    D1024 = 1024,
    /// 512-dimensional output.
    D512 = 512,
    /// 256-dimensional output (maximum compression, minimal quality loss).
    D256 = 256,
}

impl MrlDimension {
    /// Get the numeric dimension value.
    #[inline]
    pub fn as_usize(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// Output Precision
// ---------------------------------------------------------------------------

/// Output vector precision for the embedding model.
///
/// Voyage-4 Nano uses quantization-aware training (QAT), enabling native
/// int8 output with minimal retrieval quality degradation compared to
/// float32. This reduces memory footprint by 4× and accelerates
/// distance computations on integer-capable hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputPrecision {
    /// Standard 32-bit floating point (baseline).
    Float32,
    /// Signed 8-bit integer (QAT-optimized, 4× smaller).
    Int8,
}

// ---------------------------------------------------------------------------
// EmbeddingConfig
// ---------------------------------------------------------------------------

/// Configuration for the Voyage-4 Nano local embedding engine.
///
/// # Example
/// ```rust,no_run
/// use imece_core::embedding::config::{EmbeddingConfig, MrlDimension, OutputPrecision};
///
/// let config = EmbeddingConfig {
///     model_dir: "models/voyage-4-nano-onnx".into(),
///     mrl_dimension: MrlDimension::D256,
///     output_precision: OutputPrecision::Int8,
///     num_threads: 4,
///     max_length: 512,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Path to the directory containing the ONNX model (`model.onnx`)
    /// and tokenizer (`tokenizer.json`) files.
    pub model_dir: String,

    /// MRL truncation dimension for the output embedding.
    /// The model's native 2048-d output is truncated to this dimension.
    pub mrl_dimension: MrlDimension,

    /// Output vector precision.
    /// When set to `Int8`, the engine quantizes normalized float embeddings
    /// to signed 8-bit integers using the model's QAT-trained scale.
    pub output_precision: OutputPrecision,

    /// Number of CPU threads for ONNX Runtime intra-op parallelism.
    pub num_threads: usize,

    /// Maximum token sequence length. Inputs exceeding this are truncated.
    pub max_length: usize,
}

impl Default for EmbeddingConfig {
    /// Default configuration optimized for IMECE edge deployment:
    /// - 256-dimensional MRL output
    /// - int8 quantized vectors
    /// - 4 inference threads
    /// - 512 max token length
    fn default() -> Self {
        Self {
            model_dir: "models/voyage-4-nano-onnx".to_string(),
            mrl_dimension: MrlDimension::D256,
            output_precision: OutputPrecision::Int8,
            num_threads: 4,
            max_length: 512,
        }
    }
}
