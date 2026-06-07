//! # Embedding Configuration
//!
//! Configuration types for the pluggable embedding subsystem.
//!
//! ## Architecture
//!
//! [`EmbeddingServiceConfig`] is an internally-tagged enum (`#[serde(tag = "type")]`)
//! that provides compile-time type safety for backend-specific configuration.
//! Each variant wraps a fully-typed config struct for its backend, enabling
//! direct deserialization without opaque intermediate types.
//!
//! ## Defaults
//!
//! The default configuration targets the IMECE low-VRAM edge profile
//! using the Voyage-4 Nano backend:
//! - **MRL Dimension**: 256 (minimum supported, maximum storage efficiency)
//! - **Output Precision**: `Int8` (leveraging quantization-aware training)
//! - **Threads**: 4 (matches typical edge-device core count)
//! - **Max Length**: 512 tokens (sufficient for memory node payloads)

use serde::{Deserialize, Serialize};

use super::backend::EmbeddingBackend;
use super::error::EmbeddingResult;

// ---------------------------------------------------------------------------
// EmbeddingServiceConfig
// ---------------------------------------------------------------------------

/// Top-level embedding configuration.
///
/// Each variant fully specifies the configuration for a particular backend.
/// The `#[serde(tag = "type")]` attribute enables type-safe deserialization
/// from JSON, TOML, or any serde-supported format:
///
/// ```json
/// {
///   "type": "voyage_nano",
///   "model_dir": "models/voyage-4-nano-onnx",
///   "mrl_dimension": "D256",
///   "output_precision": "Int8",
///   "num_threads": 4,
///   "max_length": 512
/// }
/// ```
///
/// ## Adding a New Backend
///
/// 1. Define a `YourBackendConfig` struct with `Serialize + Deserialize`.
/// 2. Add a variant to this enum: `YourBackend(YourBackendConfig)`.
/// 3. Extend [`create_backend`](EmbeddingServiceConfig::create_backend) to construct it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EmbeddingServiceConfig {
    /// Voyage-4 Nano local ONNX backend.
    #[serde(rename = "voyage_nano")]
    VoyageNano(VoyageNanoConfig),
}

impl Default for EmbeddingServiceConfig {
    /// Default: Voyage-4 Nano with edge-optimized settings.
    fn default() -> Self {
        EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig::default())
    }
}

impl EmbeddingServiceConfig {
    /// Instantiate the embedding backend described by this configuration.
    ///
    /// Returns a trait object that can be used interchangeably regardless
    /// of the underlying backend engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot be initialized (e.g., model
    /// files missing, invalid configuration).
    pub fn create_backend(&self) -> EmbeddingResult<Box<dyn EmbeddingBackend>> {
        match self {
            EmbeddingServiceConfig::VoyageNano(cfg) => {
                let engine = super::engine::VoyageNanoEngine::new(cfg.clone())?;
                Ok(Box::new(engine))
            }
        }
    }
}

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
// VoyageNanoConfig
// ---------------------------------------------------------------------------

/// Configuration for the Voyage-4 Nano local embedding engine.
///
/// # Example
/// ```rust,no_run
/// use imece_core::embedding::config::{VoyageNanoConfig, MrlDimension, OutputPrecision};
///
/// let config = VoyageNanoConfig {
///     model_dir: "models/voyage-4-nano-onnx".into(),
///     mrl_dimension: MrlDimension::D256,
///     output_precision: OutputPrecision::Int8,
///     num_threads: 4,
///     max_length: 512,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoyageNanoConfig {
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

impl Default for VoyageNanoConfig {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_config_serde_roundtrip() {
        let config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
            model_dir: "models/test".to_string(),
            mrl_dimension: MrlDimension::D512,
            output_precision: OutputPrecision::Float32,
            num_threads: 8,
            max_length: 256,
        });

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"voyage_nano\""));

        let deserialized: EmbeddingServiceConfig = serde_json::from_str(&json).unwrap();
        match deserialized {
            EmbeddingServiceConfig::VoyageNano(cfg) => {
                assert_eq!(cfg.model_dir, "models/test");
                assert_eq!(cfg.mrl_dimension, MrlDimension::D512);
                assert_eq!(cfg.output_precision, OutputPrecision::Float32);
                assert_eq!(cfg.num_threads, 8);
                assert_eq!(cfg.max_length, 256);
            }
        }
    }

    #[test]
    fn test_service_config_default() {
        let config = EmbeddingServiceConfig::default();
        match config {
            EmbeddingServiceConfig::VoyageNano(cfg) => {
                assert_eq!(cfg.mrl_dimension, MrlDimension::D256);
                assert_eq!(cfg.output_precision, OutputPrecision::Int8);
                assert_eq!(cfg.num_threads, 4);
                assert_eq!(cfg.max_length, 512);
            }
        }
    }

    #[test]
    fn test_mrl_dimension_as_usize() {
        assert_eq!(MrlDimension::D2048.as_usize(), 2048);
        assert_eq!(MrlDimension::D1024.as_usize(), 1024);
        assert_eq!(MrlDimension::D512.as_usize(), 512);
        assert_eq!(MrlDimension::D256.as_usize(), 256);
    }

    #[test]
    fn test_deserialize_from_json_string() {
        let json = r#"{
            "type": "voyage_nano",
            "model_dir": "models/v4",
            "mrl_dimension": "D1024",
            "output_precision": "Int8",
            "num_threads": 2,
            "max_length": 128
        }"#;

        let config: EmbeddingServiceConfig = serde_json::from_str(json).unwrap();
        match config {
            EmbeddingServiceConfig::VoyageNano(cfg) => {
                assert_eq!(cfg.model_dir, "models/v4");
                assert_eq!(cfg.mrl_dimension, MrlDimension::D1024);
                assert_eq!(cfg.num_threads, 2);
            }
        }
    }
}
