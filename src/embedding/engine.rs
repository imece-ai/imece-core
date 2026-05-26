//! # Voyage-4 Nano Embedding Engine
//!
//! Local ONNX Runtime-based inference engine for the Voyage-4 Nano
//! embedding model. Implements:
//!
//! - **Matryoshka Representation Learning (MRL)**: Truncates output from
//!   the native 2048 dimensions down to the configured dimension (default 256)
//!   without retrieval quality loss, leveraging the model's MRL training.
//! - **Native int8 quantization**: Leverages the model's quantization-aware
//!   training (QAT) to output int8 vectors directly, reducing storage by 4×.
//!
//! ## Architecture
//!
//! ```text
//! Input Text
//!   │
//!   ├─ Prepend task prompt (query / document)
//!   │
//!   ├─ Tokenize (HuggingFace tokenizers, Rust-native)
//!   │
//!   ├─ ONNX Runtime inference → pooler_output ∈ ℝ^2048
//!   │
//!   ├─ MRL Truncation → ℝ^256 (first 256 dims)
//!   │
//!   ├─ L2 Normalization
//!   │
//!   └─ int8 Quantization (QAT) → ℤ^256 ∈ [-128, 127]
//! ```
//!
//! ## Task Prompts
//!
//! The Voyage-4 model family uses task-specific prompt prefixes:
//! - **Query**: `"Represent the query for retrieving supporting documents: "`
//! - **Document**: `"Represent the document for retrieval: "`

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::info;

use super::config::{EmbeddingConfig, OutputPrecision};
use super::error::{EmbeddingError, EmbeddingResult};

// ---------------------------------------------------------------------------
// Task Prompt Constants
// ---------------------------------------------------------------------------

/// Query prompt prefix as defined by the Voyage-4 model specification.
const QUERY_PROMPT: &str = "Represent the query for retrieving supporting documents: ";

/// Document prompt prefix as defined by the Voyage-4 model specification.
const DOCUMENT_PROMPT: &str = "Represent the document for retrieval: ";

// ---------------------------------------------------------------------------
// VoyageNanoEngine
// ---------------------------------------------------------------------------

/// The Voyage-4 Nano local embedding engine.
///
/// Loads the ONNX-exported model and HuggingFace tokenizer from a local
/// directory. Provides methods to embed queries and documents with
/// configurable MRL dimensionality and int8 quantization.
///
/// # Thread Safety
///
/// The underlying ONNX Runtime session handles thread safety internally
/// via its intra-op thread pool. The tokenizer is immutable after loading.
pub struct VoyageNanoEngine {
    /// ONNX Runtime inference session.
    session: Mutex<Session>,
    /// HuggingFace fast tokenizer.
    tokenizer: Tokenizer,
    /// Engine configuration (MRL dim, precision, etc.).
    config: EmbeddingConfig,
}

impl VoyageNanoEngine {
    // ── Construction ──────────────────────────────────────────────────────

    /// Initialize the engine by loading the ONNX model and tokenizer
    /// from the configured `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns `ModelNotFound` if the model or tokenizer files are missing,
    /// or `OrtError` if ONNX Runtime initialization fails.
    pub fn new(config: EmbeddingConfig) -> EmbeddingResult<Self> {
        let model_path = format!("{}/model.onnx", config.model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", config.model_dir);

        // Validate paths.
        if !std::path::Path::new(&model_path).exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: model_path,
                reason: "Run `python models/export_voyage_nano.py` to download the model."
                    .to_string(),
            });
        }
        if !std::path::Path::new(&tokenizer_path).exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: tokenizer_path,
                reason: "Run `python models/export_voyage_nano.py` to download the tokenizer."
                    .to_string(),
            });
        }

        // Initialize ONNX Runtime session.
        let session = Session::builder()
            .map_err(|e| EmbeddingError::OrtError(e.to_string()))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| EmbeddingError::OrtError(e.to_string()))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| EmbeddingError::OrtError(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbeddingError::OrtError(e.to_string()))?;

        // Load tokenizer.
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbeddingError::TokenizerError(e.to_string()))?;

        info!(
            "VoyageNanoEngine initialized: model='{}', mrl_dim={}, precision={:?}, threads={}",
            model_path,
            config.mrl_dimension.as_usize(),
            config.output_precision,
            config.num_threads,
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
        })
    }

    // ── Public Embedding API ─────────────────────────────────────────────

    /// Embed a query string for retrieval.
    ///
    /// Automatically prepends the query task prompt before encoding.
    /// Returns an `Array1<i8>` (256-d int8) or `Array1<f32>` depending
    /// on the configured `OutputPrecision`.
    pub fn embed_query(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        let prompted = format!("{}{}", QUERY_PROMPT, text);
        self.embed_raw(&prompted)
    }

    /// Embed a document string for indexing.
    ///
    /// Automatically prepends the document task prompt before encoding.
    pub fn embed_document(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        let prompted = format!("{}{}", DOCUMENT_PROMPT, text);
        self.embed_raw(&prompted)
    }

    /// Embed a batch of query strings.
    pub fn embed_query_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<EmbeddingOutput>> {
        texts.iter().map(|t| self.embed_query(t)).collect()
    }

    /// Embed a batch of document strings.
    pub fn embed_document_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<EmbeddingOutput>> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }

    /// Get the configured output dimensionality.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.config.mrl_dimension.as_usize()
    }

    /// Get the configured output precision.
    #[inline]
    pub fn output_precision(&self) -> OutputPrecision {
        self.config.output_precision
    }

    /// Get a reference to the current configuration.
    #[inline]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    // ── Internal Pipeline ────────────────────────────────────────────────

    /// Core embedding pipeline: tokenize → infer → MRL truncate → normalize → quantize.
    fn embed_raw(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        // Step 1: Tokenize.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::TokenizerError(e.to_string()))?;

        let token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate to max_length if needed.
        let seq_len = token_ids.len().min(self.config.max_length);
        let token_ids = &token_ids[..seq_len];
        let attention_mask = &attention_mask[..seq_len];

        // Step 2: Prepare ONNX inputs as i64 flat vectors with shape.
        let input_ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let attn_mask: Vec<i64> = attention_mask.iter().map(|&m| m as i64).collect();

        // Step 3: Create ONNX tensors from (shape, data) tuples.
        // This avoids the ndarray version conflict (ort uses 0.17, project uses 0.16).
        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))
            .map_err(|e| EmbeddingError::OrtError(format!("Input tensor error: {}", e)))?;
        let attn_mask_tensor = Tensor::from_array(([1usize, seq_len], attn_mask))
            .map_err(|e| EmbeddingError::OrtError(format!("Mask tensor error: {}", e)))?;

        // Step 4: Run ONNX inference and extract outputs inside the lock scope.
        let full_embedding = {
            let mut session_guard = self.session.lock().unwrap();
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attn_mask_tensor,
                ])
                .map_err(|e| EmbeddingError::OrtError(format!("Inference error: {}", e)))?;

            // The ONNX community model outputs:
            //   [0] last_hidden_state: (batch, seq_len, 2048)
            //   [1] pooler_output:     (batch, 2048)       — mean-pooled
            //
            // We prefer pooler_output for efficiency. If unavailable, fall back
            // to manual mean pooling over last_hidden_state.
            if outputs.len() > 1 {
                // Use pre-computed pooler_output — shape: [1, 2048].
                let (shape, data) = outputs[1].try_extract_tensor::<f32>().map_err(|e| {
                    EmbeddingError::OrtError(format!("Pooler extract error: {}", e))
                })?;
                let shape_dims = shape.iter().copied().collect::<Vec<i64>>();
                let hidden_dim = if shape_dims.len() >= 2 {
                    shape_dims[1] as usize
                } else {
                    data.len()
                };
                data[..hidden_dim].to_vec()
            } else {
                // Fallback: manual mean pooling over last_hidden_state.
                // Shape: [1, seq_len, hidden_dim]
                let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                    EmbeddingError::OrtError(format!("Hidden extract error: {}", e))
                })?;
                let shape_dims = shape.iter().copied().collect::<Vec<i64>>();
                let hidden_dim = shape_dims[2] as usize;
                let mut pooled = vec![0.0f32; hidden_dim];
                let mut mask_sum = 0.0f32;

                for t in 0..seq_len {
                    let mask_val = attention_mask[t] as f32;
                    mask_sum += mask_val;
                    let offset = t * hidden_dim;
                    for d in 0..hidden_dim {
                        pooled[d] += data[offset + d] * mask_val;
                    }
                }

                if mask_sum > f32::EPSILON {
                    for d in pooled.iter_mut() {
                        *d /= mask_sum;
                    }
                }
                pooled
            }
        };

        // Step 5: MRL Truncation — take the first `mrl_dim` dimensions.
        let mrl_dim = self.config.mrl_dimension.as_usize();
        if full_embedding.len() < mrl_dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: mrl_dim,
                got: full_embedding.len(),
            });
        }
        let truncated: Vec<f32> = full_embedding[..mrl_dim].to_vec();

        // Step 6: L2 Normalization.
        let normalized = l2_normalize(&truncated);

        // Step 7: Output based on configured precision.
        match self.config.output_precision {
            OutputPrecision::Float32 => Ok(EmbeddingOutput::Float32(Array1::from_vec(normalized))),
            OutputPrecision::Int8 => {
                let quantized = quantize_int8(&normalized);
                Ok(EmbeddingOutput::Int8(Array1::from_vec(quantized)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingOutput
// ---------------------------------------------------------------------------

/// Output of the embedding engine, typed by the configured precision.
///
/// The `to_f32()` method provides a universal conversion path for
/// compatibility with the existing `MemoryNode` and `MemoryStore`
/// infrastructure, which operates on `Array1<f32>`.
#[derive(Debug, Clone)]
pub enum EmbeddingOutput {
    /// Standard 32-bit float embedding (L2-normalized).
    Float32(Array1<f32>),
    /// Quantized signed 8-bit integer embedding.
    /// Values ∈ [-128, 127], scaled from the L2-normalized float vector.
    Int8(Array1<i8>),
}

impl EmbeddingOutput {
    /// Convert to `Array1<f32>` regardless of the underlying precision.
    ///
    /// For `Int8` outputs, this rescales from `[-128, 127]` back to
    /// approximate float values by dividing by 127.0. The result is
    /// suitable for cosine similarity computation in the memory subsystem.
    pub fn to_f32(&self) -> Array1<f32> {
        match self {
            EmbeddingOutput::Float32(arr) => arr.clone(),
            EmbeddingOutput::Int8(arr) => {
                Array1::from_vec(arr.iter().map(|&v| v as f32 / 127.0).collect())
            }
        }
    }

    /// Get the dimensionality of the embedding vector.
    pub fn dim(&self) -> usize {
        match self {
            EmbeddingOutput::Float32(arr) => arr.len(),
            EmbeddingOutput::Int8(arr) => arr.len(),
        }
    }

    /// Extract the raw int8 vector (panics if precision is Float32).
    pub fn as_int8(&self) -> &Array1<i8> {
        match self {
            EmbeddingOutput::Int8(arr) => arr,
            _ => panic!("EmbeddingOutput is Float32, not Int8"),
        }
    }

    /// Extract the raw float32 vector (panics if precision is Int8).
    pub fn as_float32(&self) -> &Array1<f32> {
        match self {
            EmbeddingOutput::Float32(arr) => arr,
            _ => panic!("EmbeddingOutput is Int8, not Float32"),
        }
    }
}

// ---------------------------------------------------------------------------
// Math Utilities
// ---------------------------------------------------------------------------

/// L2-normalize a vector: `v̂ = v / ‖v‖₂`.
///
/// Returns the zero vector if the input has zero magnitude.
#[inline]
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Quantize an L2-normalized float vector to signed int8.
///
/// Uses the standard QAT quantization scheme:
///   `int8_val = clamp(round(float_val × 127), -128, 127)`
///
/// This is the same method used by sentence-transformers for models
/// trained with quantization-aware training.
#[inline]
fn quantize_int8(v: &[f32]) -> Vec<i8> {
    v.iter()
        .map(|&x| {
            let scaled = (x * 127.0).round();
            scaled.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_quantize_int8() {
        // A unit vector component of 1.0 should map to 127.
        let v = vec![1.0, -1.0, 0.0, 0.5];
        let q = quantize_int8(&v);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
        assert_eq!(q[2], 0);
        assert_eq!(q[3], 64); // round(0.5 * 127) = round(63.5) = 64
    }

    #[test]
    fn test_quantize_int8_clamp() {
        // Values beyond [-1, 1] should be clamped.
        let v = vec![2.0, -2.0];
        let q = quantize_int8(&v);
        assert_eq!(q[0], 127); // clamped from 254
        assert_eq!(q[1], -128); // clamped from -254
    }

    #[test]
    fn test_embedding_output_to_f32_float32() {
        let arr = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let output = EmbeddingOutput::Float32(arr.clone());
        let converted = output.to_f32();
        assert_eq!(converted, arr);
    }

    #[test]
    fn test_embedding_output_to_f32_int8() {
        let arr = Array1::from_vec(vec![127i8, -127, 0, 64]);
        let output = EmbeddingOutput::Int8(arr);
        let converted = output.to_f32();
        assert!((converted[0] - 1.0).abs() < 1e-6);
        assert!((converted[1] + 1.0).abs() < 1e-6);
        assert!(converted[2].abs() < 1e-6);
    }

    #[test]
    fn test_embedding_output_dim() {
        let float_out = EmbeddingOutput::Float32(Array1::from_vec(vec![0.0; 256]));
        assert_eq!(float_out.dim(), 256);

        let int8_out = EmbeddingOutput::Int8(Array1::from_vec(vec![0i8; 256]));
        assert_eq!(int8_out.dim(), 256);
    }
}
