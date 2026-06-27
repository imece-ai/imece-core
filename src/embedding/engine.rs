//! # Voyage-4 Nano Embedding Engine
//!
//! Local ONNX Runtime-based inference engine for the Voyage-4 Nano
//! embedding model. Implements the [`EmbeddingBackend`] trait with:
//!
//! - **Matryoshka Representation Learning (MRL)**: Truncates output from
//!   the native 2048 dimensions down to the configured dimension (default 256)
//!   without retrieval quality loss, leveraging the model's MRL training.
//! - **Native int8 quantization**: Leverages the model's quantization-aware
//!   training (QAT) to output int8 vectors directly, reducing storage by 4×.
//!
//! ## Pipeline
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
//!   ├─ MRL Truncation → ℝ^D (first D dims)
//!   │
//!   ├─ L2 Normalization
//!   │
//!   └─ Optional int8 Quantization (QAT) → ℤ^D ∈ [-128, 127]
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

use super::backend::{l2_normalize, quantize_int8, EmbeddingBackend, EmbeddingOutput};
use super::config::{OutputPrecision, VoyageNanoConfig};
use super::error::{EmbeddingError, EmbeddingResult};

// ---------------------------------------------------------------------------
// Task Prompt Constants
// ---------------------------------------------------------------------------

/// Query prompt prefix as defined by the Voyage-4 model specification.
const QUERY_PROMPT: &str = "Represent the query for retrieving supporting documents: ";

/// Document prompt prefix as defined by the Voyage-4 model specification.
const DOCUMENT_PROMPT: &str = "Represent the document for retrieval: ";

/// Backend identifier for error reporting and logging.
const BACKEND_NAME: &str = "voyage-4-nano";

// ---------------------------------------------------------------------------
// VoyageNanoEngine
// ---------------------------------------------------------------------------

/// # Voyage-4 Nano Engine
///
/// The default shipped local inference backend for the embedding subsystem,
/// powered by [ONNX Runtime](https://onnxruntime.ai/).
///
/// This engine executes the Voyage-4 Nano model entirely on-device. It
/// automatically manages the tokenization pipeline (via HuggingFace
/// `tokenizers`) and handles the tensor operations required for Matryoshka
/// Truncation and int8 Quantization.
///
/// ## Optimizations
///
/// 1. **MRL Truncation**: Extracts the first `D` dimensions of the 2048-d
///    output vector and re-normalizes them.
/// 2. **Int8 Quantization**: Converts float32 output into `i8` arrays,
///    mapping the `-1.0` to `1.0` range into `-127` to `127`.
///
/// ## ONNX Initialization
///
/// The `ort` crate requires the ONNX Runtime environment to be initialized
/// exactly once per process. This engine handles initialization automatically
/// during `VoyageNanoEngine::new()`.
///
/// ## Usage via Config
///
/// It is recommended to instantiate this engine via the
/// [`EmbeddingServiceConfig::create_backend`](super::config::EmbeddingServiceConfig::create_backend)
/// factory rather than calling `VoyageNanoEngine::new()` directly.
pub struct VoyageNanoEngine {
    /// ONNX Runtime inference session, mutex-guarded because
    /// `Session::run` requires `&mut self`.
    session: Mutex<Session>,
    /// HuggingFace fast tokenizer.
    tokenizer: Tokenizer,
    /// Engine configuration (MRL dim, precision, etc.).
    config: VoyageNanoConfig,
}

impl VoyageNanoEngine {
    // ── Construction ──────────────────────────────────────────────────────

    /// Initialize the engine by loading the ONNX model and tokenizer
    /// from the configured `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns `ModelNotFound` if the model or tokenizer files are missing,
    /// or `BackendError` if ONNX Runtime initialization fails.
    pub fn new(config: VoyageNanoConfig) -> EmbeddingResult<Self> {
        let model_path = format!("{}/model.onnx", config.model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", config.model_dir);

        // Validate paths.
        if !std::path::Path::new(&model_path).exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: model_path,
                reason: "Ensure the ONNX model file exists at the specified path."
                    .to_string(),
            });
        }
        if !std::path::Path::new(&tokenizer_path).exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: tokenizer_path,
                reason: "Ensure the tokenizer.json file exists at the specified path."
                    .to_string(),
            });
        }

        // Initialize ONNX Runtime session.
        let session = Session::builder()
            .map_err(|e| ort_err(e))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| ort_err(e))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| ort_err(e))?
            .commit_from_file(&model_path)
            .map_err(|e| ort_err(e))?;

        // Load tokenizer.
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| tokenizer_err(e))?;

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

    /// Get a reference to the current configuration.
    #[inline]
    pub fn config(&self) -> &VoyageNanoConfig {
        &self.config
    }

    // ── Internal Pipeline ────────────────────────────────────────────────

    /// Core embedding pipeline: tokenize → infer → MRL truncate → normalize → quantize.
    fn embed_raw(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        // Step 1: Tokenize.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| tokenizer_err(e))?;

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
            .map_err(|e| ort_err(format!("Input tensor error: {}", e)))?;
        let attn_mask_tensor = Tensor::from_array(([1usize, seq_len], attn_mask))
            .map_err(|e| ort_err(format!("Mask tensor error: {}", e)))?;

        // Step 4: Run ONNX inference and extract outputs inside the lock scope.
        let full_embedding = {
            let mut session_guard = self.session.lock().unwrap();
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attn_mask_tensor,
                ])
                .map_err(|e| ort_err(format!("Inference error: {}", e)))?;

            // The ONNX community model outputs:
            //   [0] last_hidden_state: (batch, seq_len, 2048)
            //   [1] pooler_output:     (batch, 2048)       — mean-pooled
            //
            // We prefer pooler_output for efficiency. If unavailable, fall back
            // to manual mean pooling over last_hidden_state.
            if outputs.len() > 1 {
                // Use pre-computed pooler_output — shape: [1, 2048].
                let (shape, data) = outputs[1].try_extract_tensor::<f32>().map_err(|e| {
                    ort_err(format!("Pooler extract error: {}", e))
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
                    ort_err(format!("Hidden extract error: {}", e))
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
// EmbeddingBackend Implementation
// ---------------------------------------------------------------------------

impl EmbeddingBackend for VoyageNanoEngine {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn dimension(&self) -> usize {
        self.config.mrl_dimension.as_usize()
    }

    fn embed_query(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        let prompted = format!("{}{}", QUERY_PROMPT, text);
        self.embed_raw(&prompted)
    }

    fn embed_document(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
        let prompted = format!("{}{}", DOCUMENT_PROMPT, text);
        self.embed_raw(&prompted)
    }
}

// ---------------------------------------------------------------------------
// Backend Error Helpers
// ---------------------------------------------------------------------------

/// Wrap an ORT-originated error as a backend-specific `EmbeddingError`.
fn ort_err(e: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::BackendError {
        backend: BACKEND_NAME.to_string(),
        source: format!("ONNX Runtime: {e}").into(),
    }
}

/// Wrap a tokenizer-originated error as a backend-specific `EmbeddingError`.
fn tokenizer_err(e: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::BackendError {
        backend: BACKEND_NAME.to_string(),
        source: format!("Tokenizer: {e}").into(),
    }
}
