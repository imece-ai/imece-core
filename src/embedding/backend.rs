//! # Embedding Backend Trait
//!
//! Core definitions for the pluggable embedding subsystem.
//!
//! This module defines the [`EmbeddingBackend`] trait, which abstracts
//! the underlying embedding generation logic. You can implement this trait
//! to support any local model architecture (e.g., ONNX, candle, tch-rs)
//! or even to proxy requests to an external API, though local inference
//! is the primary design goal.
//!
//! ## Example: Custom Backend
//!
//! ```rust
//! use imece_core::embedding::backend::{EmbeddingBackend, EmbeddingOutput};
//! use imece_core::embedding::error::EmbeddingResult;
//! use ndarray::Array1;
//!
//! struct MockBackend { dim: usize }
//!
//! impl EmbeddingBackend for MockBackend {
//!     fn name(&self) -> &str { "mock" }
//!
//!     fn dimension(&self) -> usize { self.dim }
//!
//!     fn embed_query(&self, _text: &str) -> EmbeddingResult<EmbeddingOutput> {
//!         Ok(EmbeddingOutput::Float32(Array1::from_vec(vec![0.1; self.dim])))
//!     }
//!
//!     fn embed_document(&self, _text: &str) -> EmbeddingResult<EmbeddingOutput> {
//!         Ok(EmbeddingOutput::Float32(Array1::from_vec(vec![0.2; self.dim])))
//!     }
//! }
//! ```
//! ## Shared Utilities
//!
//! [`l2_normalize`] and [`quantize_int8`] are provided as public helpers for
//! backend implementations that need L2 normalization or QAT-style int8
//! quantization in their post-processing pipeline.

use ndarray::Array1;

use super::error::EmbeddingResult;

// ---------------------------------------------------------------------------
// EmbeddingBackend Trait
// ---------------------------------------------------------------------------

/// A pluggable local embedding model backend.
///
/// Implementations produce dense vector embeddings from text input,
/// running entirely on-device. The trait is synchronous and requires
/// `Send + Sync` to allow shared ownership via `Arc<dyn EmbeddingBackend>`
/// across threads.
///
/// # Query vs Document
///
/// Many embedding models use different prompt prefixes for asymmetric
/// retrieval (e.g., Voyage-4, E5, nomic-embed). Models without this
/// distinction should implement both methods identically.
///
/// # Batch Processing
///
/// Default batch implementations iterate sequentially. Backends that
/// support true tensor batching (padded inputs) should override
/// [`embed_query_batch`](EmbeddingBackend::embed_query_batch) and
/// [`embed_document_batch`](EmbeddingBackend::embed_document_batch)
/// for better throughput.
pub trait EmbeddingBackend: Send + Sync {
    /// Human-readable backend identifier (e.g., `"voyage-4-nano"`).
    fn name(&self) -> &str;

    /// Output dimensionality of embeddings produced by this backend.
    fn dimension(&self) -> usize;

    /// Embed a single text for query / retrieval.
    fn embed_query(&self, text: &str) -> EmbeddingResult<EmbeddingOutput>;

    /// Embed a single text for document indexing.
    fn embed_document(&self, text: &str) -> EmbeddingResult<EmbeddingOutput>;

    /// Embed a batch of query texts.
    ///
    /// Default: sequential iteration over [`embed_query`](EmbeddingBackend::embed_query).
    fn embed_query_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<EmbeddingOutput>> {
        texts.iter().map(|t| self.embed_query(t)).collect()
    }

    /// Embed a batch of document texts.
    ///
    /// Default: sequential iteration over [`embed_document`](EmbeddingBackend::embed_document).
    fn embed_document_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<EmbeddingOutput>> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingOutput
// ---------------------------------------------------------------------------

/// Output of an embedding backend, typed by precision.
///
/// The [`to_f32()`](EmbeddingOutput::to_f32) method provides a universal
/// conversion path for compatibility with the [`MemoryNode`] and
/// [`MemoryStore`] infrastructure, which operates on `Array1<f32>`.
///
/// [`MemoryNode`]: crate::memory::node::MemoryNode
/// [`MemoryStore`]: crate::memory::store::MemoryStore
#[derive(Debug, Clone)]
pub enum EmbeddingOutput {
    /// Standard 32-bit float embedding (L2-normalized).
    Float32(Array1<f32>),
    /// Quantized signed 8-bit integer embedding.
    /// Values ∈ \[-128, 127\], scaled from the L2-normalized float vector.
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
/// Available to all backend implementations for post-processing.
#[inline]
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
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
pub fn quantize_int8(v: &[f32]) -> Vec<i8> {
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

    // ── Test-only stub backend ───────────────────────────────────────────

    /// A trivial backend that returns constant-valued embeddings.
    /// Exists solely to verify trait-object dispatch and default batch
    /// implementations — NOT a production backend.
    struct ConstantEmbeddingBackend {
        dim: usize,
        value: f32,
    }

    impl EmbeddingBackend for ConstantEmbeddingBackend {
        fn name(&self) -> &str {
            "constant-test"
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn embed_query(&self, _text: &str) -> EmbeddingResult<EmbeddingOutput> {
            Ok(EmbeddingOutput::Float32(Array1::from_elem(self.dim, self.value)))
        }

        fn embed_document(&self, text: &str) -> EmbeddingResult<EmbeddingOutput> {
            self.embed_query(text)
        }
    }

    // ── Trait dispatch tests ─────────────────────────────────────────────

    #[test]
    fn test_trait_object_dispatch() {
        let backend: Box<dyn EmbeddingBackend> = Box::new(ConstantEmbeddingBackend {
            dim: 256,
            value: 0.5,
        });
        assert_eq!(backend.name(), "constant-test");
        assert_eq!(backend.dimension(), 256);

        let output = backend.embed_query("test").unwrap();
        assert_eq!(output.dim(), 256);
        assert!((output.to_f32()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_default_batch_impl() {
        let backend: Box<dyn EmbeddingBackend> = Box::new(ConstantEmbeddingBackend {
            dim: 128,
            value: 1.0,
        });
        let results = backend.embed_query_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.dim(), 128);
        }

        let doc_results = backend.embed_document_batch(&["x", "y"]).unwrap();
        assert_eq!(doc_results.len(), 2);
    }

    #[test]
    fn test_arc_shared_backend() {
        use std::sync::Arc;

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(ConstantEmbeddingBackend {
            dim: 64,
            value: 0.25,
        });

        let b1 = Arc::clone(&backend);
        let b2 = Arc::clone(&backend);

        // Both references can call methods — proves Send + Sync bounds.
        assert_eq!(b1.dimension(), b2.dimension());
        let o1 = b1.embed_query("a").unwrap();
        let o2 = b2.embed_document("b").unwrap();
        assert_eq!(o1.dim(), o2.dim());
    }

    // ── Math utility tests ───────────────────────────────────────────────

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

    // ── EmbeddingOutput tests ────────────────────────────────────────────

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
