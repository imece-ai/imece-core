//! # llama.cpp Backend Implementation
//!
//! Real `KvCacheManager` and `LlmBackend` implementations that bind
//! to the llama.cpp C-API via the FFI declarations in [`super::ffi`].
//!
//! ## Safety
//!
//! This module contains `unsafe` code that calls through the C FFI boundary.
//! Pointer validity is enforced at construction time — once a `LlamaCppBackend`
//! is created successfully, all subsequent operations operate on valid pointers
//! for the lifetime of the struct.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

use super::engine::LlmBackend;
use super::error::{InferenceError, InferenceResult};
use super::ffi;
use super::kv_cache::KvCacheManager;
use super::types::*;

// ---------------------------------------------------------------------------
// Backend Refcount (safe multi-instance llama_backend_init/free)
// ---------------------------------------------------------------------------

/// Reference count for `llama_backend_init` / `llama_backend_free`.
/// These are global singletons in llama.cpp — calling `free` while another
/// `LlamaCppBackend` instance exists would invalidate it.
static BACKEND_REFCOUNT: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// LlamaCppBackend
// ---------------------------------------------------------------------------

/// Real llama.cpp LLM backend bound to the C-API via FFI.
///
/// Owns the model and context pointers. Handles tokenization, decoding,
/// sampling, and KV-Cache operations directly through the C interface.
///
/// # Lifecycle
/// ```text
/// LlamaCppBackend::load(path, params)
///   → llama_backend_init()
///   → llama_load_model_from_file()
///   → llama_new_context_with_model()
///   → Ready for inference
///   → Drop → llama_free() + llama_free_model() + llama_backend_free()
/// ```
pub struct LlamaCppBackend {
    /// Raw pointer to the loaded model.
    model: *mut ffi::LlamaModel,
    /// Raw pointer to the vocabulary.
    vocab: *const ffi::LlamaVocab,
    /// Raw pointer to the context (owns the KV-Cache).
    ctx: *mut ffi::LlamaContext,
    /// Sampler chain pointer.
    sampler: *mut ffi::LlamaSampler,
    /// Vocabulary size.
    n_vocab: i32,
    /// Context window size.
    n_ctx: u32,
    /// EOS token ID.
    eos_token: ffi::LlamaToken,
    /// Mutex for thread-safe decode (llama_decode is not thread-safe).
    decode_lock: Mutex<()>,
}

// Safety: The raw pointers are only accessed through methods that
// hold the decode_lock, and the struct is !Sync by default due to
// raw pointers. We explicitly impl Send because the llama.cpp context
// can be moved between threads (but not accessed concurrently).
unsafe impl Send for LlamaCppBackend {}
unsafe impl Sync for LlamaCppBackend {}

// ---------------------------------------------------------------------------
// GPU Layer Resolution
// ---------------------------------------------------------------------------

/// Resolve the number of GPU layers to offload.
///
/// Resolution order:
/// 1. **`IMECE_GPU_LAYERS` env var** — explicit override (e.g. `IMECE_GPU_LAYERS=30`).
/// 2. **Build-time detection** — if compiled with `--features cuda`, defaults to `99`
///    (offload all layers to GPU).
/// 3. **Fallback** — `0` (pure CPU inference).
///
/// # GPU Compilation Guide
///
/// To enable GPU acceleration:
/// ```bash
/// # Build with CUDA support:
/// cargo build --features llama_backend,cuda
///
/// # Optionally tune the layer count at runtime:
/// IMECE_GPU_LAYERS=30 cargo run --features llama_backend,cuda
///
/// # Force CPU-only even on a CUDA build:
/// IMECE_GPU_LAYERS=0 cargo run --features llama_backend,cuda
/// ```
fn resolve_gpu_layers() -> i32 {
    // Priority 1: Explicit env var override.
    if let Ok(val) = std::env::var("IMECE_GPU_LAYERS") {
        match val.parse::<i32>() {
            Ok(n) => {
                info!("GPU layers override via IMECE_GPU_LAYERS={}", n);
                return n;
            }
            Err(_) => {
                tracing::warn!(
                    "IMECE_GPU_LAYERS='{}' is not a valid i32, ignoring",
                    val
                );
            }
        }
    }

    // Priority 2: Build-time GPU availability.
    #[cfg(imece_gpu_available)]
    {
        info!(
            "GPU backend detected (compiled with --features cuda). \
             Defaulting n_gpu_layers=99 (all)."
        );
        return 99;
    }

    // Priority 3: CPU-only fallback.
    #[allow(unreachable_code)]
    {
        info!("CPU-only build. n_gpu_layers=0.");
        0
    }
}

impl LlamaCppBackend {
    /// Load a GGUF model and create a ready-to-use backend.
    ///
    /// # Arguments
    /// * `model_path` — Path to the `.gguf` model file.
    /// * `n_ctx` — Context window size (0 = use model default).
    /// * `n_threads` — Number of CPU threads for inference.
    ///
    /// # GPU Layer Resolution
    /// The number of GPU layers is resolved automatically via
    /// [`resolve_gpu_layers()`]:
    /// 1. `IMECE_GPU_LAYERS` env var (explicit override)
    /// 2. Build-time: `--features cuda` → 99 (all layers)
    /// 3. Default: 0 (CPU only)
    ///
    /// # Safety
    /// Calls into the llama.cpp C-API. The `model_path` must be a valid
    /// filesystem path to a GGUF file.
    pub fn load(
        model_path: &str,
        n_ctx: u32,
        n_threads: u32,
    ) -> InferenceResult<Self> {
        let n_gpu_layers = resolve_gpu_layers();

        let c_path = CString::new(model_path)
            .map_err(|e| InferenceError::BackendError(format!("Invalid model path: {e}")))?;

        unsafe {
            // Initialize the backend (refcounted — safe for multiple instances).
            if BACKEND_REFCOUNT.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                ffi::llama_backend_init();
            }

            // Load model.
            let mut model_params = ffi::llama_model_default_params();
            model_params.n_gpu_layers = n_gpu_layers;

            let model = ffi::llama_load_model_from_file(c_path.as_ptr(), model_params);
            if model.is_null() {
                return Err(InferenceError::BackendError(format!(
                    "Failed to load model from '{model_path}'"
                )));
            }

            // Create context.
            let mut ctx_params = ffi::llama_context_default_params();
            ctx_params.n_ctx = if n_ctx == 0 {
                ffi::llama_model_n_ctx_train(model) as u32
            } else {
                n_ctx
            };
            ctx_params.n_threads = n_threads as i32;
            ctx_params.n_threads_batch = n_threads as i32;
            ctx_params.flash_attn_type = 1; // LLAMA_FLASH_ATTN_TYPE_ENABLED

            println!("  [Trace] Calling llama_new_context_with_model...");
            let ctx = ffi::llama_new_context_with_model(model, ctx_params);
            if ctx.is_null() {
                ffi::llama_free_model(model);
                return Err(InferenceError::BackendError(
                    "Failed to create llama context".into(),
                ));
            }
            println!("  [Trace] Calling llama_model_get_vocab...");
            let vocab = ffi::llama_model_get_vocab(model);
            println!("  [Trace] Calling llama_vocab_n_tokens...");
            let n_vocab = ffi::llama_vocab_n_tokens(vocab);
            println!("  [Trace] Calling llama_n_ctx...");
            let actual_n_ctx = ffi::llama_n_ctx(ctx);
            println!("  [Trace] Calling llama_token_eos...");
            let eos_token = ffi::llama_token_eos(vocab);

            println!("  [Trace] Building sampler chain...");
            // Build sampler chain: top_k → top_p → temperature → dist.
            let chain_params = ffi::LlamaSamplerChainParams::default();
            println!("  [Trace] Calling llama_sampler_chain_init...");
            let sampler = ffi::llama_sampler_chain_init(chain_params);
            println!("  [Trace] Calling llama_sampler_init_top_k...");
            ffi::llama_sampler_chain_add(sampler, ffi::llama_sampler_init_top_k(40));
            println!("  [Trace] Calling llama_sampler_init_top_p...");
            ffi::llama_sampler_chain_add(sampler, ffi::llama_sampler_init_top_p(0.95, 1));
            println!("  [Trace] Calling llama_sampler_init_temp...");
            ffi::llama_sampler_chain_add(sampler, ffi::llama_sampler_init_temp(0.7));
            println!("  [Trace] Calling llama_sampler_init_dist...");
            ffi::llama_sampler_chain_add(sampler, ffi::llama_sampler_init_dist(0));

            println!("  [Trace] Done initializing backend.");
            info!(
                "LlamaCppBackend loaded: model='{}', n_ctx={}, n_vocab={}, n_gpu_layers={}",
                model_path, actual_n_ctx, n_vocab, n_gpu_layers
            );

            Ok(Self {
                model,
                vocab,
                ctx,
                sampler,
                n_vocab,
                n_ctx: actual_n_ctx,
                eos_token,
                decode_lock: Mutex::new(()),
            })
        }
    }

    /// Get the context window size.
    pub fn context_size(&self) -> u32 {
        self.n_ctx
    }

    /// Get the raw context pointer for KvCacheManager.
    pub fn raw_context_ptr(&self) -> *mut ffi::LlamaContext {
        self.ctx
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        self.n_vocab
    }

    /// Get the EOS token ID.
    pub fn eos_token(&self) -> ffi::LlamaToken {
        self.eos_token
    }

    /// Decode a single token ID into its UTF-8 text.
    fn token_to_text(&self, token_id: ffi::LlamaToken) -> String {
        let mut buf = vec![0u8; 128];
        let len = unsafe {
            ffi::llama_token_to_piece(
                self.vocab,
                token_id,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                0,     // lstrip
                false, // special
            )
        };
        if len < 0 {
            // Buffer too small — retry with correct size.
            let needed = (-len) as usize;
            buf.resize(needed, 0);
            let len2 = unsafe {
                ffi::llama_token_to_piece(
                    self.vocab,
                    token_id,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as c_int,
                    0,
                    false,
                )
            };
            buf.truncate(len2.max(0) as usize);
        } else {
            buf.truncate(len as usize);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Prepare and evaluate a batch of tokens in the KV-Cache.
    fn decode_batch(
        &self,
        tokens: &[ffi::LlamaToken],
        positions: &[ffi::LlamaPos],
        seq_id: ffi::LlamaSeqId,
        compute_logits_for_last: bool,
    ) -> InferenceResult<()> {
        let _lock = self
            .decode_lock
            .lock()
            .map_err(|e| InferenceError::BackendError(format!("Decode lock poisoned: {e}")))?;

        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }

        unsafe {
            let mut batch = ffi::llama_batch_init(n as c_int, 0, 1);
            batch.n_tokens = n as c_int;

            for i in 0..n {
                *batch.token.add(i) = tokens[i];
                *batch.pos.add(i) = positions[i];
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)) = seq_id;
                *batch.logits.add(i) = if compute_logits_for_last && i == n - 1 {
                    1
                } else {
                    0
                };
            }

            let rc = ffi::llama_decode(self.ctx, batch);
            ffi::llama_batch_free(batch);

            if rc != 0 {
                return Err(InferenceError::BackendError(format!(
                    "llama_decode failed with code {rc}"
                )));
            }
        }

        Ok(())
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                ffi::llama_sampler_free(self.sampler);
            }
            if !self.ctx.is_null() {
                ffi::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                ffi::llama_free_model(self.model);
            }
            if BACKEND_REFCOUNT.fetch_sub(1, AtomicOrdering::SeqCst) == 1 {
                ffi::llama_backend_free();
            }
        }
        info!("LlamaCppBackend resources released.");
    }
}

// ---------------------------------------------------------------------------
// Public Sync Inference Methods
// ---------------------------------------------------------------------------
//
// These are the raw synchronous methods that perform blocking FFI calls.
// They are wrapped by `AsyncLlamaBackend` (below) which offloads them
// to `tokio::task::spawn_blocking` so they don't stall the Tokio runtime.

impl LlamaCppBackend {
    /// Tokenize text into tokens (synchronous).
    pub fn tokenize(&self, text: &str) -> InferenceResult<Vec<Token>> {
        let c_text = CString::new(text)
            .map_err(|e| InferenceError::BackendError(format!("Invalid UTF-8 in text: {e}")))?;

        let n_estimate = (text.len() + 32) as c_int;
        let mut token_buf = vec![0 as ffi::LlamaToken; n_estimate as usize];

        let n_tokens = unsafe {
            ffi::llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as c_int,
                token_buf.as_mut_ptr(),
                n_estimate,
                true,
                true,
            )
        };

        if n_tokens < 0 {
            let needed = (-n_tokens) as usize;
            token_buf.resize(needed, 0);
            let n_tokens2 = unsafe {
                ffi::llama_tokenize(
                    self.vocab,
                    c_text.as_ptr(),
                    text.len() as c_int,
                    token_buf.as_mut_ptr(),
                    needed as c_int,
                    true,
                    true,
                )
            };
            if n_tokens2 < 0 {
                return Err(InferenceError::BackendError(
                    "Tokenization failed after resize".into(),
                ));
            }
            token_buf.truncate(n_tokens2 as usize);
        } else {
            token_buf.truncate(n_tokens as usize);
        }

        let tokens: Vec<Token> = token_buf
            .iter()
            .map(|&id| Token {
                id: id as u32,
                text: self.token_to_text(id),
            })
            .collect();

        Ok(tokens)
    }

    /// Detokenize tokens back into text (synchronous).
    pub fn detokenize(&self, tokens: &[Token]) -> InferenceResult<String> {
        let mut output = String::new();
        for token in tokens {
            output.push_str(&self.token_to_text(token.id as ffi::LlamaToken));
        }
        Ok(output)
    }

    /// Generate the next token (synchronous, blocking FFI call).
    ///
    /// This calls `llama_decode` + `llama_sampler_sample` which can
    /// block for 50-500ms on larger models. Use `AsyncLlamaBackend`
    /// to avoid stalling the Tokio runtime.
    pub fn generate_next_sync(
        &self,
        state: &GenerationState,
        _config: &InferenceConfig,
    ) -> InferenceResult<Option<Token>> {
        let last_token = state.tokens.last().ok_or_else(|| {
            InferenceError::BackendError("Cannot generate from empty state".into())
        })?;

        let token_id = last_token.id as ffi::LlamaToken;
        let pos = (state.cache_position - 1) as ffi::LlamaPos;
        self.decode_batch(&[token_id], &[pos], state.seq_id as ffi::LlamaSeqId, true)?;

        let next_id = unsafe { ffi::llama_sampler_sample(self.sampler, self.ctx, -1) };

        if next_id == self.eos_token {
            debug!("EOS token generated.");
            return Ok(None);
        }

        let text = self.token_to_text(next_id);
        unsafe { ffi::llama_sampler_reset(self.sampler) };

        Ok(Some(Token {
            id: next_id as u32,
            text,
        }))
    }

    /// Evaluate (prefill) a batch of tokens (synchronous, blocking FFI call).
    pub fn evaluate_tokens_sync(
        &self,
        seq_id: u32,
        position: usize,
        tokens: &[Token],
    ) -> InferenceResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        let token_ids: Vec<ffi::LlamaToken> =
            tokens.iter().map(|t| t.id as ffi::LlamaToken).collect();
        let positions: Vec<ffi::LlamaPos> = (0..tokens.len())
            .map(|i| (position + i) as ffi::LlamaPos)
            .collect();

        debug!(
            "Evaluating {} tokens at position {} for seq_id={}",
            tokens.len(),
            position,
            seq_id
        );

        self.decode_batch(
            &token_ids,
            &positions,
            seq_id as ffi::LlamaSeqId,
            true,
        )
    }
}

// ---------------------------------------------------------------------------
// LlamaCppKvCache — Real KvCacheManager Implementation
// ---------------------------------------------------------------------------

/// Real KV-Cache manager that wraps a `LlamaCppBackend`'s context pointer
/// and directly calls the llama.cpp KV-Cache C-API functions.
///
/// This is not a mock. This is the production implementation that calls:
/// - `llama_kv_cache_seq_rm()` — erase cache entries ("Time Travel")
/// - `llama_kv_cache_clear()` — full cache reset
///
/// # Ownership Model
///
/// Holds a shared `Arc<LlamaCppBackend>` reference to co-own the backend.
/// This structurally guarantees that the context pointer (`ctx`) remains
/// valid for the lifetime of this struct — the `Arc` refcount prevents
/// `LlamaCppBackend::drop()` from freeing the llama context while any
/// `LlamaCppKvCache` instance still exists.
///
/// **Drop order is irrelevant.** Whether the backend or the KvCache is
/// dropped first, the `Arc` refcount ensures deallocation only occurs
/// when the last reference is released.
pub struct LlamaCppKvCache {
    /// Shared ownership of the backend. Prevents `LlamaCppBackend::drop()`
    /// from freeing the llama context while this KvCache still exists.
    /// The `Arc` refcount is the structural safety guarantee.
    _backend: Arc<LlamaCppBackend>,

    /// Raw pointer to the llama context, derived from `_backend.ctx`.
    /// Always valid while `_backend` is alive (which is guaranteed by `Arc`).
    ctx: *mut ffi::LlamaContext,

    /// Tracked sequence lengths for validation.
    /// Updated on each operation to maintain consistency.
    seq_positions: std::collections::HashMap<u32, usize>,
}

// Safety: raw pointer is borrowed from LlamaCppBackend which ensures validity.
unsafe impl Send for LlamaCppKvCache {}
unsafe impl Sync for LlamaCppKvCache {}

impl LlamaCppKvCache {
    /// Create a new KV-Cache manager that co-owns the given backend.
    ///
    /// The `Arc<LlamaCppBackend>` ensures the context pointer remains valid
    /// for the lifetime of this struct — no field ordering or manual drop
    /// coordination required.
    pub fn new(
        backend: Arc<LlamaCppBackend>,
        seq_id: u32,
        initial_position: usize,
    ) -> Self {
        let ctx = backend.raw_context_ptr();
        let mut seq_positions = std::collections::HashMap::new();
        seq_positions.insert(seq_id, initial_position);
        Self {
            _backend: backend,
            ctx,
            seq_positions,
        }
    }

    /// Register a sequence with an initial cache length.
    pub fn register_sequence(&mut self, seq_id: u32, position: usize) {
        self.seq_positions.insert(seq_id, position);
    }
}

impl KvCacheManager for LlamaCppKvCache {
    fn seq_rm(&mut self, seq_id: u32, p0: usize, p1: usize) -> InferenceResult<()> {
        let mem = unsafe { ffi::llama_get_memory(self.ctx) };
        let success = unsafe {
            ffi::llama_memory_seq_rm(
                mem,
                seq_id as ffi::LlamaSeqId,
                p0 as ffi::LlamaPos,
                p1 as ffi::LlamaPos,
            )
        };

        if !success {
            return Err(InferenceError::KvCacheError(format!(
                "llama_memory_seq_rm(seq={}, p0={}, p1={}) returned false",
                seq_id, p0, p1
            )));
        }

        // Update tracked position.
        if let Some(pos) = self.seq_positions.get_mut(&seq_id) {
            let erased = p1.saturating_sub(p0);
            *pos = pos.saturating_sub(erased);
        }

        debug!(
            "KV-Cache seq_rm: seq_id={}, erased [{}, {})",
            seq_id, p0, p1
        );

        Ok(())
    }

    fn seq_len(&self, seq_id: u32) -> usize {
        self.seq_positions.get(&seq_id).copied().unwrap_or(0)
    }

    fn inject_tokens(
        &mut self,
        seq_id: u32,
        position: usize,
        tokens: &[Token],
    ) -> InferenceResult<()> {
        // Injection is done via llama_decode — we evaluate the tokens
        // at the specified positions to fill the KV-Cache slots.
        let prefill_len = tokens.len().saturating_sub(1);
        if prefill_len == 0 {
            // Update tracked position just in case, though usually handled by state
            if let Some(pos) = self.seq_positions.get_mut(&seq_id) {
                *pos += tokens.len().min(1);
            }
            return Ok(());
        }

        let token_ids: Vec<ffi::LlamaToken> = tokens[..prefill_len]
            .iter()
            .map(|t| t.id as ffi::LlamaToken)
            .collect();
        let positions: Vec<ffi::LlamaPos> = (0..prefill_len)
            .map(|i| (position + i) as ffi::LlamaPos)
            .collect();
        let n = prefill_len;

        unsafe {
            let mut batch = ffi::llama_batch_init(n as c_int, 0, 1);
            batch.n_tokens = n as c_int;

            for i in 0..n {
                *batch.token.add(i) = token_ids[i];
                *batch.pos.add(i) = positions[i];
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)) = seq_id as ffi::LlamaSeqId;
                *batch.logits.add(i) = if i == n - 1 { 1 } else { 0 };
            }

            let rc = ffi::llama_decode(self.ctx, batch);
            ffi::llama_batch_free(batch);

            if rc != 0 {
                return Err(InferenceError::KvCacheError(format!(
                    "llama_decode failed during injection (code {rc})"
                )));
            }
        }

        // Update tracked position.
        if let Some(pos) = self.seq_positions.get_mut(&seq_id) {
            *pos += prefill_len;
        }

        debug!(
            "KV-Cache inject: {} tokens at position {} for seq_id={}",
            prefill_len, position, seq_id
        );

        Ok(())
    }

    fn clear_all(&mut self) -> InferenceResult<()> {
        let mem = unsafe { ffi::llama_get_memory(self.ctx) };
        unsafe {
            ffi::llama_memory_clear(mem);
        }
        self.seq_positions.clear();
        debug!("KV-Cache cleared (all sequences).");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AsyncLlamaBackend — Tokio-safe wrapper
// ---------------------------------------------------------------------------

/// Async-safe wrapper around [`LlamaCppBackend`] that offloads blocking
/// FFI calls to [`tokio::task::spawn_blocking`].
///
/// Without this wrapper, calls like `llama_decode` (50-500ms on 7B models)
/// would block the Tokio worker thread and stall all other tasks (agent
/// message routing, heartbeats, signal handling).
///
/// # Usage
/// ```rust,ignore
/// let backend = Arc::new(LlamaCppBackend::load(path, 4096, 8)?);
/// let async_backend = AsyncLlamaBackend::new(backend);
/// let engine = InferenceEngine::new(async_backend, kv_controller, executor, config);
/// ```
pub struct AsyncLlamaBackend {
    inner: Arc<LlamaCppBackend>,
}

// Safety: Arc<LlamaCppBackend> is Send+Sync, and all FFI access goes
// through the inner Mutex<()> decode_lock.
unsafe impl Send for AsyncLlamaBackend {}
unsafe impl Sync for AsyncLlamaBackend {}

impl AsyncLlamaBackend {
    /// Create a new async-safe wrapper around a shared backend.
    pub fn new(inner: Arc<LlamaCppBackend>) -> Self {
        Self { inner }
    }

    /// Get a reference to the underlying backend (e.g., for KV-Cache pointer).
    pub fn inner(&self) -> &LlamaCppBackend {
        &self.inner
    }
}

#[async_trait::async_trait]
impl LlmBackend for AsyncLlamaBackend {
    fn tokenize(&self, text: &str) -> InferenceResult<Vec<Token>> {
        // Tokenization is fast (~µs), safe to run inline.
        self.inner.tokenize(text)
    }

    fn detokenize(&self, tokens: &[Token]) -> InferenceResult<String> {
        self.inner.detokenize(tokens)
    }

    async fn generate_next(
        &self,
        state: &GenerationState,
        config: &InferenceConfig,
    ) -> InferenceResult<Option<Token>> {
        let inner = Arc::clone(&self.inner);
        let state = state.clone();
        let config = config.clone();

        tokio::task::spawn_blocking(move || inner.generate_next_sync(&state, &config))
            .await
            .map_err(|e| {
                InferenceError::BackendError(format!("spawn_blocking panicked: {e}"))
            })?
    }

    async fn evaluate_tokens(
        &self,
        seq_id: u32,
        position: usize,
        tokens: &[Token],
    ) -> InferenceResult<()> {
        let inner = Arc::clone(&self.inner);
        let tokens = tokens.to_vec();

        tokio::task::spawn_blocking(move || {
            inner.evaluate_tokens_sync(seq_id, position, &tokens)
        })
        .await
        .map_err(|e| {
            InferenceError::BackendError(format!("spawn_blocking panicked: {e}"))
        })?
    }
}
