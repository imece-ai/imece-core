//! # llama.cpp C-API FFI Bindings
//!
//! Raw `extern "C"` declarations mirroring the llama.cpp public C API.
//! These are the actual function signatures from `llama.h` — no wrappers,
//! no abstraction layers, no mocks.
//!
//! ## Linking
//!
//! The consumer must link against a compiled `libllama`:
//!
//! ```toml
//! # In your .cargo/config.toml:
//! [build]
//! rustflags = ["-L", "/path/to/llama.cpp/build", "-l", "llama"]
//! ```
//!
//! Or set `LLAMA_LIB_PATH` environment variable and use the `build.rs` script.

use std::os::raw::{c_char, c_float, c_int};

// ---------------------------------------------------------------------------
// Opaque Types
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded llama model (`struct llama_model *`).
#[repr(C)]
pub struct LlamaModel {
    _opaque: [u8; 0],
}

/// Opaque handle to a llama context (`struct llama_context *`).
#[repr(C)]
pub struct LlamaContext {
    _opaque: [u8; 0],
}

/// Opaque handle to a llama sampler chain (`struct llama_sampler *`).
#[repr(C)]
pub struct LlamaSampler {
    _opaque: [u8; 0],
}

/// Opaque handle to a llama memory state (`llama_memory_t`).
#[repr(C)]
pub struct LlamaMemory {
    _opaque: [u8; 0],
}

/// Opaque handle to a llama vocab (`struct llama_vocab *`).
#[repr(C)]
pub struct LlamaVocab {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Primitive Type Aliases (from llama.h)
// ---------------------------------------------------------------------------

/// Token ID — vocabulary index.
pub type LlamaToken = c_int;

/// Position in the KV-Cache timeline.
pub type LlamaPos = c_int;

/// Sequence ID for multi-sequence batching.
pub type LlamaSeqId = c_int;

// ---------------------------------------------------------------------------
// Parameter Structs
// ---------------------------------------------------------------------------

/// Default model loading parameters. Mirrors `struct llama_model_params`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LlamaModelParams {
    pub devices: *mut std::ffi::c_void,
    pub tensor_buft_overrides: *mut std::ffi::c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *mut std::ffi::c_void,
    pub progress_callback_user_data: *mut std::ffi::c_void,
    pub kv_overrides: *mut std::ffi::c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

/// Default context parameters. Mirrors `struct llama_context_params`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *mut std::ffi::c_void,
    pub cb_eval_user_data: *mut std::ffi::c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *mut std::ffi::c_void,
    pub abort_callback_data: *mut std::ffi::c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut std::ffi::c_void,
    pub n_samplers: usize,
}

/// Token data for sampling. Mirrors `struct llama_token_data`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaTokenData {
    /// Token ID.
    pub id: LlamaToken,
    /// Log-odds (logit) of this token.
    pub logit: c_float,
    /// Probability (after softmax).
    pub p: c_float,
}

/// Array of token data for sampling. Mirrors `struct llama_token_data_array`.
#[repr(C)]
#[derive(Debug)]
pub struct LlamaTokenDataArray {
    /// Pointer to the token data array.
    pub data: *mut LlamaTokenData,
    /// Number of elements.
    pub size: usize,
    /// Index of the selected token (set by sampler).
    pub selected: i64,
    /// Whether the array is sorted by logit descending.
    pub sorted: bool,
}

/// Batch of tokens for evaluation. Mirrors `struct llama_batch`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaBatch {
    /// Number of tokens in this batch.
    pub n_tokens: c_int,

    /// Token IDs (array of n_tokens).
    pub token: *mut LlamaToken,
    /// Embedding data (NULL if using token IDs).
    pub embd: *mut c_float,
    /// Position of each token in the sequence.
    pub pos: *mut LlamaPos,
    /// Number of sequence IDs per token.
    pub n_seq_id: *mut c_int,
    /// Sequence ID arrays (one per token).
    pub seq_id: *mut *mut LlamaSeqId,
    /// Per-token logit output flag (1 = compute logits for this token).
    pub logits: *mut i8,
}

// ---------------------------------------------------------------------------
// External C-API Functions
// ---------------------------------------------------------------------------

extern "C" {
    // ── Backend Lifecycle ─────────────────────────────────────────────

    /// Initialize the llama.cpp backend (call once at startup).
    pub fn llama_backend_init();

    /// Free the llama.cpp backend resources (call once at shutdown).
    pub fn llama_backend_free();

    // ── Model Defaults ───────────────────────────────────────────────

    /// Get default model loading parameters.
    pub fn llama_model_default_params() -> LlamaModelParams;

    /// Get default context parameters.
    pub fn llama_context_default_params() -> LlamaContextParams;

    // ── Model Lifecycle ──────────────────────────────────────────────

    /// Load a GGUF model from file.
    pub fn llama_load_model_from_file(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;

    /// Create a new context from a loaded model.
    pub fn llama_new_context_with_model(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext;

    /// Free a context.
    pub fn llama_free(ctx: *mut LlamaContext);

    /// Free a model.
    pub fn llama_free_model(model: *mut LlamaModel);

    // ── Vocab ────────────────────────────────────────────────────────

    /// Get the vocabulary initialized with the model.
    pub fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;

    // ── Tokenization ─────────────────────────────────────────────────

    /// Tokenize a UTF-8 string.
    ///
    /// Returns the number of tokens written. If the buffer is too small,
    /// returns the negated required size.
    pub fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut LlamaToken,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    /// Convert a token ID to its UTF-8 text piece.
    ///
    /// Returns the number of bytes written.
    pub fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    /// Get the End-Of-Sequence token ID.
    pub fn llama_token_eos(vocab: *const LlamaVocab) -> LlamaToken;

    /// Get the Beginning-Of-Sequence token ID.
    pub fn llama_token_bos(vocab: *const LlamaVocab) -> LlamaToken;

    /// Get the vocabulary size.
    pub fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> c_int;

    /// Get the context size (n_ctx) from the model's training config.
    pub fn llama_model_n_ctx_train(model: *const LlamaModel) -> c_int;

    // ── Batch Management ─────────────────────────────────────────────

    /// Allocate a batch with capacity for `n_tokens`.
    pub fn llama_batch_init(n_tokens: c_int, embd: c_int, n_seq_max: c_int) -> LlamaBatch;

    /// Free a batch.
    pub fn llama_batch_free(batch: LlamaBatch);

    // ── Decode (Inference) ───────────────────────────────────────────

    /// Decode (evaluate) a batch of tokens.
    ///
    /// Returns 0 on success, negative on error.
    /// This populates the KV-Cache for the given positions.
    pub fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;

    /// Get pointer to the logits array (vocab_size floats) after decode.
    /// The logits correspond to the last token for which `logits[i] = 1` was set.
    pub fn llama_get_logits(ctx: *mut LlamaContext) -> *mut c_float;

    /// Get pointer to logits for a specific token index in the batch.
    pub fn llama_get_logits_ith(ctx: *mut LlamaContext, i: c_int) -> *mut c_float;

    // ── KV-Cache Operations (Now Memory Operations) ──────────────────
    // These are the core functions for the "Time Travel" rollback protocol.

    /// Remove KV-Cache entries for sequence `seq_id` in positions `[p0, p1)`.
    ///
    /// This is the heart of the rollback mechanism:
    ///   `llama_memory_seq_rm(mem, seq_id, p0, p1)`
    ///
    /// Setting `p0 = -1` means from the beginning.
    /// Setting `p1 = -1` means until the end.
    ///
    /// Returns `true` on success.
    pub fn llama_memory_seq_rm(
        mem: *mut LlamaMemory,
        seq_id: LlamaSeqId,
        p0: LlamaPos,
        p1: LlamaPos,
    ) -> bool;

    /// Clear the entire KV-Cache (all sequences, all positions).
    pub fn llama_memory_clear(mem: *mut LlamaMemory);

    /// Copy KV-Cache from one sequence to another.
    pub fn llama_memory_seq_cp(
        mem: *mut LlamaMemory,
        seq_id_src: LlamaSeqId,
        seq_id_dst: LlamaSeqId,
        p0: LlamaPos,
        p1: LlamaPos,
    );

    /// Shift all positions in a sequence's KV-Cache by `delta`.
    pub fn llama_memory_seq_add(
        mem: *mut LlamaMemory,
        seq_id: LlamaSeqId,
        p0: LlamaPos,
        p1: LlamaPos,
        delta: LlamaPos,
    );

    /// Get the memory state from the context.
    pub fn llama_get_memory(ctx: *const LlamaContext) -> *mut LlamaMemory;

    // ── Context State ────────────────────────────────────────────────

    /// Get the actual context size currently used.
    pub fn llama_n_ctx(ctx: *const LlamaContext) -> u32;

    /// Get the model from a context.
    pub fn llama_get_model(ctx: *const LlamaContext) -> *const LlamaModel;

    // ── Sampler Chain API ────────────────────────────────────────────

    /// Create a sampler chain with defaults.
    pub fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;

    /// Add a sampler to the chain.
    pub fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);

    /// Sample a token from the logits using the sampler chain.
    pub fn llama_sampler_sample(
        smpl: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: c_int,
    ) -> LlamaToken;

    /// Reset the sampler chain state.
    pub fn llama_sampler_reset(smpl: *mut LlamaSampler);

    /// Free a sampler chain.
    pub fn llama_sampler_free(smpl: *mut LlamaSampler);

    // ── Built-in Samplers ────────────────────────────────────────────

    /// Temperature sampler.
    pub fn llama_sampler_init_temp(temp: c_float) -> *mut LlamaSampler;

    /// Top-K sampler.
    pub fn llama_sampler_init_top_k(k: c_int) -> *mut LlamaSampler;

    /// Top-P (nucleus) sampler.
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut LlamaSampler;

    /// Greedy (argmax) sampler.
    pub fn llama_sampler_init_greedy() -> *mut LlamaSampler;

    /// Distribution (random) sampler.
    pub fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
}

// ---------------------------------------------------------------------------
// Sampler Chain Params
// ---------------------------------------------------------------------------

/// Parameters for `llama_sampler_chain_init`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaSamplerChainParams {
    /// Disable performance counters.
    pub no_perf: bool,
}

impl Default for LlamaSamplerChainParams {
    fn default() -> Self {
        Self { no_perf: false }
    }
}
