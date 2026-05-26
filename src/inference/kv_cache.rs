//! # KV-Cache Manipulation Layer
//!
//! Abstracts `llama.cpp`'s KV-Cache operations behind a Rust trait.
//!
//! ## The "Time Travel" Protocol
//!
//! When an action execution fails:
//! 1. Identify token `t_k` where the logical error began.
//! 2. Call `kv_cache_seq_rm(seq_id, t_k, t_n)` → erase cache from `t_k` to end.
//! 3. Inject `Observation: <Error Log>` directly at `t_k`.
//! 4. Resume autoregressive generation.
//!
//! **Result:** The LLM experiences this as "thinking mid-sentence, realizing
//! a mistake, and correcting it instantly." Zero context bloat. Zero
//! prompt-recalculation (Prefill) overhead.
//!
//! ## Design
//!
//! The `KvCacheManager` trait is implemented by `LlamaCppKvCache` in
//! [`super::backend`], which binds directly to the llama.cpp C-API via FFI.

use tracing::debug;

use super::error::{InferenceError, InferenceResult};
use super::types::{GenerationState, RollbackTarget, Token};

// ---------------------------------------------------------------------------
// KvCacheManager Trait
// ---------------------------------------------------------------------------

/// Abstraction over the llama.cpp KV-Cache memory block.
///
/// This trait represents the minimal surface area needed for the
/// KV-Cache Rollback Protocol. The production implementation
/// `LlamaCppKvCache` in [`super::backend`] binds to the llama.cpp C-API.
pub trait KvCacheManager: Send + Sync {
    /// Erase KV-Cache entries for `seq_id` from position `p0` to `p1` (exclusive).
    ///
    /// Maps directly to: `llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)`
    ///
    /// # Errors
    /// Returns `KvCacheError` if the backend operation fails.
    fn seq_rm(&mut self, seq_id: u32, p0: usize, p1: usize) -> InferenceResult<()>;

    /// Get the current cache length for a given sequence.
    fn seq_len(&self, seq_id: u32) -> usize;

    /// Inject tokens at a specific position in the KV-Cache.
    ///
    /// This is used to insert the error observation after rollback.
    /// The tokens are evaluated (prefilled) at the injection position.
    fn inject_tokens(
        &mut self,
        seq_id: u32,
        position: usize,
        tokens: &[Token],
    ) -> InferenceResult<()>;

    /// Clear the entire KV-Cache (all sequences).
    fn clear_all(&mut self) -> InferenceResult<()>;
}

// ---------------------------------------------------------------------------
// KvCacheController
// ---------------------------------------------------------------------------

/// High-level controller for KV-Cache rollback operations.
///
/// Wraps a `KvCacheManager` implementation and provides the complete
/// "Time Travel" protocol as described in the IMECE spec.
pub struct KvCacheController<M: KvCacheManager> {
    /// The underlying KV-Cache backend.
    manager: M,

    /// Running count of total rollback operations (telemetry).
    pub rollback_count: u64,

    /// Running count of total tokens erased across all rollbacks.
    pub total_tokens_erased: u64,
}

impl<M: KvCacheManager> KvCacheController<M> {
    /// Create a new controller wrapping the given KV-Cache manager.
    pub fn new(manager: M) -> Self {
        Self {
            manager,
            rollback_count: 0,
            total_tokens_erased: 0,
        }
    }

    /// Execute the KV-Cache Rollback Protocol ("Time Travel").
    ///
    /// # Protocol
    /// 1. Validate that `target.from_position < target.to_position`.
    /// 2. Call `kv_cache_seq_rm(seq_id, t_k, t_n)` to erase the erroneous range.
    /// 3. Inject the error observation tokens at `t_k`.
    /// 4. Mutate the `GenerationState` to reflect the rollback.
    ///
    /// # Arguments
    /// * `state` — Mutable reference to the current generation state.
    /// * `target` — The rollback target specifying the erase range.
    /// * `observation_tokens` — Tokens to inject at the rollback point
    ///   (e.g., tokenized "Observation: SyntaxError at line 5\n").
    ///
    /// # Returns
    /// The number of cache entries erased.
    pub fn rollback(
        &mut self,
        state: &mut GenerationState,
        target: &RollbackTarget,
        observation_tokens: &[Token],
    ) -> InferenceResult<usize> {
        // Validate bounds.
        if target.from_position >= target.to_position {
            return Err(InferenceError::KvCacheError(format!(
                "Invalid rollback range: from={} >= to={}",
                target.from_position, target.to_position
            )));
        }

        let cache_len = self.manager.seq_len(target.seq_id);
        if target.to_position > cache_len {
            return Err(InferenceError::RollbackOutOfBounds {
                position: target.to_position,
                cache_len,
            });
        }

        if target.from_position < state.prompt_end {
            return Err(InferenceError::KvCacheError(format!(
                "Cannot rollback into prompt region (prompt_end={}, target={})",
                state.prompt_end, target.from_position
            )));
        }

        let erased = target.erased_count();

        debug!(
            "KV-Cache Rollback: seq_id={}, erasing [{}, {}) — {} entries",
            target.seq_id, target.from_position, target.to_position, erased
        );

        // Step 1: Erase the KV-Cache range.
        self.manager
            .seq_rm(target.seq_id, target.from_position, target.to_position)?;

        // Step 2: Truncate the generation state's token buffer.
        state.tokens.truncate(target.from_position);
        state.cache_position = target.from_position;

        // Step 3: Inject observation tokens at the rollback point.
        if !observation_tokens.is_empty() {
            debug!(
                "Injecting {} observation tokens at position {}",
                observation_tokens.len(),
                target.from_position
            );

            self.manager
                .inject_tokens(target.seq_id, target.from_position, observation_tokens)?;

            // Append observation tokens to the state.
            for token in observation_tokens {
                state.push_token(token.clone());
            }
        }

        // Update telemetry.
        self.rollback_count += 1;
        self.total_tokens_erased += erased as u64;

        debug!(
            "Rollback complete. New cache_position={}, total_rollbacks={}",
            state.cache_position, self.rollback_count
        );

        Ok(erased)
    }

    /// Check if a rollback is safe to perform.
    ///
    /// Returns `None` if safe, or `Some(error)` describing why it's not.
    pub fn validate_rollback(
        &self,
        state: &GenerationState,
        target: &RollbackTarget,
    ) -> Option<InferenceError> {
        if target.from_position >= target.to_position {
            return Some(InferenceError::KvCacheError("from >= to".into()));
        }

        let cache_len = self.manager.seq_len(target.seq_id);
        if target.to_position > cache_len {
            return Some(InferenceError::RollbackOutOfBounds {
                position: target.to_position,
                cache_len,
            });
        }

        if target.from_position < state.prompt_end {
            return Some(InferenceError::KvCacheError(format!(
                "Would rollback into prompt (prompt_end={})",
                state.prompt_end
            )));
        }

        None
    }

    /// Get a reference to the underlying KV-Cache manager.
    pub fn manager(&self) -> &M {
        &self.manager
    }

    /// Get a mutable reference to the underlying KV-Cache manager.
    pub fn manager_mut(&mut self) -> &mut M {
        &mut self.manager
    }
}
