//! # Inference Engine — The Execution Sandbox Loop
//!
//! Orchestrates the full inference pipeline:
//!
//! ```text
//! ┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌──────────────┐
//! │   GENERATE   │────▶│ INTERCEPT │────▶│ EXECUTE  │────▶│  EVALUATE    │
//! │ (tokens)     │     │ (stop seq)│     │ (sandbox)│     │ (success/err)│
//! └──────────────┘     └───────────┘     └──────────┘     └──┬───────────┘
//!       ▲                                                     │
//!       │                  ┌───────────────────┐              │
//!       └──────────────────│  KV-CACHE ROLLBACK │◀────────────┘
//!                          │  ("Time Travel")   │   (on error)
//!                          └───────────────────┘
//! ```
//!
//! The engine wraps a `KvCacheController` and an `ActionExecutor`,
//! orchestrating generation, interception, execution, and rollback
//! in a tight loop. On the real backend, this binds to llama.cpp's
//! C-API for generation; here we define the protocol abstraction.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, info, warn};

use super::error::{InferenceError, InferenceResult};
use super::executor::{ActionExecutor, ActionParser};
use super::kv_cache::{KvCacheController, KvCacheManager};
use super::types::*;

// ---------------------------------------------------------------------------
// LlmBackend Trait
// ---------------------------------------------------------------------------

/// Abstraction over the LLM generation backend (llama.cpp C-API).
///
/// This trait encapsulates the minimal operations needed for
/// autoregressive generation, token-by-token.
#[async_trait::async_trait]
pub trait LlmBackend: Send + Sync {
    /// Tokenize text into tokens.
    fn tokenize(&self, text: &str) -> InferenceResult<Vec<Token>>;

    /// Detokenize tokens back into text.
    fn detokenize(&self, tokens: &[Token]) -> InferenceResult<String>;

    /// Generate the next token given the current state.
    ///
    /// The backend should use the KV-Cache state — it does NOT need
    /// to re-evaluate the entire sequence, just the last token position.
    ///
    /// # Arguments
    /// * `state` — Current generation state (token history + cache position).
    /// * `config` — Sampling parameters (temperature, top_k, top_p).
    ///
    /// # Returns
    /// The next `Token`, or `None` if EOS.
    async fn generate_next(
        &self,
        state: &GenerationState,
        config: &InferenceConfig,
    ) -> InferenceResult<Option<Token>>;

    /// Evaluate (prefill) a batch of tokens starting at a given position.
    ///
    /// Used after rollback to inject observation tokens into the KV-Cache.
    async fn evaluate_tokens(
        &self,
        seq_id: u32,
        position: usize,
        tokens: &[Token],
    ) -> InferenceResult<()>;
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// The main inference engine orchestrating generation, execution, and rollback.
///
/// This is the central coordinator described in Module 2 of the IMECE spec:
/// - Generates tokens via the LLM backend.
/// - Intercepts stop sequences (action boundaries).
/// - Routes actions to the sandboxed executor.
/// - On failure, triggers KV-Cache rollback and resumes.
pub struct InferenceEngine<B, M, E>
where
    B: LlmBackend,
    M: KvCacheManager,
    E: ActionExecutor,
{
    /// The LLM generation backend.
    backend: B,

    /// KV-Cache controller for rollback operations.
    kv_cache: KvCacheController<M>,

    /// Sandboxed action executor.
    executor: E,

    /// Engine configuration.
    config: InferenceConfig,
}

impl<B, M, E> InferenceEngine<B, M, E>
where
    B: LlmBackend,
    M: KvCacheManager,
    E: ActionExecutor,
{
    /// Create a new inference engine.
    pub fn new(
        backend: B,
        kv_cache: KvCacheController<M>,
        executor: E,
        config: InferenceConfig,
    ) -> Self {
        Self {
            backend,
            kv_cache,
            executor,
            config,
        }
    }

    /// Get a reference to the LLM backend.
    ///
    /// Used by `InferenceAgent::handle_escalation` for isolated
    /// tokenization and generation on a separate `seq_id`.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Clean up KV-Cache entries for an isolated sequence.
    ///
    /// Called after an escalation request completes to free the
    /// temporary cache entries. Delegates to
    /// `kv_cache.manager_mut().seq_rm(seq_id, 0, end_position)`.
    ///
    /// # Arguments
    /// * `seq_id` — The isolated sequence ID to clean up.
    /// * `end_position` — The cache position up to which to erase.
    pub fn cleanup_seq(&mut self, seq_id: u32, end_position: usize) -> InferenceResult<()> {
        if end_position == 0 {
            return Ok(()); // Nothing to clean up.
        }
        self.kv_cache.manager_mut().seq_rm(seq_id, 0, end_position)
    }

    /// Run the full inference pipeline for a given prompt.
    ///
    /// Convenience wrapper around [`run_streaming`] with no token callback
    /// and no cancellation. See that method for full documentation.
    pub async fn run(&mut self, prompt: &str) -> InferenceResult<InferenceSession> {
        let no_cancel = AtomicBool::new(false);
        self.run_streaming(prompt, &no_cancel, |_| {}).await
    }

    /// Run the full inference pipeline with streaming token output.
    ///
    /// This implements the **Execution Sandbox Loop**:
    /// 1. Tokenize and prefill the prompt.
    /// 2. Generate tokens until a stop sequence or max_tokens.
    /// 3. If an action boundary is hit, execute the action.
    /// 4. On failure, rollback and retry (up to `max_rollback_retries`).
    /// 5. On success or terminal stop, return the result.
    ///
    /// # Arguments
    /// * `prompt` — The input prompt text.
    /// * `cancel` — Checked between tokens; if `true`, generation halts.
    /// * `on_token` — Called with each generated token's text for streaming.
    ///
    /// # Returns
    /// An `InferenceSession` containing the final state and all
    /// execution events that occurred.
    pub async fn run_streaming<F>(
        &mut self,
        prompt: &str,
        cancel: &AtomicBool,
        mut on_token: F,
    ) -> InferenceResult<InferenceSession>
    where
        F: FnMut(String),
    {
        // Step 0: Tokenize the prompt.
        let prompt_tokens = self.backend.tokenize(prompt)?;
        let mut state = GenerationState::new(0, prompt_tokens);
        let mut session = InferenceSession::new();
        let mut retry_count = 0;

        info!(
            "Inference started. prompt_len={}, max_tokens={}",
            state.prompt_end, self.config.max_tokens
        );

        // Step 1: Evaluate (prefill) the prompt.
        let prefill_len = state.tokens.len().saturating_sub(1);
        if prefill_len > 0 {
            self.backend
                .evaluate_tokens(state.seq_id, 0, &state.tokens[..prefill_len])
                .await?;
        }

        // Main generation loop.
        loop {
            // Check cancellation before each generation pass.
            if cancel.load(Ordering::Relaxed) {
                info!("Inference cancelled by external signal.");
                break;
            }

            // Generate tokens until a stop sequence or max length.
            let stop_result = self
                .generate_until_stop_streaming(&mut state, cancel, &mut on_token)
                .await?;

            match stop_result {
                StopResult::ActionBoundary { generated_text } => {
                    // Extract the action payload.
                    let action = ActionParser::extract_action(&generated_text, state.prompt_end);

                    let Some(payload) = action else {
                        warn!("Stop sequence hit but no valid action block found");
                        session.events.push(SessionEvent::Warning(
                            "Action boundary without valid action block".into(),
                        ));
                        continue;
                    };

                    debug!(
                        "Action intercepted: type='{}', code_len={}",
                        payload.action_type,
                        payload.code.len()
                    );

                    // Execute the action.
                    let timeout = Duration::from_millis(self.config.execution_timeout_ms);
                    let outcome = self.executor.execute(&payload, timeout).await?;

                    session.events.push(SessionEvent::ActionExecuted {
                        action_type: payload.action_type.clone(),
                        outcome: outcome.clone(),
                    });

                    match outcome {
                        ExecutionOutcome::Success { stdout } => {
                            info!("Action succeeded. Injecting observation.");

                            // Inject success observation and continue.
                            let obs_text =
                                format!("\nObservation: Action succeeded.\n{}\n", stdout);
                            let obs_tokens = self.backend.tokenize(&obs_text)?;
                            for t in &obs_tokens {
                                state.push_token(t.clone());
                            }
                            let prefill_len = obs_tokens.len().saturating_sub(1);
                            if prefill_len > 0 {
                                self.backend
                                    .evaluate_tokens(
                                        state.seq_id,
                                        state.cache_position - obs_tokens.len(),
                                        &obs_tokens[..prefill_len],
                                    )
                                    .await?;
                            }

                            retry_count = 0; // Reset on success.
                        }

                        ExecutionOutcome::Failure {
                            exit_code,
                            stderr,
                            error_token_hint,
                        } => {
                            retry_count += 1;
                            warn!(
                                "Action failed (attempt {}/{}). exit_code={}, stderr='{}'",
                                retry_count, self.config.max_rollback_retries, exit_code, stderr
                            );

                            if retry_count > self.config.max_rollback_retries {
                                return Err(InferenceError::MaxRetriesExceeded {
                                    max_retries: self.config.max_rollback_retries,
                                });
                            }

                            // ── THE "TIME TRAVEL" ──
                            // Determine rollback anchor (t_k).
                            let rollback_pos = error_token_hint.unwrap_or(payload.start_token_pos);

                            let target = RollbackTarget::new(
                                state.seq_id,
                                rollback_pos,
                                state.cache_position,
                            );

                            // Create observation text.
                            let obs_text =
                                format!("\nObservation: Error (exit_code={exit_code}): {stderr}\n");
                            let obs_tokens = self.backend.tokenize(&obs_text)?;

                            // Execute rollback.
                            let erased =
                                self.kv_cache.rollback(&mut state, &target, &obs_tokens)?;

                            session.events.push(SessionEvent::Rollback {
                                position: rollback_pos,
                                tokens_erased: erased,
                                retry_number: retry_count,
                            });

                            info!(
                                "KV-Cache rollback: erased {} tokens, \
                                 injected observation at pos {}, resuming.",
                                erased, rollback_pos
                            );

                            // Resume generation from the new state.
                            // The loop continues — next iteration generates
                            // from the corrected position.
                        }

                        ExecutionOutcome::Timeout { timeout_ms } => {
                            retry_count += 1;
                            warn!("Action timed out after {timeout_ms}ms");

                            if retry_count > self.config.max_rollback_retries {
                                return Err(InferenceError::MaxRetriesExceeded {
                                    max_retries: self.config.max_rollback_retries,
                                });
                            }

                            // Rollback and inject timeout observation.
                            let target = RollbackTarget::new(
                                state.seq_id,
                                payload.start_token_pos,
                                state.cache_position,
                            );
                            let obs_text = format!(
                                "\nObservation: Execution timed out after {timeout_ms}ms. \
                                 Simplify the approach.\n"
                            );
                            let obs_tokens = self.backend.tokenize(&obs_text)?;

                            let erased =
                                self.kv_cache.rollback(&mut state, &target, &obs_tokens)?;

                            session.events.push(SessionEvent::Rollback {
                                position: payload.start_token_pos,
                                tokens_erased: erased,
                                retry_number: retry_count,
                            });
                        }
                    }
                }

                StopResult::Terminal => {
                    info!("Terminal stop sequence reached.");
                    break;
                }

                StopResult::MaxTokens => {
                    info!("Max tokens ({}) reached.", self.config.max_tokens);
                    break;
                }
            }
        }

        session.final_text = state.generated_text();
        session.total_tokens_generated = state.generated_count();
        session.total_rollbacks = self.kv_cache.rollback_count;
        session.total_tokens_erased = self.kv_cache.total_tokens_erased;

        Ok(session)
    }

    /// Generate tokens until a stop sequence is matched or max tokens reached.
    ///
    /// Convenience wrapper — no streaming, no cancellation.
    #[allow(dead_code)]
    async fn generate_until_stop(
        &self,
        state: &mut GenerationState,
    ) -> InferenceResult<StopResult> {
        let no_cancel = AtomicBool::new(false);
        self.generate_until_stop_streaming(state, &no_cancel, &mut |_| {})
            .await
    }

    /// Generate tokens with streaming output and cancellation support.
    ///
    /// # Arguments
    /// * `state` — Mutable generation state (token history + cache position).
    /// * `cancel` — Checked between tokens; if `true`, returns `Terminal`.
    /// * `on_token` — Called with each token's text as it is generated.
    async fn generate_until_stop_streaming<F>(
        &self,
        state: &mut GenerationState,
        cancel: &AtomicBool,
        on_token: &mut F,
    ) -> InferenceResult<StopResult>
    where
        F: FnMut(String),
    {
        let mut generated_text = String::new();

        for _ in 0..self.config.max_tokens {
            // Check cancellation flag.
            if cancel.load(Ordering::Relaxed) {
                return Ok(StopResult::Terminal);
            }

            let token = self.backend.generate_next(state, &self.config).await?;

            let Some(token) = token else {
                return Ok(StopResult::Terminal);
            };

            // Stream the token to the caller.
            on_token(token.text.clone());

            generated_text.push_str(&token.text);
            state.push_token(token);

            // Check for stop sequences.
            for stop in &self.config.stop_sequences {
                if generated_text.ends_with(&stop.pattern) {
                    if stop.is_action_boundary {
                        return Ok(StopResult::ActionBoundary { generated_text });
                    } else {
                        return Ok(StopResult::Terminal);
                    }
                }
            }
        }

        Ok(StopResult::MaxTokens)
    }

    /// Get a reference to the KV-Cache controller.
    pub fn kv_cache(&self) -> &KvCacheController<M> {
        &self.kv_cache
    }

    /// Get a reference to the engine configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// StopResult
// ---------------------------------------------------------------------------

/// Why token generation stopped.
#[derive(Debug)]
enum StopResult {
    /// An action boundary stop sequence was matched.
    ActionBoundary {
        /// The full generated text up to this point.
        generated_text: String,
    },
    /// A terminal stop sequence was matched.
    Terminal,
    /// Max tokens reached without a stop sequence.
    MaxTokens,
}

// ---------------------------------------------------------------------------
// InferenceSession
// ---------------------------------------------------------------------------

/// The complete result of an inference run.
///
/// Captures the final output text along with all events (executions,
/// rollbacks, warnings) that occurred during the session.
#[derive(Debug, Clone)]
pub struct InferenceSession {
    /// The final generated text (after all rollbacks/corrections).
    pub final_text: String,

    /// Total tokens generated across all attempts (including erased ones).
    pub total_tokens_generated: usize,

    /// Number of KV-Cache rollbacks performed.
    pub total_rollbacks: u64,

    /// Total tokens erased via rollback.
    pub total_tokens_erased: u64,

    /// Ordered list of events that occurred during the session.
    pub events: Vec<SessionEvent>,
}

impl InferenceSession {
    fn new() -> Self {
        Self {
            final_text: String::new(),
            total_tokens_generated: 0,
            total_rollbacks: 0,
            total_tokens_erased: 0,
            events: Vec::new(),
        }
    }
}

/// An event that occurred during an inference session.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// An action was extracted and executed.
    ActionExecuted {
        action_type: String,
        outcome: ExecutionOutcome,
    },

    /// A KV-Cache rollback was performed.
    Rollback {
        position: usize,
        tokens_erased: usize,
        retry_number: usize,
    },

    /// A non-fatal warning.
    Warning(String),
}
