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

use std::collections::HashMap;
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

        // Proactive error prevention: per-position failure history.
        // Only allocated when the feature is enabled — zero overhead otherwise.
        let prevention = &self.config.rollback_error_prevention;
        let mut failure_history: Option<HashMap<usize, Vec<FailureRecord>>> =
            if prevention.enabled {
                Some(HashMap::new())
            } else {
                None
            };

        info!(
            "Inference started. prompt_len={}, max_tokens={}, error_prevention={}",
            state.prompt_end, self.config.max_tokens, prevention.enabled
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

                            // Capture the content about to be erased BEFORE the
                            // rollback truncates the token buffer. This is the
                            // model's "forgotten" output — critical for cumulative
                            // observations so the LLM can see what it tried.
                            let erased_content =
                                state.text_range(rollback_pos, state.cache_position);

                            let target = RollbackTarget::new(
                                state.seq_id,
                                rollback_pos,
                                state.cache_position,
                            );

                            // Build the observation — cumulative if prevention is
                            // enabled and there's prior history, plain otherwise.
                            let obs_text = Self::build_observation(
                                &mut failure_history,
                                prevention,
                                rollback_pos,
                                &erased_content,
                                &stderr,
                                exit_code,
                            );
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

                            let rollback_pos = payload.start_token_pos;

                            // Capture erased content before rollback.
                            let erased_content =
                                state.text_range(rollback_pos, state.cache_position);

                            let target = RollbackTarget::new(
                                state.seq_id,
                                rollback_pos,
                                state.cache_position,
                            );

                            let timeout_error =
                                format!("Execution timed out after {timeout_ms}ms");

                            // Build observation — cumulative or plain.
                            let obs_text = Self::build_observation(
                                &mut failure_history,
                                prevention,
                                rollback_pos,
                                &erased_content,
                                &timeout_error,
                                -1, // Conventional exit code for timeout.
                            );
                            let obs_tokens = self.backend.tokenize(&obs_text)?;

                            let erased =
                                self.kv_cache.rollback(&mut state, &target, &obs_tokens)?;

                            session.events.push(SessionEvent::Rollback {
                                position: rollback_pos,
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

    /// Build the observation text for a rollback.
    ///
    /// When proactive error prevention is enabled, this records the failure
    /// in the per-position history and synthesizes a cumulative observation
    /// that shows the model what it tried and why each attempt failed.
    ///
    /// When disabled, produces the same plain observation as the original
    /// code path.
    fn build_observation(
        failure_history: &mut Option<HashMap<usize, Vec<FailureRecord>>>,
        prevention: &RollbackErrorPrevention,
        rollback_pos: usize,
        erased_content: &str,
        error_message: &str,
        exit_code: i32,
    ) -> String {
        // Fast path: feature disabled — no tracking, plain observation.
        let Some(history) = failure_history.as_mut() else {
            return format!(
                "\nObservation: Error (exit_code={exit_code}): {error_message}\n"
            );
        };

        // Record this failure.
        let records = history.entry(rollback_pos).or_default();

        // Enforce max_history_per_position cap (drop oldest).
        if records.len() >= prevention.max_history_per_position {
            records.remove(0);
        }

        records.push(FailureRecord::new(
            erased_content.to_owned(),
            error_message.to_owned(),
            exit_code,
            prevention.max_erased_content_bytes,
            prevention.max_error_message_bytes,
        ));

        // Build observation.
        let attempt_count = records.len();

        if attempt_count <= 1 {
            // First failure at this position — include current error with
            // the erased content so the model knows what it wrote.
            let truncated_content = if erased_content.len() > prevention.max_erased_content_bytes {
                let boundary =
                    erased_content.ceil_char_boundary(
                        erased_content.len() - prevention.max_erased_content_bytes,
                    );
                format!("[...] {}", &erased_content[boundary..])
            } else {
                erased_content.to_owned()
            };

            format!(
                "\nObservation: Error (exit_code={exit_code}): {error_message}\n\
                 Your previous output was: \u{ab}{truncated_content}\u{bb}\n"
            )
        } else {
            // Repeated failure — cumulative observation with full history.
            let mut obs = format!(
                "\nObservation: Error (exit_code={exit_code}): {error_message}\n\
                 [!] This position has failed {attempt_count} time(s). \
                 You MUST use a substantially different approach.\n\
                 Previous failed attempts:\n"
            );

            for (i, record) in records.iter().enumerate() {
                obs.push_str(&format!(
                    "  [Attempt {}] Generated: \u{ab}{}\u{bb} \u{2192} \
                     Error (exit_code={}): {}\n",
                    i + 1,
                    record.erased_content,
                    record.exit_code,
                    record.error_message,
                ));
            }

            obs
        }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::InferenceResult;
    use std::sync::Mutex;
    use std::time::Duration;

    // ── Mock LLM Backend ──────────────────────────────────────────────

    /// A mock backend that returns a predetermined sequence of tokens,
    /// then a terminal stop sequence.
    struct MockBackend {
        /// Tokens to emit, in order. After exhaustion, returns EOS.
        tokens: Mutex<Vec<Token>>,
    }

    impl MockBackend {
        fn new(tokens: Vec<Token>) -> Self {
            Self {
                tokens: Mutex::new(tokens),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for MockBackend {
        fn tokenize(&self, text: &str) -> InferenceResult<Vec<Token>> {
            // Simple tokenizer: one token per character.
            Ok(text
                .chars()
                .enumerate()
                .map(|(i, c)| Token {
                    id: i as u32,
                    text: c.to_string(),
                })
                .collect())
        }

        fn detokenize(&self, tokens: &[Token]) -> InferenceResult<String> {
            Ok(tokens.iter().map(|t| t.text.as_str()).collect())
        }

        async fn generate_next(
            &self,
            _state: &GenerationState,
            _config: &InferenceConfig,
        ) -> InferenceResult<Option<Token>> {
            let mut tokens = self.tokens.lock().unwrap();
            if tokens.is_empty() {
                Ok(None)
            } else {
                Ok(Some(tokens.remove(0)))
            }
        }

        async fn evaluate_tokens(
            &self,
            _seq_id: u32,
            _position: usize,
            _tokens: &[Token],
        ) -> InferenceResult<()> {
            Ok(())
        }
    }

    // ── Mock KV-Cache Manager ─────────────────────────────────────────

    /// A mock KV-Cache that accepts all operations without tracking real
    /// cache state. In the real system, KV-Cache length grows with token
    /// generation — the mock can't observe that, so `seq_len()` returns
    /// `usize::MAX` to always pass bounds checks. Bounds-checking
    /// correctness is covered by `KvCacheController`'s own unit tests.
    struct MockKvCache;

    impl MockKvCache {
        fn new(_len: usize) -> Self {
            Self
        }
    }

    impl KvCacheManager for MockKvCache {
        fn seq_rm(&mut self, _seq_id: u32, _p0: usize, _p1: usize) -> InferenceResult<()> {
            Ok(())
        }

        fn seq_len(&self, _seq_id: u32) -> usize {
            // Always pass bounds checks — the mock doesn't track
            // token generation, so it can't report a real length.
            usize::MAX
        }

        fn inject_tokens(
            &mut self,
            _seq_id: u32,
            _position: usize,
            _tokens: &[Token],
        ) -> InferenceResult<()> {
            Ok(())
        }

        fn clear_all(&mut self) -> InferenceResult<()> {
            Ok(())
        }
    }

    // ── Mock Executor ─────────────────────────────────────────────────

    /// A mock executor that returns a configurable sequence of outcomes.
    struct MockExecutor {
        outcomes: Mutex<Vec<ExecutionOutcome>>,
    }

    impl MockExecutor {
        fn new(outcomes: Vec<ExecutionOutcome>) -> Self {
            Self {
                outcomes: Mutex::new(outcomes),
            }
        }
    }

    #[async_trait::async_trait]
    impl ActionExecutor for MockExecutor {
        async fn execute(
            &self,
            _payload: &ActionPayload,
            _timeout: Duration,
        ) -> InferenceResult<ExecutionOutcome> {
            let mut outcomes = self.outcomes.lock().unwrap();
            if outcomes.is_empty() {
                Ok(ExecutionOutcome::Success {
                    stdout: String::new(),
                })
            } else {
                Ok(outcomes.remove(0))
            }
        }
    }

    // ── Helper: build tokens for an action block ──────────────────────

    fn action_tokens(code: &str) -> Vec<Token> {
        let text = format!("<action type=\"python\">{}\n</action>", code);
        text.chars()
            .enumerate()
            .map(|(i, c)| Token {
                id: (100 + i) as u32,
                text: c.to_string(),
            })
            .collect()
    }

    // ── Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_build_observation_disabled() {
        // When prevention is disabled, failure_history is None.
        let prevention = RollbackErrorPrevention {
            enabled: false,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = None;

        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "print(x)",
            "NameError: name 'x' is not defined",
            1,
        );

        // Should be the plain observation format.
        assert!(obs.contains("Error (exit_code=1)"));
        assert!(obs.contains("NameError"));
        // Should NOT contain cumulative context.
        assert!(!obs.contains("failed"));
        assert!(!obs.contains("Attempt"));
        assert!(!obs.contains("previous output"));
        // History should remain None.
        assert!(history.is_none());
    }

    #[test]
    fn test_build_observation_first_failure() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "print(x)",
            "NameError: name 'x' is not defined",
            1,
        );

        // First failure includes erased content but no cumulative history.
        assert!(obs.contains("Error (exit_code=1)"));
        assert!(obs.contains("NameError"));
        assert!(obs.contains("print(x)"));
        assert!(obs.contains("previous output"));
        assert!(!obs.contains("Attempt"));
        // History should have one record at position 50.
        let records = history.as_ref().unwrap().get(&50).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].exit_code, 1);
    }

    #[test]
    fn test_build_observation_cumulative() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            max_history_per_position: 5,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        // First failure.
        let _ = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "print(x)",
            "NameError: name 'x' is not defined",
            1,
        );

        // Second failure at the same position.
        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "print(foo)",
            "NameError: name 'foo' is not defined",
            1,
        );

        // Should include cumulative history.
        assert!(obs.contains("failed 2 time(s)"));
        assert!(obs.contains("[Attempt 1]"));
        assert!(obs.contains("[Attempt 2]"));
        assert!(obs.contains("print(x)"));
        assert!(obs.contains("print(foo)"));
        assert!(obs.contains("MUST use a substantially different approach"));

        // Third failure.
        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "import os; print(os.getcwd())",
            "SyntaxError: unexpected token",
            1,
        );

        assert!(obs.contains("failed 3 time(s)"));
        assert!(obs.contains("[Attempt 1]"));
        assert!(obs.contains("[Attempt 2]"));
        assert!(obs.contains("[Attempt 3]"));
    }

    #[test]
    fn test_build_observation_different_positions() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        // Failure at position 50.
        let _ = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "code_A",
            "error_A",
            1,
        );

        // Failure at position 100 — different position, no history here.
        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            100,
            "code_B",
            "error_B",
            2,
        );

        // Should be a first-failure observation (no cumulative history at pos 100).
        assert!(obs.contains("previous output"));
        assert!(obs.contains("code_B"));
        assert!(!obs.contains("code_A"));
        assert!(!obs.contains("Attempt"));

        // Verify both positions are tracked independently.
        let h = history.as_ref().unwrap();
        assert_eq!(h.get(&50).unwrap().len(), 1);
        assert_eq!(h.get(&100).unwrap().len(), 1);
    }

    #[test]
    fn test_build_observation_max_history_cap() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            max_history_per_position: 2, // Only keep 2 records.
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        // Three failures at position 50.
        for i in 0..3 {
            let _ = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
                &mut history,
                &prevention,
                50,
                &format!("code_{i}"),
                &format!("error_{i}"),
                1,
            );
        }

        // Should only have 2 records (oldest dropped).
        let records = history.as_ref().unwrap().get(&50).unwrap();
        assert_eq!(records.len(), 2);
        // The oldest record (code_0) should be gone.
        assert!(records[0].erased_content.contains("code_1"));
        assert!(records[1].erased_content.contains("code_2"));
    }

    #[test]
    fn test_build_observation_timeout_tracking() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        // First: a normal failure.
        let _ = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "long_running_code()",
            "NameError",
            1,
        );

        // Second: a timeout at the same position.
        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            "even_longer_code()",
            "Execution timed out after 30000ms",
            -1, // Timeout convention.
        );

        // Should show cumulative history mixing failure types.
        assert!(obs.contains("failed 2 time(s)"));
        assert!(obs.contains("[Attempt 1]"));
        assert!(obs.contains("NameError"));
        assert!(obs.contains("[Attempt 2]"));
        assert!(obs.contains("timed out"));
    }

    #[test]
    fn test_build_observation_erased_content_truncation() {
        let prevention = RollbackErrorPrevention {
            enabled: true,
            max_erased_content_bytes: 20,
            max_error_message_bytes: 256,
            ..Default::default()
        };
        let mut history: Option<HashMap<usize, Vec<FailureRecord>>> = Some(HashMap::new());

        let long_content = "x".repeat(1000);
        let obs = InferenceEngine::<MockBackend, MockKvCache, MockExecutor>::build_observation(
            &mut history,
            &prevention,
            50,
            &long_content,
            "error",
            1,
        );

        // The erased content in the observation should be truncated.
        assert!(obs.contains("[...]"));
        // The full 1000-char content should NOT be present.
        assert!(!obs.contains(&"x".repeat(1000)));
    }

    #[tokio::test]
    async fn test_engine_disabled_prevention_plain_observation() {
        // Two action blocks: first fails, second succeeds.
        let mut tokens = action_tokens("print(x)");
        tokens.extend(action_tokens("print(42)"));

        let backend = MockBackend::new(tokens);
        let kv = MockKvCache::new(1000);
        let executor = MockExecutor::new(vec![
            ExecutionOutcome::Failure {
                exit_code: 1,
                stderr: "NameError".into(),
                error_token_hint: None,
            },
            ExecutionOutcome::Success {
                stdout: "42".into(),
            },
        ]);

        let config = InferenceConfig {
            max_rollback_retries: 3,
            rollback_error_prevention: RollbackErrorPrevention {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let controller = KvCacheController::new(kv);
        let mut engine = InferenceEngine::new(backend, controller, executor, config);

        let session = engine.run("test").await.unwrap();

        // Should have had one rollback.
        assert_eq!(session.total_rollbacks, 1);
        // Should have events for both executions.
        let action_events: Vec<_> = session
            .events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ActionExecuted { .. }))
            .collect();
        assert_eq!(action_events.len(), 2);
    }

    #[tokio::test]
    async fn test_engine_enabled_prevention_tracks_failures() {
        // Three action blocks: first two fail at same position, third succeeds.
        let mut tokens = action_tokens("print(x)");
        tokens.extend(action_tokens("print(y)"));
        tokens.extend(action_tokens("print(42)"));

        let backend = MockBackend::new(tokens);
        let kv = MockKvCache::new(1000);
        let executor = MockExecutor::new(vec![
            ExecutionOutcome::Failure {
                exit_code: 1,
                stderr: "NameError: x".into(),
                error_token_hint: None,
            },
            ExecutionOutcome::Failure {
                exit_code: 1,
                stderr: "NameError: y".into(),
                error_token_hint: None,
            },
            ExecutionOutcome::Success {
                stdout: "42".into(),
            },
        ]);

        let config = InferenceConfig {
            max_rollback_retries: 5,
            rollback_error_prevention: RollbackErrorPrevention {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let controller = KvCacheController::new(kv);
        let mut engine = InferenceEngine::new(backend, controller, executor, config);

        let session = engine.run("test").await.unwrap();

        // Should have had two rollbacks.
        assert_eq!(session.total_rollbacks, 2);
    }
}
