//! # Inference Types
//!
//! Core data structures for the inference pipeline. These are
//! backend-agnostic — they model concepts from llama.cpp's KV-Cache
//! semantics without directly depending on the C bindings (which are
//! injected as trait objects at runtime).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single token produced by the LLM's tokenizer.
///
/// Wraps the raw integer token ID with its decoded text representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token {
    /// Integer token ID from the vocabulary.
    pub id: u32,
    /// Decoded UTF-8 text of this token (may be a subword / byte).
    pub text: String,
}

// ---------------------------------------------------------------------------
// GenerationState
// ---------------------------------------------------------------------------

/// Snapshot of the autoregressive generation state at a given point in time.
///
/// This is the "timeline" that KV-Cache rollback can restore to.
#[derive(Debug, Clone)]
pub struct GenerationState {
    /// Sequence ID within the llama.cpp batch context.
    pub seq_id: u32,

    /// All tokens generated so far (including the prompt tokens).
    pub tokens: Vec<Token>,

    /// Index of the first generated (non-prompt) token.
    /// Everything before this is prompt/context — never rolled back.
    pub prompt_end: usize,

    /// Current position in the KV-Cache (= `tokens.len()`).
    pub cache_position: usize,
}

impl GenerationState {
    /// Create a new state for a fresh generation pass.
    pub fn new(seq_id: u32, prompt_tokens: Vec<Token>) -> Self {
        let prompt_end = prompt_tokens.len();
        let cache_position = prompt_end;
        Self {
            seq_id,
            tokens: prompt_tokens,
            prompt_end,
            cache_position,
        }
    }

    /// Append a newly generated token.
    pub fn push_token(&mut self, token: Token) {
        self.tokens.push(token);
        self.cache_position = self.tokens.len();
    }

    /// Number of generated (non-prompt) tokens.
    #[inline]
    pub fn generated_count(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_end)
    }

    /// Get the text of all generated tokens concatenated.
    pub fn generated_text(&self) -> String {
        self.tokens[self.prompt_end..]
            .iter()
            .map(|t| t.text.as_str())
            .collect()
    }

    /// Get the text of tokens in range `[from, to)`.
    pub fn text_range(&self, from: usize, to: usize) -> String {
        let end = to.min(self.tokens.len());
        self.tokens[from..end]
            .iter()
            .map(|t| t.text.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RollbackTarget
// ---------------------------------------------------------------------------

/// Specifies where to rollback the KV-Cache.
///
/// Maps directly to the llama.cpp function:
///   `llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)`
///
/// Which erases cache entries from position `p0` to `p1` (exclusive)
/// for the given sequence.
#[derive(Debug, Clone, Copy)]
pub struct RollbackTarget {
    /// Sequence ID to target.
    pub seq_id: u32,

    /// First token position to erase (inclusive).
    /// This is `t_k` in the spec — where the logical error began.
    pub from_position: usize,

    /// Last token position to erase (exclusive).
    /// Typically `cache_position` (= end of generated sequence).
    pub to_position: usize,
}

impl RollbackTarget {
    /// Create a rollback target that erases from `error_position` to `end`.
    pub fn new(seq_id: u32, error_position: usize, end: usize) -> Self {
        Self {
            seq_id,
            from_position: error_position,
            to_position: end,
        }
    }

    /// Number of KV-Cache entries that will be erased.
    #[inline]
    pub fn erased_count(&self) -> usize {
        self.to_position.saturating_sub(self.from_position)
    }
}

// ---------------------------------------------------------------------------
// StopSequence
// ---------------------------------------------------------------------------

/// A stop sequence that triggers interception during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopSequence {
    /// The text pattern to match (e.g., `</action>`, `</code>`).
    pub pattern: String,

    /// Whether this stop sequence indicates an action to execute.
    pub is_action_boundary: bool,
}

impl StopSequence {
    /// Create a new action-boundary stop sequence.
    pub fn action(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            is_action_boundary: true,
        }
    }

    /// Create a generic (non-action) stop sequence.
    pub fn terminal(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            is_action_boundary: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ActionPayload
// ---------------------------------------------------------------------------

/// An extracted action from the LLM's generation, parsed from the text
/// between `<action>` and `</action>` tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPayload {
    /// The type of action (e.g., "python", "bash", "sql").
    pub action_type: String,

    /// The raw code / command to execute.
    pub code: String,

    /// Token position where the `<action>` tag started.
    /// This is the rollback anchor — if execution fails, we erase from here.
    pub start_token_pos: usize,

    /// Token position where the `</action>` tag ended.
    pub end_token_pos: usize,
}

// ---------------------------------------------------------------------------
// ExecutionResult
// ---------------------------------------------------------------------------

/// Outcome of executing an action in the sandboxed environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionOutcome {
    /// Action succeeded — include optional stdout.
    Success { stdout: String },

    /// Action failed — include error details.
    Failure {
        exit_code: i32,
        stderr: String,
        /// Best-effort identification of which token position caused the error.
        /// Used as the rollback anchor `t_k`.
        error_token_hint: Option<usize>,
    },

    /// Action timed out.
    Timeout { timeout_ms: u64 },
}

impl ExecutionOutcome {
    /// Whether this outcome represents a success.
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, ExecutionOutcome::Success { .. })
    }
}

// ---------------------------------------------------------------------------
// InferenceConfig
// ---------------------------------------------------------------------------

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum tokens to generate per pass.
    pub max_tokens: usize,

    /// Temperature for sampling.
    pub temperature: f32,

    /// Top-K sampling parameter.
    pub top_k: u32,

    /// Top-P (nucleus) sampling parameter.
    pub top_p: f32,

    /// Maximum number of rollback retries before aborting.
    pub max_rollback_retries: usize,

    /// Stop sequences to intercept.
    pub stop_sequences: Vec<StopSequence>,

    /// Timeout for action execution (milliseconds).
    pub execution_timeout_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            max_rollback_retries: 3,
            stop_sequences: vec![
                StopSequence::action("</action>"),
                StopSequence::terminal("</s>"),
            ],
            execution_timeout_ms: 30_000,
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
    fn test_generation_state_new() {
        let prompt = vec![
            Token {
                id: 1,
                text: "Hello".into(),
            },
            Token {
                id: 2,
                text: " world".into(),
            },
        ];
        let state = GenerationState::new(0, prompt);
        assert_eq!(state.seq_id, 0);
        assert_eq!(state.prompt_end, 2);
        assert_eq!(state.cache_position, 2);
        assert_eq!(state.generated_count(), 0);
    }

    #[test]
    fn test_generation_state_push_and_text() {
        let prompt = vec![Token {
            id: 1,
            text: "Hi".into(),
        }];
        let mut state = GenerationState::new(0, prompt);

        state.push_token(Token {
            id: 10,
            text: " there".into(),
        });
        state.push_token(Token {
            id: 11,
            text: "!".into(),
        });

        assert_eq!(state.generated_count(), 2);
        assert_eq!(state.generated_text(), " there!");
        assert_eq!(state.cache_position, 3);
    }

    #[test]
    fn test_rollback_target() {
        let target = RollbackTarget::new(0, 50, 100);
        assert_eq!(target.erased_count(), 50);
        assert_eq!(target.from_position, 50);
        assert_eq!(target.to_position, 100);
    }

    #[test]
    fn test_execution_outcome_success() {
        let outcome = ExecutionOutcome::Success {
            stdout: "42".into(),
        };
        assert!(outcome.is_success());
    }

    #[test]
    fn test_execution_outcome_failure() {
        let outcome = ExecutionOutcome::Failure {
            exit_code: 1,
            stderr: "SyntaxError".into(),
            error_token_hint: Some(42),
        };
        assert!(!outcome.is_success());
    }

    #[test]
    fn test_stop_sequence_types() {
        let action = StopSequence::action("</action>");
        assert!(action.is_action_boundary);

        let terminal = StopSequence::terminal("</s>");
        assert!(!terminal.is_action_boundary);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.max_rollback_retries, 3);
        assert_eq!(config.stop_sequences.len(), 2);
    }
}
