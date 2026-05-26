//! # Actor Model Types
//!
//! Core data structures for the multi-agent swarm:
//! - `AgentId`: Unique agent identifier.
//! - `Signal`: Interrupt and control signals.
//! - `Envelope`: A message sent between agents via MPSC channels.
//! - `AgentStatus`: Runtime state of an agent.
//! - `EscalationResponse`: LLM-generated review verdict for the escalation pipeline.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// AgentId
// ---------------------------------------------------------------------------

/// Unique identifier for an agent in the swarm.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId {
    /// UUID of this agent.
    pub uuid: Uuid,
    /// Human-readable name (e.g., "Coder", "Reviewer", "Planner").
    pub name: String,
    /// Agent role classification.
    pub role: AgentRole,
}

impl AgentId {
    /// Create a new agent ID with a generated UUID.
    pub fn new(name: impl Into<String>, role: AgentRole) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: name.into(),
            role,
        }
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.name, &self.uuid.to_string()[..8])
    }
}

// ---------------------------------------------------------------------------
// AgentRole
// ---------------------------------------------------------------------------

/// Classification of an agent's function in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Generates code / text.
    Coder,
    /// Reviews output from other agents.
    Reviewer,
    /// High-level planning and task decomposition.
    Planner,
    /// Executes tools and returns observations.
    Executor,
    /// Custom / user-defined role.
    Custom,
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// Control signals sent via the MPSC channel system.
///
/// These are the asynchronous interrupts described in the IMECE spec:
/// - `Agent_B` (Reviewer) detects anomaly → sends `INTERRUPT_SIGNAL`.
/// - Core Engine halts `Agent_A`'s generation.
/// - Triggers KV-Cache Rollback (Module 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    /// Interrupt a specific agent's current generation.
    /// Contains the target agent ID and a reason string.
    Interrupt {
        target: AgentId,
        reason: String,
        /// Severity: higher = more urgent.
        severity: u8,
    },

    /// Request a specific agent to halt gracefully.
    Halt { target: AgentId },

    /// Notify all agents that a rollback occurred.
    RollbackNotification {
        source: AgentId,
        tokens_erased: usize,
        rollback_position: usize,
    },

    /// Agent completed its current task.
    TaskComplete { source: AgentId, result: TaskResult },

    /// Heartbeat signal for health monitoring.
    Heartbeat { source: AgentId },

    /// Shutdown the entire swarm.
    Shutdown,
}

// ---------------------------------------------------------------------------
// TaskResult
// ---------------------------------------------------------------------------

/// The outcome of an agent's task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    /// Task completed successfully with output text.
    Success { output: String },

    /// Task failed with an error description.
    Failure { error: String },

    /// Task was interrupted before completion.
    Interrupted { partial_output: String },
}

// ---------------------------------------------------------------------------
// Envelope
// ---------------------------------------------------------------------------

/// A message envelope wrapping content sent between agents.
///
/// All inter-agent communication goes through typed envelopes via
/// MPSC channels — agents never share mutable state.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Unique message ID.
    pub id: Uuid,

    /// Sender agent.
    pub from: AgentId,

    /// Target agent(s). `None` = broadcast to all.
    pub to: Option<AgentId>,

    /// The message payload.
    pub payload: MessagePayload,

    /// When this message was created.
    pub timestamp: Instant,

    /// Priority (lower = higher priority). Used for queue ordering.
    pub priority: u8,
}

impl Envelope {
    /// Create a new directed message.
    pub fn new(from: AgentId, to: AgentId, payload: MessagePayload) -> Self {
        Self {
            id: Uuid::new_v4(),
            from,
            to: Some(to),
            payload,
            timestamp: Instant::now(),
            priority: 128, // Default mid-priority.
        }
    }

    /// Create a broadcast message (no specific target).
    pub fn broadcast(from: AgentId, payload: MessagePayload) -> Self {
        Self {
            id: Uuid::new_v4(),
            from,
            to: None,
            payload,
            timestamp: Instant::now(),
            priority: 128,
        }
    }

    /// Set the priority (0 = highest, 255 = lowest).
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

// ---------------------------------------------------------------------------
// MessagePayload
// ---------------------------------------------------------------------------

/// The typed content of an inter-agent message.
#[derive(Debug, Clone)]
pub enum MessagePayload {
    /// A text chunk from a streaming generation.
    TextChunk {
        /// The text fragment.
        text: String,
        /// Whether this is the final chunk.
        is_final: bool,
        /// Token count in this chunk.
        token_count: usize,
    },

    /// A signal (interrupt, halt, shutdown, etc.).
    Signal(Signal),

    /// A review/feedback request.
    ReviewRequest {
        /// The content to review.
        content: String,
        /// What to check for.
        criteria: Vec<String>,
    },

    /// A review response.
    ReviewResponse {
        /// Whether the review passed.
        approved: bool,
        /// Feedback / issues found.
        feedback: String,
        /// Specific issues that should trigger rollback.
        rollback_issues: Vec<String>,
    },

    /// A task assignment from the Planner.
    TaskAssignment {
        /// Task description.
        description: String,
        /// Context (e.g., memory chain from Module 1).
        context: String,
    },

    /// A request to run LLM inference on a prompt.
    /// Sent from a worker agent (e.g., Coder) to the InferenceAgent.
    InferenceRequest {
        /// The full prompt string (system + context + task).
        prompt: String,
        /// The agent that should receive streaming TextChunk responses.
        reply_to: AgentId,
    },

    /// A lightweight, isolated LLM inference request from a peripheral
    /// agent (e.g., Reviewer) that needs semantic analysis.
    ///
    /// # KV-Cache Isolation (Constraint #3)
    ///
    /// The `seq_id` field specifies an **isolated** sequence that does NOT
    /// overlap with any active generation session. The InferenceAgent
    /// creates a temporary `GenerationState` for this seq_id, generates
    /// up to `max_tokens`, and discards the state — the primary Coder's
    /// KV-Cache is never touched.
    ///
    /// # Deadlock Prevention
    ///
    /// The response is sent via a `tokio::sync::oneshot` channel embedded
    /// in `reply_tx`, not through the inbox MPSC. This means the requesting
    /// agent can `.await` the oneshot inside `handle_message` without
    /// blocking its inbox and without risk of backpressure deadlock.
    ///
    /// # Why `Arc<Mutex<Option<Sender>>>` ?
    ///
    /// `Envelope` derives `Clone` (for broadcasting), but `oneshot::Sender`
    /// is not `Clone`. Wrapping in `Arc<Mutex<Option<_>>>` lets us `.take()`
    /// the sender exactly once on the receiving side. This is NOT shared
    /// mutable state in the actor model sense — it's a single-use transfer.
    EscalationRequest {
        /// The structured prompt for the review task.
        prompt: String,
        /// Oneshot sender for the response — taken exactly once by the receiver.
        reply_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<EscalationResponse>>>>,
        /// Isolated sequence ID — MUST NOT overlap with active generation.
        /// Reserved range: `1000..2000`.
        seq_id: u32,
        /// Hard token cap for this request. Enforced by InferenceAgent.
        max_tokens: u16,
    },

    /// Raw bytes (for future extensions like embeddings).
    Binary { data: Vec<u8>, mime_type: String },
}

// ---------------------------------------------------------------------------
// AgentStatus
// ---------------------------------------------------------------------------

/// Runtime status of an agent in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle, waiting for work.
    Idle,
    /// Agent is actively generating / processing.
    Active,
    /// Agent has been interrupted and is rolling back.
    Interrupted,
    /// Agent encountered an error.
    Error,
    /// Agent has been shut down.
    Shutdown,
}

// ---------------------------------------------------------------------------
// AgentMetrics
// ---------------------------------------------------------------------------

/// Runtime metrics for a single agent (telemetry for edge-device monitoring).
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// Total messages sent by this agent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Number of interrupts received.
    pub interrupts_received: u64,
    /// Number of tasks completed.
    pub tasks_completed: u64,
    /// Number of tasks failed.
    pub tasks_failed: u64,
    /// Total tokens generated.
    pub tokens_generated: u64,
}

// ---------------------------------------------------------------------------
// EscalationResponse
// ---------------------------------------------------------------------------

/// The LLM's response to an [`EscalationRequest`](MessagePayload::EscalationRequest).
///
/// Contains the raw generated text and metadata about the inference.
/// The requesting agent (e.g., `LlmStage`) is responsible for parsing
/// this into a structured [`Verdict`](super::escalation::Verdict).
#[derive(Debug, Clone)]
pub struct EscalationResponse {
    /// The generated text from the LLM.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Whether inference completed normally or was cancelled/errored.
    pub status: EscalationStatus,
}

/// Outcome of an escalation inference request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationStatus {
    /// Generation completed normally (hit stop sequence or max_tokens).
    Complete,
    /// Generation was cancelled via cooperative interrupt.
    Cancelled,
    /// An error occurred during inference.
    Error(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_display() {
        let id = AgentId::new("Coder", AgentRole::Coder);
        let display = format!("{id}");
        assert!(display.starts_with("Coder["));
        assert_eq!(display.len(), "Coder[".len() + 9); // 8 hex + ']'
    }

    #[test]
    fn test_envelope_directed() {
        let from = AgentId::new("Coder", AgentRole::Coder);
        let to = AgentId::new("Reviewer", AgentRole::Reviewer);
        let env = Envelope::new(
            from.clone(),
            to.clone(),
            MessagePayload::TextChunk {
                text: "fn main() {}".into(),
                is_final: true,
                token_count: 5,
            },
        );

        assert_eq!(env.from.name, "Coder");
        assert_eq!(env.to.as_ref().unwrap().name, "Reviewer");
        assert_eq!(env.priority, 128);
    }

    #[test]
    fn test_envelope_broadcast() {
        let from = AgentId::new("Planner", AgentRole::Planner);
        let env = Envelope::broadcast(from, MessagePayload::Signal(Signal::Shutdown));
        assert!(env.to.is_none());
    }

    #[test]
    fn test_envelope_priority() {
        let from = AgentId::new("A", AgentRole::Custom);
        let to = AgentId::new("B", AgentRole::Custom);
        let env =
            Envelope::new(from, to, MessagePayload::Signal(Signal::Shutdown)).with_priority(0);
        assert_eq!(env.priority, 0);
    }
}
