//! # Agent Trait & Handle
//!
//! Defines the `Agent` trait that all swarm participants implement,
//! and the `AgentHandle` which is the runtime wrapper connecting an
//! agent to the MPSC channel system.
//!
//! ## Concurrent Interrupt Architecture (Dual-Channel)
//!
//! Each agent runs with **two** MPSC channels:
//! - **`inbox_rx`**: Normal data messages (TaskAssignment, TextChunk, etc.)
//! - **`signal_rx`**: High-priority control signals (Interrupt, Halt, Shutdown)
//!
//! When an agent is busy inside `handle_message().await`, the loop uses
//! `tokio::select!` to race the in-flight future against `signal_rx`.
//! If a signal arrives mid-execution:
//! 1. The agent's `cancel_token` (`Arc<AtomicBool>`) is set immediately.
//! 2. The in-flight `handle_message` checks this flag cooperatively and
//!    returns partial results — no future is dropped.
//! 3. After `handle_message` completes, deferred signals are processed
//!    via `handle_interrupt()` for state cleanup.
//!
//! This avoids KV-Cache corruption (soft-kill), borrow checker conflicts
//! (cancel token is `Arc`, not `&mut self`), and shared mutable state.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::types::*;

// ---------------------------------------------------------------------------
// Agent Trait
// ---------------------------------------------------------------------------

/// The core trait that all swarm agents implement.
///
/// Each agent runs in its own Tokio task, receiving messages from an
/// MPSC channel and sending responses through a sender handle.
///
/// # Design Principles
/// - **No shared state:** Agents own all their data privately.
/// - **Message-only communication:** All interaction goes through `Envelope`.
/// - **Interruptible:** Must handle `Signal::Interrupt` and `Signal::Halt`.
#[async_trait::async_trait]
pub trait Agent: Send + 'static {
    /// The agent's unique identifier.
    fn id(&self) -> &AgentId;

    /// Current status of the agent.
    fn status(&self) -> AgentStatus;

    /// Current metrics.
    fn metrics(&self) -> &AgentMetrics;

    /// Handle an incoming message envelope.
    ///
    /// # Returns
    /// An optional response envelope. Return `None` if no response is needed.
    async fn handle_message(&mut self, envelope: Envelope) -> Option<Envelope>;

    /// Handle an interrupt signal.
    ///
    /// Called when another agent (e.g., Reviewer) sends an interrupt.
    /// The agent should:
    /// 1. Stop current work.
    /// 2. Return any partial output.
    /// 3. Transition to `AgentStatus::Interrupted`.
    async fn handle_interrupt(&mut self, reason: &str, severity: u8) -> Option<Envelope>;

    /// Graceful shutdown hook.
    async fn shutdown(&mut self);

    /// Return a cancel token for cooperative mid-task cancellation.
    ///
    /// If the agent supports long-running interruptible operations (e.g.,
    /// LLM token generation), it should return `Some(arc)` pointing to
    /// an `AtomicBool` that is checked between work units.
    ///
    /// The `spawn_agent` loop will set this flag to `true` immediately
    /// when a high-priority signal arrives, even while `handle_message`
    /// is still executing. This enables **cooperative cancellation**
    /// without dropping the in-flight future.
    ///
    /// Agents without long-running tasks can use the default (`None`).
    fn cancel_token(&self) -> Option<Arc<AtomicBool>> {
        None
    }
}

// ---------------------------------------------------------------------------
// AgentHandle
// ---------------------------------------------------------------------------

/// Runtime handle for a running agent.
///
/// Holds separate senders for data messages and control signals,
/// enabling concurrent interrupt delivery even when the agent is
/// busy processing a long-running message.
pub struct AgentHandle {
    /// The agent's identity.
    pub id: AgentId,

    /// Sender end of the agent's inbox channel (normal messages).
    pub inbox_tx: mpsc::Sender<Envelope>,

    /// Sender end of the agent's signal channel (high-priority control).
    ///
    /// Signals sent here bypass the inbox queue and are processed
    /// concurrently with any in-flight `handle_message` call.
    pub signal_tx: mpsc::Sender<Envelope>,

    /// Handle to the agent's Tokio task.
    pub task_handle: tokio::task::JoinHandle<()>,
}

impl AgentHandle {
    /// Send a normal message to this agent.
    pub async fn send(&self, envelope: Envelope) -> Result<(), mpsc::error::SendError<Envelope>> {
        self.inbox_tx.send(envelope).await
    }

    /// Check if the agent's task is still running.
    pub fn is_alive(&self) -> bool {
        !self.task_handle.is_finished()
    }

    /// Send an interrupt signal via the dedicated signal channel.
    ///
    /// This bypasses the inbox queue, ensuring the interrupt is delivered
    /// even if the agent is busy inside a long-running `handle_message`.
    pub async fn interrupt(
        &self,
        from: AgentId,
        reason: impl Into<String>,
        severity: u8,
    ) -> Result<(), mpsc::error::SendError<Envelope>> {
        let signal = Signal::Interrupt {
            target: self.id.clone(),
            reason: reason.into(),
            severity,
        };
        let envelope =
            Envelope::new(from, self.id.clone(), MessagePayload::Signal(signal)).with_priority(0);

        self.signal_tx.send(envelope).await
    }

    /// Send a halt signal via the dedicated signal channel.
    pub async fn halt(&self, from: AgentId) -> Result<(), mpsc::error::SendError<Envelope>> {
        let signal = Signal::Halt {
            target: self.id.clone(),
        };
        let envelope =
            Envelope::new(from, self.id.clone(), MessagePayload::Signal(signal)).with_priority(0);

        self.signal_tx.send(envelope).await
    }
}

// ---------------------------------------------------------------------------
// Control-flow helpers
// ---------------------------------------------------------------------------

/// Control flow result from handling a signal.
enum ControlFlow {
    Continue,
    Shutdown,
}

/// Handle a control signal envelope when the agent is **idle** (not mid-task).
async fn handle_control_signal<A: Agent>(
    agent: &mut A,
    agent_id: &AgentId,
    envelope: Envelope,
    outbox_tx: &mpsc::Sender<Envelope>,
) -> ControlFlow {
    match &envelope.payload {
        MessagePayload::Signal(Signal::Halt { .. }) => {
            info!("Agent {} received HALT signal.", agent_id);
            agent.shutdown().await;
            ControlFlow::Shutdown
        }
        MessagePayload::Signal(Signal::Shutdown) => {
            info!("Agent {} received SHUTDOWN signal.", agent_id);
            agent.shutdown().await;
            ControlFlow::Shutdown
        }
        MessagePayload::Signal(Signal::Interrupt {
            reason, severity, ..
        }) => {
            warn!(
                "Agent {} INTERRUPTED: reason='{}', severity={}",
                agent_id, reason, severity
            );
            if let Some(response) = agent.handle_interrupt(reason, *severity).await {
                let _ = outbox_tx.send(response).await;
            }
            ControlFlow::Continue
        }
        _ => ControlFlow::Continue,
    }
}

/// Check if an envelope contains a control signal.
fn is_control_signal(envelope: &Envelope) -> bool {
    matches!(
        &envelope.payload,
        MessagePayload::Signal(Signal::Interrupt { .. })
            | MessagePayload::Signal(Signal::Halt { .. })
            | MessagePayload::Signal(Signal::Shutdown)
    )
}

/// Check if a signal envelope is a halt or shutdown.
fn is_halt_or_shutdown(envelope: &Envelope) -> bool {
    matches!(
        &envelope.payload,
        MessagePayload::Signal(Signal::Halt { .. }) | MessagePayload::Signal(Signal::Shutdown)
    )
}

// ---------------------------------------------------------------------------
// spawn_agent — launches an agent into a Tokio task
// ---------------------------------------------------------------------------

/// Spawn an agent into its own Tokio task with dual MPSC channels.
///
/// # Dual-Channel Concurrent Interrupt Architecture
///
/// The agent loop has two phases:
///
/// **Phase 1 (Idle):** `select!` waits on both `inbox_rx` and `signal_rx`.
/// Signals are handled immediately via `handle_interrupt()` / `shutdown()`.
/// A normal message transitions to Phase 2.
///
/// **Phase 2 (Processing):** The `handle_message` future is pinned and
/// raced against `signal_rx` in a `select!` loop.
///
/// When a signal arrives during Phase 2:
/// 1. The `cancel_token` (`Arc<AtomicBool>`) is set to `true` immediately —
///    this does **not** require `&mut agent` (it's an `Arc`, not a reference).
/// 2. The signal is pushed into a `deferred_signals` vec.
/// 3. The loop continues polling `handle_message`, which will see the
///    cancel flag at its next yield point and return cooperatively.
/// 4. After `handle_message` completes (the `&mut agent` borrow ends),
///    deferred signals are processed via `handle_interrupt()` / `shutdown()`.
///
/// ## Why this compiles
///
/// The pinned `handle_message` future captures `&mut agent`. While it exists,
/// we cannot call any `&mut self` methods on `agent`. But the signal branch
/// only touches `cancel_token` (an owned `Arc`) and `deferred_signals` (a
/// local `Vec`) — neither borrows `agent`. The `&mut agent` borrow ends when
/// the block containing the pinned future exits, after which deferred signals
/// are processed normally.
///
/// ## Why this is safe (Soft-Kill)
///
/// The `handle_message` future is **never dropped** mid-execution. The
/// `cancel_token` causes cooperative exit at the next token boundary,
/// preserving KV-Cache state and allowing partial result collection.
pub fn spawn_agent<A: Agent>(
    mut agent: A,
    inbox_capacity: usize,
    outbox_tx: mpsc::Sender<Envelope>,
) -> AgentHandle {
    let id = agent.id().clone();
    let (inbox_tx, mut inbox_rx) = mpsc::channel::<Envelope>(inbox_capacity);
    let (signal_tx, mut signal_rx) = mpsc::channel::<Envelope>(16);

    // Extract the cancel token BEFORE moving agent into the task.
    // This is an Arc<AtomicBool> — an owned, cloneable handle that does
    // not borrow the agent. It can be set without &mut agent.
    let cancel_token = agent.cancel_token();

    let agent_id = id.clone();
    let task_handle = tokio::spawn(async move {
        info!("Agent {} started (dual-channel mode).", agent_id);

        'event_loop: loop {
            // ── Phase 1: IDLE — wait for next message or signal ──────
            let envelope = loop {
                tokio::select! {
                    biased; // Always check signals first.

                    signal = signal_rx.recv() => {
                        let Some(signal_env) = signal else {
                            debug!("Agent {} signal channel closed.", agent_id);
                            break 'event_loop;
                        };
                        match handle_control_signal(
                            &mut agent, &agent_id, signal_env, &outbox_tx,
                        ).await {
                            ControlFlow::Continue => continue,
                            ControlFlow::Shutdown => break 'event_loop,
                        }
                    }

                    msg = inbox_rx.recv() => {
                        let Some(envelope) = msg else {
                            debug!("Agent {} inbox closed.", agent_id);
                            break 'event_loop;
                        };
                        // Backward compat: signals may arrive on inbox.
                        if is_control_signal(&envelope) {
                            match handle_control_signal(
                                &mut agent, &agent_id, envelope, &outbox_tx,
                            ).await {
                                ControlFlow::Continue => continue,
                                ControlFlow::Shutdown => break 'event_loop,
                            }
                        }
                        break envelope;
                    }
                }
            };

            // ── Phase 2: PROCESSING — race handle_message vs signals ─
            debug!("Agent {} processing message from {}", agent_id, envelope.from);

            let mut deferred_signals: Vec<Envelope> = Vec::new();
            let mut should_shutdown = false;

            // Scope the pinned future so &mut agent is released at block end.
            let response = {
                let msg_future = agent.handle_message(envelope);
                tokio::pin!(msg_future);

                loop {
                    tokio::select! {
                        biased;

                        signal = signal_rx.recv() => {
                            match signal {
                                None => {
                                    // Signal channel closed mid-task.
                                    // Let handle_message finish, then exit.
                                    should_shutdown = true;
                                    continue;
                                }
                                Some(signal_env) => {
                                    // ── IMMEDIATE EFFECT ──
                                    // Set the cancel token NOW. The in-flight
                                    // handle_message → run_streaming will see
                                    // this at the next token boundary and exit
                                    // cooperatively.
                                    if let Some(ref ct) = cancel_token {
                                        if is_control_signal(&signal_env) {
                                            debug!(
                                                "Agent {} setting cancel token for mid-task signal.",
                                                agent_id
                                            );
                                            ct.store(true, Ordering::SeqCst);
                                        }
                                    }

                                    if is_halt_or_shutdown(&signal_env) {
                                        should_shutdown = true;
                                    }

                                    // Defer full processing (handle_interrupt,
                                    // shutdown) until handle_message completes
                                    // and &mut agent is released.
                                    deferred_signals.push(signal_env);
                                    continue;
                                }
                            }
                        }

                        result = &mut msg_future => {
                            break result;
                        }
                    }
                }
                // msg_future dropped here → &mut agent borrow released.
            };

            // ── Post-processing ─────────────────────────────────────
            // Send the response from handle_message.
            if let Some(response) = response {
                if let Err(e) = outbox_tx.send(response).await {
                    warn!("Agent {} failed to send response: {}", agent_id, e);
                    break;
                }
            }

            // Process deferred signals now that &mut agent is available.
            for signal_env in deferred_signals {
                match handle_control_signal(
                    &mut agent, &agent_id, signal_env, &outbox_tx,
                ).await {
                    ControlFlow::Continue => {}
                    ControlFlow::Shutdown => {
                        should_shutdown = true;
                    }
                }
            }

            if should_shutdown {
                break;
            }
        }

        info!("Agent {} exited.", agent_id);
    });

    AgentHandle {
        id,
        inbox_tx,
        signal_tx,
        task_handle,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Minimal test agent ───────────────────────────────────────────

    struct EchoAgent {
        id: AgentId,
        status: AgentStatus,
        metrics: AgentMetrics,
    }

    impl EchoAgent {
        fn new(name: &str) -> Self {
            Self {
                id: AgentId::new(name, AgentRole::Custom),
                status: AgentStatus::Idle,
                metrics: AgentMetrics::default(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Agent for EchoAgent {
        fn id(&self) -> &AgentId {
            &self.id
        }

        fn status(&self) -> AgentStatus {
            self.status
        }

        fn metrics(&self) -> &AgentMetrics {
            &self.metrics
        }

        async fn handle_message(&mut self, envelope: Envelope) -> Option<Envelope> {
            self.metrics.messages_received += 1;

            // Echo back the text.
            if let MessagePayload::TextChunk { text, .. } = &envelope.payload {
                let response = Envelope::new(
                    self.id.clone(),
                    envelope.from.clone(),
                    MessagePayload::TextChunk {
                        text: format!("echo: {text}"),
                        is_final: true,
                        token_count: 0,
                    },
                );
                self.metrics.messages_sent += 1;
                Some(response)
            } else {
                None
            }
        }

        async fn handle_interrupt(&mut self, _reason: &str, _severity: u8) -> Option<Envelope> {
            self.status = AgentStatus::Interrupted;
            self.metrics.interrupts_received += 1;
            None
        }

        async fn shutdown(&mut self) {
            self.status = AgentStatus::Shutdown;
        }
    }

    // ── Slow agent for testing mid-task interrupts ───────────────────

    struct SlowAgent {
        id: AgentId,
        status: AgentStatus,
        metrics: AgentMetrics,
        cancel_flag: Arc<AtomicBool>,
        was_cancelled: bool,
    }

    impl SlowAgent {
        fn new(name: &str) -> Self {
            Self {
                id: AgentId::new(name, AgentRole::Custom),
                status: AgentStatus::Idle,
                metrics: AgentMetrics::default(),
                cancel_flag: Arc::new(AtomicBool::new(false)),
                was_cancelled: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl Agent for SlowAgent {
        fn id(&self) -> &AgentId {
            &self.id
        }
        fn status(&self) -> AgentStatus {
            self.status
        }
        fn metrics(&self) -> &AgentMetrics {
            &self.metrics
        }

        fn cancel_token(&self) -> Option<Arc<AtomicBool>> {
            Some(Arc::clone(&self.cancel_flag))
        }

        async fn handle_message(&mut self, envelope: Envelope) -> Option<Envelope> {
            self.metrics.messages_received += 1;
            self.status = AgentStatus::Active;
            self.cancel_flag.store(false, Ordering::SeqCst);

            // Simulate long-running work that checks cancel_flag.
            for i in 0..100 {
                if self.cancel_flag.load(Ordering::Relaxed) {
                    self.was_cancelled = true;
                    self.status = AgentStatus::Interrupted;
                    return Some(Envelope::new(
                        self.id.clone(),
                        envelope.from.clone(),
                        MessagePayload::TextChunk {
                            text: format!("partial:{i}"),
                            is_final: true,
                            token_count: i,
                        },
                    ));
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            self.status = AgentStatus::Idle;
            Some(Envelope::new(
                self.id.clone(),
                envelope.from.clone(),
                MessagePayload::TextChunk {
                    text: "complete".into(),
                    is_final: true,
                    token_count: 100,
                },
            ))
        }

        async fn handle_interrupt(&mut self, _reason: &str, _severity: u8) -> Option<Envelope> {
            self.status = AgentStatus::Interrupted;
            self.metrics.interrupts_received += 1;
            None
        }

        async fn shutdown(&mut self) {
            self.status = AgentStatus::Shutdown;
        }
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_spawn_and_send_message() {
        let (outbox_tx, mut outbox_rx) = mpsc::channel::<Envelope>(32);

        let agent = EchoAgent::new("TestEcho");
        let handle = spawn_agent(agent, 32, outbox_tx);

        assert!(handle.is_alive());

        // Send a text chunk.
        let sender_id = AgentId::new("Tester", AgentRole::Custom);
        let msg = Envelope::new(
            sender_id.clone(),
            handle.id.clone(),
            MessagePayload::TextChunk {
                text: "hello".into(),
                is_final: true,
                token_count: 1,
            },
        );
        handle.send(msg).await.unwrap();

        // Receive the echo response.
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), outbox_rx.recv())
            .await
            .unwrap()
            .unwrap();

        if let MessagePayload::TextChunk { text, .. } = &response.payload {
            assert_eq!(text, "echo: hello");
        } else {
            panic!("Expected TextChunk response");
        }

        // Shutdown.
        handle.halt(sender_id).await.unwrap();

        // Wait for the task to finish.
        tokio::time::timeout(std::time::Duration::from_secs(1), handle.task_handle)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn test_interrupt_signal() {
        let (outbox_tx, _outbox_rx) = mpsc::channel::<Envelope>(32);

        let agent = EchoAgent::new("Interruptible");
        let handle = spawn_agent(agent, 32, outbox_tx);

        let sender = AgentId::new("Supervisor", AgentRole::Planner);
        handle
            .interrupt(sender.clone(), "Code quality issue", 9)
            .await
            .unwrap();

        // Give the agent a moment to process.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Agent should still be alive (interrupt doesn't kill it).
        assert!(handle.is_alive());

        // Clean shutdown.
        handle.halt(sender).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), handle.task_handle)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn test_mid_task_interrupt_via_cancel_token() {
        let (outbox_tx, mut outbox_rx) = mpsc::channel::<Envelope>(32);

        let agent = SlowAgent::new("SlowWorker");
        let handle = spawn_agent(agent, 32, outbox_tx);

        let sender = AgentId::new("Supervisor", AgentRole::Planner);

        // Send a task that takes ~1000ms.
        let msg = Envelope::new(
            sender.clone(),
            handle.id.clone(),
            MessagePayload::TextChunk {
                text: "do_work".into(),
                is_final: false,
                token_count: 0,
            },
        );
        handle.send(msg).await.unwrap();

        // Wait a bit, then send interrupt while the task is running.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        handle
            .interrupt(sender.clone(), "Anomaly detected", 9)
            .await
            .unwrap();

        // The agent should return partial results relatively quickly.
        let response = tokio::time::timeout(std::time::Duration::from_secs(2), outbox_rx.recv())
            .await
            .expect("Should receive response before timeout")
            .expect("Channel should not be closed");

        if let MessagePayload::TextChunk { text, .. } = &response.payload {
            // Should be partial, not "complete".
            assert!(
                text.starts_with("partial:"),
                "Expected partial result, got: {text}"
            );
        } else {
            panic!("Expected TextChunk response");
        }

        // Clean shutdown.
        handle.halt(sender).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), handle.task_handle)
            .await
            .unwrap()
            .unwrap();
    }
}
