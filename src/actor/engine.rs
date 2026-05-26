//! # Swarm Engine — Core Multi-Agent Orchestrator
//!
//! The `SwarmEngine` manages the lifecycle of all agents in the swarm:
//!
//! ```text
//! ┌─────────┐   MPSC   ┌──────────────┐   MPSC   ┌──────────┐
//! │ Agent_A  │─────────▶│              │◀─────────│ Agent_B  │
//! │ (Coder)  │◀─────────│ SwarmEngine  │─────────▶│(Reviewer)│
//! └─────────┘           │  (Router)    │          └──────────┘
//!                       │              │
//!                       │  ┌────────┐  │
//!                       │  │Interrupt│  │
//!                       │  │Handler │  │
//!                       │  └────────┘  │
//!                       └──────────────┘
//! ```
//!
//! ## Responsibilities
//! 1. Route envelopes between agents.
//! 2. Handle interrupt signals (halt agent, trigger rollback).
//! 3. Broadcast messages.
//! 4. Monitor agent health (heartbeats).
//! 5. Graceful shutdown.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::agent::{spawn_agent, Agent, AgentHandle};
use super::types::*;

// ---------------------------------------------------------------------------
// SwarmEngine
// ---------------------------------------------------------------------------

/// The central orchestrator for the multi-agent swarm.
///
/// Manages agent lifecycles, routes messages, and handles interrupts.
/// Runs on the Tokio runtime with full async concurrency.
pub struct SwarmEngine {
    /// All registered agents, keyed by their UUID.
    agents: HashMap<uuid::Uuid, AgentHandle>,

    /// The outbox receiver — all agent responses flow through here.
    outbox_rx: mpsc::Receiver<Envelope>,

    /// The outbox sender — cloned and given to each spawned agent.
    outbox_tx: mpsc::Sender<Envelope>,

    /// Channel capacity for agent inboxes.
    inbox_capacity: usize,

    /// Callback for interrupt signals (bridges to Module 2 KV-Cache Rollback).
    interrupt_handler: Option<Box<dyn Fn(&Signal) + Send + Sync>>,

    /// Running swarm metrics.
    pub metrics: SwarmMetrics,
}

/// Swarm-level metrics.
#[derive(Debug, Clone, Default)]
pub struct SwarmMetrics {
    /// Total messages routed.
    pub messages_routed: u64,

    /// Total interrupts processed.
    pub interrupts_processed: u64,

    /// Total broadcasts sent.
    pub broadcasts_sent: u64,

    /// Number of currently active agents.
    pub active_agents: usize,
}

impl SwarmEngine {
    /// Create a new swarm engine.
    ///
    /// # Arguments
    /// * `outbox_capacity` — Buffer size for the central message bus.
    /// * `inbox_capacity` — Buffer size for each agent's inbox.
    pub fn new(outbox_capacity: usize, inbox_capacity: usize) -> Self {
        let (outbox_tx, outbox_rx) = mpsc::channel(outbox_capacity);

        Self {
            agents: HashMap::new(),
            outbox_rx,
            outbox_tx,
            inbox_capacity,
            interrupt_handler: None,
            metrics: SwarmMetrics::default(),
        }
    }

    /// Get a clone of the outbox sender for agents that need to stream
    /// multiple messages during a single `handle_message` call.
    ///
    /// Used by `InferenceAgent` to push `TextChunk` envelopes directly
    /// during token generation without buffering the entire response.
    pub fn outbox_sender(&self) -> mpsc::Sender<Envelope> {
        self.outbox_tx.clone()
    }

    /// Register an interrupt handler callback.
    ///
    /// This bridges the Actor Model (Module 3) to the KV-Cache Rollback
    /// (Module 2): when an agent sends an interrupt, this callback fires,
    /// allowing the core engine to trigger rollback.
    pub fn set_interrupt_handler<F>(&mut self, handler: F)
    where
        F: Fn(&Signal) + Send + Sync + 'static,
    {
        self.interrupt_handler = Some(Box::new(handler));
    }

    /// Spawn a new agent into the swarm.
    ///
    /// The agent is launched in its own Tokio task with MPSC channels
    /// connected to the swarm engine's message bus.
    pub fn spawn<A: Agent>(&mut self, agent: A) -> AgentId {
        let id = agent.id().clone();
        let handle = spawn_agent(agent, self.inbox_capacity, self.outbox_tx.clone());

        info!("Swarm: spawned agent {}", id);
        self.agents.insert(id.uuid, handle);
        self.metrics.active_agents = self.agents.len();

        id
    }

    /// Send a directed message from the engine to a specific agent.
    pub async fn send_to(
        &self,
        target_uuid: &uuid::Uuid,
        envelope: Envelope,
    ) -> Result<(), String> {
        if let Some(handle) = self.agents.get(target_uuid) {
            handle
                .send(envelope)
                .await
                .map_err(|e| format!("Failed to send to agent: {e}"))
        } else {
            Err(format!("Agent {} not found", target_uuid))
        }
    }

    /// Broadcast a message to all agents.
    pub async fn broadcast(&self, envelope: Envelope) {
        debug!("Swarm: broadcasting message from {}", envelope.from);

        for (_, handle) in &self.agents {
            let mut cloned = envelope.clone();
            cloned.to = Some(handle.id.clone());
            let _ = handle.send(cloned).await;
        }
    }

    /// Run the swarm engine's main message routing loop.
    ///
    /// This loop:
    /// 1. Receives messages from the outbox (sent by any agent).
    /// 2. Routes them to the target agent (or broadcasts).
    /// 3. Handles interrupt signals.
    ///
    /// The loop runs until a `Shutdown` signal is received or all
    /// agents have exited.
    pub async fn run(&mut self) {
        info!("Swarm engine started. {} agents active.", self.agents.len());

        while let Some(envelope) = self.outbox_rx.recv().await {
            self.metrics.messages_routed += 1;

            debug!(
                "Swarm routing: {} → {:?} (priority={})",
                envelope.from,
                envelope.to.as_ref().map(|a| a.to_string()),
                envelope.priority
            );

            // Handle signals specially.
            if let MessagePayload::Signal(ref signal) = envelope.payload {
                self.handle_signal(signal).await;

                // Check for shutdown.
                if matches!(signal, Signal::Shutdown) {
                    self.shutdown_all().await;
                    break;
                }

                continue;
            }

            // Route to specific target or broadcast.
            if let Some(ref target) = envelope.to {
                if let Some(handle) = self.agents.get(&target.uuid) {
                    if let Err(e) = handle.send(envelope.clone()).await {
                        warn!("Failed to route to {}: {}", target, e);
                    }
                } else {
                    warn!("Target agent {} not found", target);
                }
            } else {
                // Broadcast.
                self.metrics.broadcasts_sent += 1;
                self.broadcast(envelope).await;
            }

            // Prune dead agents.
            self.prune_dead_agents();
        }

        info!("Swarm engine stopped.");
    }

    /// Handle a signal message.
    async fn handle_signal(&mut self, signal: &Signal) {
        match signal {
            Signal::Interrupt {
                target,
                reason,
                severity,
            } => {
                self.metrics.interrupts_processed += 1;
                warn!(
                    "Swarm: INTERRUPT for {} — reason='{}', severity={}",
                    target, reason, severity
                );

                // Forward to the target agent.
                if let Some(handle) = self.agents.get(&target.uuid) {
                    let from = AgentId::new("SwarmEngine", AgentRole::Custom);
                    let _ = handle.interrupt(from, reason.clone(), *severity).await;
                }

                // Fire the interrupt handler callback (Module 2 bridge).
                if let Some(ref handler) = self.interrupt_handler {
                    handler(signal);
                }
            }

            Signal::Halt { target } => {
                info!("Swarm: HALT signal for {}", target);
                if let Some(handle) = self.agents.get(&target.uuid) {
                    let from = AgentId::new("SwarmEngine", AgentRole::Custom);
                    let _ = handle.halt(from).await;
                }
            }

            Signal::TaskComplete { source, result } => {
                info!("Swarm: Task complete from {} — {:?}", source, result);
            }

            Signal::Heartbeat { source } => {
                debug!("Swarm: Heartbeat from {}", source);
            }

            Signal::RollbackNotification {
                source,
                tokens_erased,
                rollback_position,
            } => {
                info!(
                    "Swarm: Rollback notification from {} — {} tokens erased at pos {}",
                    source, tokens_erased, rollback_position
                );
            }

            Signal::Shutdown => {
                info!("Swarm: SHUTDOWN signal received.");
            }
        }
    }

    /// Send shutdown signals to all agents and wait for them to exit.
    async fn shutdown_all(&mut self) {
        info!("Swarm: shutting down all agents...");

        let from = AgentId::new("SwarmEngine", AgentRole::Custom);

        for (_, handle) in &self.agents {
            let _ = handle.halt(from.clone()).await;
        }

        // Give agents a moment to process the halt signal.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Drop all handles (aborts any still-running tasks).
        self.agents.clear();
        self.metrics.active_agents = 0;
    }

    /// Remove agents whose Tokio tasks have finished.
    fn prune_dead_agents(&mut self) {
        let before = self.agents.len();
        self.agents.retain(|_, handle| handle.is_alive());
        let pruned = before - self.agents.len();

        if pruned > 0 {
            debug!("Swarm: pruned {} dead agents", pruned);
            self.metrics.active_agents = self.agents.len();
        }
    }

    /// Number of currently registered (alive) agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get a reference to an agent handle by UUID.
    pub fn get_agent(&self, uuid: &uuid::Uuid) -> Option<&AgentHandle> {
        self.agents.get(uuid)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::agent::Agent;
    use super::*;

    // ── Counter Agent — counts messages ─────────────────────────────

    struct CounterAgent {
        id: AgentId,
        status: AgentStatus,
        metrics: AgentMetrics,
        count: u32,
    }

    impl CounterAgent {
        fn new(name: &str) -> Self {
            Self {
                id: AgentId::new(name, AgentRole::Custom),
                status: AgentStatus::Idle,
                metrics: AgentMetrics::default(),
                count: 0,
            }
        }
    }

    #[async_trait::async_trait]
    impl Agent for CounterAgent {
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
            self.count += 1;
            self.metrics.messages_received += 1;

            // After 3 messages, send a task complete signal.
            if self.count >= 3 {
                let signal = Signal::TaskComplete {
                    source: self.id.clone(),
                    result: TaskResult::Success {
                        output: format!("Processed {} messages", self.count),
                    },
                };
                Some(Envelope::new(
                    self.id.clone(),
                    envelope.from,
                    MessagePayload::Signal(signal),
                ))
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

    #[tokio::test]
    async fn test_swarm_spawn_and_count() {
        let mut swarm = SwarmEngine::new(64, 32);

        let a1 = CounterAgent::new("Alpha");
        let a2 = CounterAgent::new("Beta");

        swarm.spawn(a1);
        swarm.spawn(a2);

        assert_eq!(swarm.agent_count(), 2);
    }

    #[tokio::test]
    async fn test_swarm_send_to_agent() {
        let mut swarm = SwarmEngine::new(64, 32);

        let agent = CounterAgent::new("Worker");
        let agent_id = swarm.spawn(agent);

        // Send a message.
        let sender = AgentId::new("Test", AgentRole::Custom);
        let envelope = Envelope::new(
            sender,
            agent_id.clone(),
            MessagePayload::TextChunk {
                text: "work".into(),
                is_final: false,
                token_count: 1,
            },
        );

        swarm.send_to(&agent_id.uuid, envelope).await.unwrap();

        // Give the agent time to process.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Agent should still be alive.
        assert!(swarm.get_agent(&agent_id.uuid).unwrap().is_alive());
    }
}
