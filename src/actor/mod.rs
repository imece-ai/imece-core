//! # Actor Model — Asynchronous Multi-Agent Swarm (Module 3)
//!
//! Implements a concurrent multi-agent system using Rust's `tokio` runtime
//! and MPSC (Multi-Producer, Single-Consumer) channels.
//!
//! ## Architecture
//!
//! Agents operate as independent, asynchronous actors communicating solely
//! through typed message envelopes. There is no shared mutable state.
//!
//! ```text
//! ┌───────────────┐        ┌─────────────────┐        ┌─────────────────┐
//! │               │        │                 │        │                 │
//! │  Planner      │───────▶│   SwarmEngine   │───────▶│   Reviewer      │
//! │  (Agent)      │        │  (Orchestrator) │        │   (Agent)       │
//! │               │◀───────│                 │◀───────│                 │
//! └───────────────┘        └─────────────────┘        └─────────────────┘
//!        │                          │                          │
//!        │ inbox_rx                 │ outbox_rx                │ inbox_rx
//!        │ signal_rx                │ signal_rx                │ signal_rx
//! ```
//!
//! ## Dual-Channel Interrupt System
//!
//! Each agent holds two receiver channels:
//! 1. `inbox_rx` for standard data payloads (tasks, text chunks).
//! 2. `signal_rx` for high-priority control signals (Interrupt, Shutdown).
//!
//! This dual-channel design allows the swarm to instantly halt an agent's
//! generation loop if another agent detects a critical error or security
//! violation, triggering KV-Cache Rollback at the engine level without
//! waiting for the current token stream to finish.
//!
//! ## Submodules
//! - [`types`]: Message envelopes, agent identifiers, payloads, and signals.
//! - [`agent`]: The `Agent` trait defining the actor interface, and `AgentHandle`.
//! - [`engine`]: The `SwarmEngine` orchestrator that routes messages and manages lifecycles.
//! - [`escalation`]: Typestate escalation pipeline for progressive, cost-efficient analysis.

pub mod agent;
pub mod engine;
pub mod escalation;
pub mod types;
