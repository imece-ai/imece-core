//! # Actor Model — Asynchronous Multi-Agent Swarm (Module 3)
//!
//! Implements the MPSC-based actor system using Rust's `tokio` runtime
//! for maximum CPU/GPU utilization via decoupled agent states.
//!
//! ## Architecture
//!
//! - **Message Passing:** MPSC (Multi-Producer, Single-Consumer) channels.
//! - **State Segregation:** Agents do NOT share global variables.
//! - **Interruptibility:** Any agent can send `INTERRUPT_SIGNAL` to the
//!   Core Engine, which halts generation and triggers KV-Cache Rollback.
//!
//! ## Submodules
//! - [`types`]: Message envelopes, agent identifiers, signals.
//! - [`agent`]: The `Agent` trait and `AgentHandle`.
//! - [`engine`]: The `SwarmEngine` — core orchestrator.
//! - [`escalation`]: Typestate escalation pipeline for intelligent analysis.

pub mod agent;
pub mod engine;
pub mod escalation;
pub mod types;
