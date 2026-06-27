//! # Inference & Reasoning Engine (Module 2)
//!
//! Implements KV-Cache Rollback and Execution Guided Generation for
//! hardware-level self-correction on low-VRAM edge devices.
//!
//! ## The "Time Travel" Protocol
//!
//! Instead of naive "Prompt-Reprompt" loops (which discard context and
//! recalculate from scratch), this module directly mutates the llama.cpp
//! KV-Cache memory block during autoregressive generation:
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
//! 1. **Generate** — tokens `t_0..t_n` via llama.cpp C-API.
//! 2. **Intercept** — pause on stop sequences (e.g., `</action>`).
//! 3. **Execute** — route action payloads to a sandboxed executor.
//! 4. **Evaluate** — on error, trigger KV-Cache rollback to `t_k`,
//!    inject the error observation, and resume generation.
//!
//! The LLM experiences this as "thinking mid-sentence, realizing a mistake,
//! and correcting it instantly" — zero context bloat, zero prompt
//! recalculation overhead.
//!
//! ## Key Components
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`engine::InferenceEngine`] | Orchestrates the full Generate → Intercept → Execute → Evaluate loop |
//! | [`kv_cache::KvCacheController`] | High-level rollback orchestrator with bounds checking and telemetry |
//! | [`backend::LlamaCppBackend`] | Production FFI binding to llama.cpp (model loading, batch decoding, GPU offload) |
//! | [`backend::AsyncLlamaBackend`] | Tokio-safe wrapper that offloads blocking FFI calls to `spawn_blocking` |
//! | [`executor::ProcessExecutor`] | Basic process-level isolation via `std::process::Command` |
//! | [`sandbox_executor::BubblejailExecutor`] | Linux namespace sandbox with PID/network/mount/IPC isolation |
//! | [`sandbox_executor::ResilientExecutor`] | Auto-probing executor that falls back gracefully |
//!
//! ## Feature Flags
//!
//! This module requires the `llama_backend` feature flag to be enabled.
//! Without it, only the trait definitions and types are available.
//!
//! ## Submodules
//! - [`types`]: Core data structures (Token, GenerationState, RollbackTarget).
//! - [`kv_cache`]: KV-Cache manipulation abstraction (rollback, trim, inject).
//! - [`executor`]: Sandboxed action execution (process isolation).
//! - [`sandbox_executor`]: Linux namespace sandbox (Bubblejail) and resilient fallback.
//! - [`backend`]: llama.cpp FFI backend and async wrapper.
//! - [`ffi`]: Raw llama.cpp C-API FFI bindings.
//! - [`engine`]: The orchestrating inference loop.
//! - [`error`]: Inference error types.

pub mod backend;
pub mod engine;
pub mod error;
pub mod executor;
pub mod ffi;
pub mod kv_cache;
pub mod sandbox_executor;
pub mod types;
