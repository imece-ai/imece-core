//! # Inference & Reasoning Engine (Module 2)
//!
//! Implements KV-Cache Rollback and Execution Guided Generation for
//! hardware-level self-correction on low-VRAM edge devices.
//!
//! ## Architecture
//!
//! Instead of naive "Prompt-Reprompt" loops, this module directly mutates
//! the `llama.cpp` KV-Cache memory block during autoregressive generation:
//!
//! 1. **Generate** — tokens `t_0..t_n` via llama.cpp C-API.
//! 2. **Intercept** — pause on stop sequences (e.g., `</action>`).
//! 3. **Execute** — route action payloads to a sandboxed executor.
//! 4. **Evaluate** — on error, trigger KV-Cache rollback to `t_k`,
//!    inject the error observation, and resume generation.
//!
//! ## Submodules
//! - [`types`]: Core data structures (Token, GenerationState, RollbackTarget).
//! - [`kv_cache`]: KV-Cache manipulation abstraction (rollback, trim, inject).
//! - [`executor`]: Sandboxed action execution (process isolation).
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
