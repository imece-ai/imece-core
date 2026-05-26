# IMECE Core

**Local-First Autonomous Agent Framework for Edge Devices**

IMECE Core is a Rust-native framework for running autonomous AI agents entirely on-device. It eliminates API dependencies and cloud lock-in by binding directly to [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference, running local embedding models via ONNX Runtime, and orchestrating multi-agent workflows through an async actor system — all optimized for machines with ≤8 GB VRAM.

```
cargo add imece_core
```

> **Status:** v0.1.0 — core architecture is implemented and tested. The framework compiles and runs on Linux, macOS, and WSL2. CUDA GPU offloading is supported via a feature flag.

---

## Why IMECE?

Most agent frameworks require cloud APIs for inference and embeddings. IMECE takes a different approach:

| Concern | Cloud Agent Frameworks | IMECE Core |
|---|---|---|
| **Inference** | OpenAI / Anthropic API calls | llama.cpp FFI — runs any GGUF model locally |
| **Embeddings** | Remote embedding endpoints | Voyage-4 Nano via ONNX Runtime — fully on-device |
| **Memory** | External vector DB (Pinecone, Weaviate) | Built-in LanceDB store with DMCE chain evolution |
| **Agent orchestration** | HTTP microservices | In-process Tokio MPSC actor system |
| **Error recovery** | Prompt-reprompt loops (expensive) | KV-Cache "Time Travel" rollback (zero-cost) |
| **Sandbox** | Docker / cloud functions | Linux namespace jails via `unshare(1)` |
| **Privacy** | Data leaves the device | Nothing ever leaves the device |

---

## Architecture

IMECE Core is organized into four modules that compose into a complete agent pipeline. The diagram below shows the runtime data flow from the client application down through the agent swarm, into the inference and memory engines, and finally to the shared embedding subsystem:

```text
                                ┌────────────────────────────────────┐
                                │        Client Application          │
                                └────────────────┬───────────────────┘
                                                 │ (MPSC Channels)
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────┐
     │                          MODULE 3 — ACTOR SWARM                                    │
     │  ┌───────────────────────┐   (Tokio MPSC)   ┌───────────────────────┐              │
     │  │      Agent A (e.g.    │ ◄──────────────► │      Agent B (e.g.    │              │
     │  │      Planner)         │                  │      Executor)        │              │
     │  └───────────┬───────────┘                  └───────────┬───────────┘              │
     └──────────────┼──────────────────────────────────────────┼──────────────────────────┘
                    │                                          │
                    │ (Sandbox Run)                            │ (Query / Store Memory)
                    ▼                                          ▼
     ┌─────────────────────────────┐            ┌─────────────────────────────┐
     │  MODULE 2 — INFERENCE       │            │  MODULE 1 — MEMORY          │
     │  ┌───────────────────────┐  │            │  ┌───────────────────────┐  │
     │  │   Sandbox Executor    │  │            │  │   Chain-of-Memory     │  │
     │  │  (Bubblejail/unshare) │  │            │  │   (DMCE / APT)        │  │
     │  └───────────┬───────────┘  │            │  └───────────┬───────────┘  │
     │              │ (Error?)     │            │              │ (Vectors)    │
     │              ▼              │            │              ▼              │
     │  ┌───────────────────────┐  │            │  ┌───────────────────────┐  │
     │  │   KV-Cache Rollback   │  │            │  │    LanceDB Store      │  │
     │  │   ("Time Travel")     │  │            │  │    (Apache Arrow)     │  │
     │  │   llama.cpp FFI       │  │            │  └───────────────────────┘  │
     │  └───────────────────────┘  │            │                             │
     └──────────────┬──────────────┘            └──────────────┬──────────────┘
                    │                                          │
                    ▼ (ort / ONNX)                             ▼ (ort / ONNX)
     ┌────────────────────────────────────────────────────────────────────────────────────┐
     │                        MODULE 4 — EMBEDDING SUBSYSTEM                              │
     │    Voyage-4 Nano (ONNX Runtime)  ──►  Matryoshka Truncation (MRL, 256-d)           │
     │                                  ──►  Native int8 Quantization (QAT)               │
     └────────────────────────────────────────────────────────────────────────────────────┘
```

### Module 1 — Memory (Chain-of-Memory)

Implements the chain-building algorithm from [Chain-of-Memory (arXiv:2601.14287v1)](https://arxiv.org/abs/2601.14287v1) for constructing contextually coherent memory chains from a flat-index vector store.

**Key components:**

- **`MemoryNode`** — Atomic memory unit: `m = (x, τ, ρ, e)` where `x` is text, `τ` is timestamp, `ρ` is role (user/agent/system), and `e ∈ ℝ^d` is the embedding vector.
- **`MemoryStore`** — In-memory flat-index with brute-force cosine similarity retrieval. Suitable for ≤10k nodes.
- **`LanceMemoryStore`** — Persistent vector store backed by [LanceDB](https://lancedb.com/) (Rust-native, Arrow columnar format). Supports IVF-PQ ANN indexing for larger datasets.
- **`DmceEngine`** — Implements Dynamic Memory Chain Evolution (DMCE):
  1. Retrieve Top-K candidates via cosine similarity.
  2. Iteratively select the candidate maximizing the **gating score**: `S_gate(m) = cos(m.e, q) × cos(m.e, C_z)` — a multiplicative gate enforcing both global relevance and contextual consistency.
  3. **Adaptive Path Truncation (APT)** terminates the chain when the score drops sharply: `s*_t < β × s_{t-1}`, preventing semantic drift and VRAM bloat.

### Module 2 — Inference (KV-Cache Rollback & Execution Guided Generation)

The core differentiator: instead of expensive prompt-reprompt loops, IMECE directly manipulates the llama.cpp KV-Cache memory to correct errors mid-generation.

**The "Time Travel" Protocol:**

```
┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌──────────────┐
│   GENERATE   │────▶│ INTERCEPT │────▶│ EXECUTE  │────▶│  EVALUATE    │
│ (tokens)     │     │ (stop seq)│     │ (sandbox)│     │ (success/err)│
└──────────────┘     └───────────┘     └──────────┘     └──┬───────────┘
      ▲                                                     │
      │                  ┌───────────────────┐              │
      └──────────────────│  KV-CACHE ROLLBACK │◀────────────┘
                         │  ("Time Travel")   │   (on error)
                         └───────────────────┘
```

1. **Generate** tokens `t_0..t_n` via llama.cpp.
2. **Intercept** at stop sequences (e.g., `</action>`).
3. **Execute** the action payload in a sandboxed environment.
4. **On failure:** identify token `t_k` where the error began, call `llama_memory_seq_rm(seq_id, t_k, t_n)` to erase the erroneous KV-Cache range, inject an `Observation: <error log>` directly at `t_k`, and resume generation.

The LLM experiences this as "thinking mid-sentence, realizing a mistake, and correcting it instantly." Zero context bloat, zero prompt recalculation overhead.

**Key components:**

- **`KvCacheManager`** (trait) — Abstraction over llama.cpp KV-Cache operations (`seq_rm`, `inject_tokens`, `clear_all`).
- **`KvCacheController`** — High-level rollback orchestrator with bounds checking and telemetry.
- **`LlmBackend`** (trait) — Backend-agnostic interface for tokenization, detokenization, and next-token generation.
- **`LlamaCppBackend`** — Production FFI binding to llama.cpp. Handles model loading, batch decoding, sampler chain management, and GPU layer offloading.
- **`AsyncLlamaBackend`** — Tokio-safe wrapper that offloads blocking FFI calls (~50–500ms per `llama_decode`) to `spawn_blocking`, preventing Tokio runtime stalls.
- **`InferenceEngine`** — Orchestrates the full Generate → Intercept → Execute → Evaluate loop with configurable retry limits.
- **`ActionExecutor`** (trait) — Interface for sandboxed code execution.
- **`ProcessExecutor`** — Basic process-level isolation via `std::process::Command`.
- **`BubblejailExecutor`** — Linux namespace sandbox using `unshare(1)` with PID/network/mount/IPC/user namespace isolation.
- **`ResilientExecutor`** — Production executor that probes namespace support at startup and gracefully falls back to `ProcessExecutor` when unprivileged user namespaces are blocked (Ubuntu 24.04+, Debian 12+). Provides detailed distro-specific remediation guidance.

### Module 3 — Actor (Multi-Agent Swarm)

An asynchronous multi-agent system built on Tokio MPSC channels where agents communicate exclusively through typed message envelopes — no shared mutable state.

**Dual-Channel Concurrent Interrupt Architecture:**

Each agent runs with two MPSC channels:
- **`inbox_rx`** — Normal data messages (tasks, text chunks, reviews).
- **`signal_rx`** — High-priority control signals (Interrupt, Halt, Shutdown).

When an agent is busy inside `handle_message()`, the event loop uses `tokio::select!` to race the in-flight future against `signal_rx`. On interrupt:
1. The agent's `cancel_token` (`Arc<AtomicBool>`) is set immediately — no `&mut self` borrow required.
2. The signal is deferred until `handle_message` completes cooperatively.
3. The in-flight future is **never dropped** (soft-kill), preserving KV-Cache state.

**Key components:**

- **`Agent`** (trait) — Core agent interface with `handle_message`, `handle_interrupt`, and `shutdown` lifecycle hooks.
- **`AgentHandle`** — Runtime handle holding separate data and signal senders for concurrent interrupt delivery.
- **`SwarmEngine`** — Central orchestrator that manages agent lifecycles, routes envelopes, handles interrupts, and bridges to Module 2's KV-Cache rollback via a configurable interrupt handler callback.
- **`EscalationPipeline`** — Typestate-driven analysis pipeline for progressive code review:
  - **Stage 0 (Heuristic):** Regex/substring checks — ~0ms, 0 tokens.
  - **Stage 1 (Syntax):** Structural analysis — ~0ms, 0 tokens.
  - **Stage 2 (LLM):** Semantic analysis on an isolated `seq_id` using the `EscalationRequest` protocol.
  
  Each stage returns a `Verdict` (Pass/Fail/Uncertain). Pass and Fail short-circuit; Uncertain escalates to the next stage with accumulated hints.

**Predefined agent roles:** `Coder`, `Reviewer`, `Planner`, `Executor`, `Custom`.

### Module 4 — Embedding (Voyage-4 Nano)

Local embedding engine powered by the **Voyage-4 Nano** model (180M non-embedding + 160M embedding parameters) running via ONNX Runtime. Zero Python runtime dependency.

**Pipeline:**

```
Input Text → Task Prompt → Tokenize → ONNX Inference (ℝ^2048)
  → MRL Truncation (ℝ^256) → L2 Normalize → int8 Quantize (ℤ^256)
```

**Optimizations:**

- **Matryoshka Representation Learning (MRL):** The model's 2048-d output can be truncated to 256 dimensions with minimal retrieval quality loss — 8× storage reduction.
- **Native int8 Quantization (QAT):** Quantization-aware training enables int8 output vectors — 4× further reduction vs. float32, yielding a combined 32× compression.
- **Task-specific prompts:** Separate prefixes for queries (`"Represent the query for retrieving supporting documents: "`) and documents (`"Represent the document for retrieval: "`).

---

## Building

### Prerequisites

- **Rust** 1.70+ (2021 edition)
- **CMake** 3.14+ (for llama.cpp compilation)
- **Git** (llama.cpp is cloned automatically if `LLAMA_CPP_DIR` is not set)
- **C++ compiler** (GCC / Clang)

### CPU-Only Build

```bash
# Build without the llama.cpp backend (embedding + memory + actor modules only)
cargo build

# Build with llama.cpp backend (full framework)
cargo build --features llama_backend
```

### CUDA GPU Build

```bash
# Full build with CUDA GPU offloading
cargo build --features llama_backend,cuda

# Control GPU layer count at runtime
IMECE_GPU_LAYERS=30 cargo run --features llama_backend,cuda

# Force CPU-only on a CUDA build
IMECE_GPU_LAYERS=0 cargo run --features llama_backend,cuda
```

### Build Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `LLAMA_CPP_DIR` | Path to a pre-built llama.cpp source tree | Auto-clones tag `b5604` |
| `IMECE_GPU_LAYERS` | Number of model layers to offload to GPU | `99` (all) with `cuda`, `0` without |
| `IMECE_BUILD_JOBS` | Parallel CMake compilation jobs | Auto-detected from available RAM (1 job per 1.5 GB) |

> **Note on low-RAM machines:** llama.cpp compilation can consume ~1.5 GB per translation unit. The build system auto-detects available memory and throttles parallelism to prevent OOM crashes.

### Embedding Model Setup

```bash
pip install huggingface_hub transformers
python models/export_voyage_nano.py
```

This places the ONNX model at `models/voyage-4-nano-onnx/model.onnx` and the tokenizer at `models/voyage-4-nano-onnx/tokenizer.json`.

### Running Tests

```bash
# All tests (no llama_backend feature needed for unit tests)
cargo test

# With llama.cpp backend tests
cargo test --features llama_backend
```

---

## Usage

### Memory — Building a Chain-of-Memory

```rust
use imece_core::memory::store::MemoryStore;
use imece_core::memory::node::{MemoryNode, Role};
use imece_core::memory::chain::DmceEngine;
use ndarray::Array1;

// Create an in-memory store (dimension must match your embedding model)
let mut store = MemoryStore::new_in_memory(256).unwrap();

// Insert memory nodes with embeddings
let node = MemoryNode::new(
    "Rust's ownership model prevents data races at compile time.".into(),
    Role::Agent,
    Array1::from_vec(embedding_vector),  // from Module 4
);
store.insert(&node).unwrap();

// Build a memory chain using DMCE
let engine = DmceEngine::new(
    0.6,   // β — APT truncation threshold
    20,    // Top-K candidate pool size
    8,     // Maximum chain length
);
let chain = engine.evolve(&store, &query_embedding);
```

### Inference — Running with KV-Cache Rollback

```rust
use imece_core::inference::engine::InferenceEngine;
use imece_core::inference::types::InferenceConfig;

let config = InferenceConfig {
    max_tokens: 2048,
    temperature: 0.7,
    max_rollback_retries: 3,
    ..Default::default()
};

let mut engine = InferenceEngine::new(backend, kv_controller, executor, config);
let session = engine.run("Write a Python function to sort a list").await?;

println!("Output: {}", session.final_text);
println!("Rollbacks: {}", session.total_rollbacks);
```

### Actor — Multi-Agent Swarm

```rust
use imece_core::actor::engine::SwarmEngine;
use imece_core::actor::types::*;

let mut swarm = SwarmEngine::new(64, 32); // outbox + inbox capacity

// Spawn agents (each runs in its own Tokio task)
let coder_id = swarm.spawn(my_coder_agent);
let reviewer_id = swarm.spawn(my_reviewer_agent);

// Set up the Module 2 bridge — interrupts trigger KV-Cache rollback
swarm.set_interrupt_handler(|signal| {
    // Trigger KV-Cache rollback when a Reviewer raises an interrupt
});

// Run the message routing loop
swarm.run().await;
```

### Embedding — Local Vector Generation

```rust
use imece_core::embedding::config::{EmbeddingConfig, MrlDimension, OutputPrecision};
use imece_core::embedding::engine::VoyageNanoEngine;

let config = EmbeddingConfig {
    model_dir: "models/voyage-4-nano-onnx".into(),
    mrl_dimension: MrlDimension::D256,
    output_precision: OutputPrecision::Int8,
    num_threads: 4,
    max_length: 512,
};

let engine = VoyageNanoEngine::new(config)?;

// Embed a query (automatically prepends the query task prompt)
let query_emb = engine.embed_query("How does Rust prevent data races?")?;
let embedding_f32 = query_emb.to_f32(); // Array1<f32> for MemoryStore

// Embed a document (automatically prepends the document task prompt)
let doc_emb = engine.embed_document("Rust uses an ownership system...")?;
```

---

## Project Structure

```
imece-core/
├── Cargo.toml              # Dependencies & feature flags
├── build.rs                # llama.cpp auto-build (CMake, git clone, GPU detection)
├── LICENSE                 # AGPL v3
└── src/
    ├── lib.rs              # Crate root — re-exports all modules
    ├── memory/
    │   ├── node.rs         # MemoryNode (x, τ, ρ, e) definition
    │   ├── store.rs        # In-memory brute-force vector store
    │   ├── lance_store.rs  # LanceDB persistent vector store
    │   ├── chain.rs        # DMCE algorithm & APT truncation
    │   └── error.rs        # Memory error types
    ├── inference/
    │   ├── types.rs         # Token, GenerationState, RollbackTarget, InferenceConfig
    │   ├── kv_cache.rs      # KvCacheManager trait & KvCacheController
    │   ├── engine.rs        # InferenceEngine — the execution sandbox loop
    │   ├── executor.rs      # ActionExecutor trait & ProcessExecutor
    │   ├── sandbox_executor.rs  # BubblejailExecutor & ResilientExecutor
    │   ├── backend.rs       # LlamaCppBackend, LlamaCppKvCache, AsyncLlamaBackend
    │   ├── ffi.rs           # Raw llama.cpp C-API FFI bindings
    │   └── error.rs         # Inference error types
    ├── actor/
    │   ├── types.rs         # AgentId, Signal, Envelope, MessagePayload
    │   ├── agent.rs         # Agent trait, AgentHandle, spawn_agent
    │   ├── engine.rs        # SwarmEngine — central message router
    │   └── escalation.rs    # EscalationPipeline & AnalysisStage trait
    └── embedding/
        ├── config.rs        # MrlDimension, OutputPrecision, EmbeddingConfig
        ├── engine.rs        # VoyageNanoEngine — ONNX inference pipeline
        └── error.rs         # Embedding error types
```

---

## Dependencies

| Crate | Purpose |
|---|---|
| `ndarray` | Linear algebra for DMCE computation & embeddings |
| `lancedb` + `arrow-*` + `lance-arrow` | Rust-native vector database for persistent memory |
| `tokio` | Async runtime for the actor system & LanceDB ops |
| `ort` | ONNX Runtime binding for Voyage-4 Nano inference |
| `tokenizers` | HuggingFace Rust-native tokenization |
| `serde` / `serde_json` | Serialization for memory nodes & configuration |
| `thiserror` | Ergonomic error type definitions |
| `async-trait` | Async trait support for Agent & LlmBackend traits |
| `tracing` | Structured logging throughout all modules |
| `uuid` | Unique identifiers for memory nodes & agents |
| `tempfile` | Ephemeral directories for sandbox execution & in-memory LanceDB |
| `futures` | Async stream collection for LanceDB query results |
| `cmake` | Build-time llama.cpp compilation (build dependency) |

---

## Roadmap

### Areas for Improvement

1. **Pluggable Embedding Backend**
   The `embedding` module is currently built around the Voyage-4 Nano model. The architecture will be refactored to provide a generic `EmbeddingBackend` trait, allowing other embedding models (e.g., `all-MiniLM-L6-v2`, `nomic-embed`, `BGE-M3`) to be integrated as drop-in replacements.

2. **Proactive KV-Cache Rollback Error Prevention**
   The "Time Travel" rollback protocol can currently enter error loops when repeated rollbacks fail at the same position. This is mitigated by a configurable retry limit (`max_rollback_retries`), but a more robust solution is planned: injecting pre-emptive context about the error *cause* into the LLM prompt before the rollback point, so the model avoids regenerating the same faulty pattern.

3. **Python Wrapper (PyIMECE)**
   A Python binding via PyO3/maturin will be provided so that AI/ML engineers can use IMECE Core from Python with a familiar API, while still benefiting from Rust's performance for the underlying inference, memory, and actor operations.

### Areas for Improvement(The following are LLM recommendations and will be evaluated.)

4. **Batch ONNX Inference for Embeddings**
   The current embedding engine processes texts one-at-a-time. Implementing true batch inference at the ONNX level (padding + batched forward pass) would significantly improve throughput when indexing large document collections.

5. **Persistent Agent State**
   Agents currently lose state on shutdown. Serializable agent checkpoints (backed by LanceDB or a lightweight KV store) would enable long-running agent sessions that survive process restarts.

6. **seccomp-bpf for Sandbox Hardening**
   The `BubblejailExecutor` currently uses namespace isolation. Adding a seccomp-bpf syscall filter would further restrict the kernel attack surface of sandboxed code execution.

7. **macOS Sandbox Support**
   The `ActionExecutor` trait is designed for cross-platform sandboxing. A `SeatbeltExecutor` using macOS's `sandbox-exec` API is planned for native macOS support without Docker.

8. **Windows Job Objects Executor**
   Similarly, a `JobObjectExecutor` using Windows Job Objects API would provide native process isolation on Windows without WSL2.

---

## License

AGPL v3 — see [LICENSE](LICENSE) for full text.
