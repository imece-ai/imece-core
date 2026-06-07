<div align="center">
  <h1>IMECE Core</h1>
</div>

<div align="center">
  <h3>Local-First Autonomous Agent Framework for Edge Devices</h3>
</div>

<div align="center">
  <a href="https://crates.io/crates/imece_core" target="_blank"><img src="https://img.shields.io/crates/v/imece_core" alt="Crates.io"></a>
  <a href="https://docs.rs/imece_core" target="_blank"><img src="https://img.shields.io/docsrs/imece_core" alt="docs.rs"></a>
  <a href="LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://x.com/imeceai" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/imeceai.svg?style=social&label=Follow%20%40imeceai" alt="Twitter / X"></a>
</div>

<br>

IMECE Core is a Rust-native framework for running autonomous AI agents entirely on-device. It eliminates API dependencies and cloud lock-in by binding directly to [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference, running local embedding models via ONNX Runtime, and orchestrating multi-agent workflows through an async actor system — all optimized for machines with ≤8 GB VRAM.

> [!TIP]
> **Status:** v0.1.0 — core architecture is implemented and tested. The framework compiles and runs on Linux, macOS, and WSL2. CUDA GPU offloading is supported via a feature flag.

## Quickstart

```bash
cargo add imece_core
```

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

## Modularity & Flexibility

IMECE Core is designed as a set of decoupled, pluggable building blocks. You are not forced to use all modules together. Depending on your project requirements:

- **Optional Memory (Chain-of-Memory):** You can bypass Module 1 (Memory) entirely. If your agents do not require contextual memory chains or dynamic evolution, you can pass plain text strings or simple message histories as context directly into your tasks.
- **Pluggable Vector Stores:** The framework provides a built-in LanceDB store and in-memory stores, but you can swap them out for any external vector database (like pgvector, Qdrant, or Pinecone) by generating embeddings via Module 4 (Embedding) and querying/storing them in your preferred database.
- **Standalone Inference:** You can use Module 2 (llama.cpp backend + KV-Cache rollback) on its own to build stateless, single-agent sandboxed execution environments without spawning an actor swarm.
- **Standalone Embeddings:** You can run local embedding generation using Voyage-4 Nano via ONNX Runtime without loading any LLM inference backend.

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

### Module 4 — Embedding Subsystem

Pluggable embedding architecture (`EmbeddingBackend` trait) with a primary local engine powered by the **Voyage-4 Nano** model (180M non-embedding + 160M embedding parameters) running via ONNX Runtime. Zero Python runtime dependency.

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

You must provide an ONNX-exported Voyage-4 Nano model (or similar) and its tokenizer. Place the `model.onnx` and `tokenizer.json` files in a directory of your choice, and pass that directory path to `VoyageNanoConfig::model_dir`.

### Running Tests

```bash
# Run unit tests
cargo test
```

*(Note: Integration tests and runnable examples are maintained in the parent workspace.)*

---

## Examples

Looking for ready-to-run code? We maintain a dedicated repository with practical, progressively complex examples:

👉 **[imece-examples](https://github.com/imece-ai/imece-examples)**

The examples repo demonstrates how to use the framework's standalone modules (like generating local embeddings or running persistent semantic search) and will soon include advanced multi-agent orchestrations.

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

#### Concrete Wiring: Constructing an `InferenceEngine` with Real Types

The example above uses abstract variable names. Here is the full dependency injection
showing how to construct each concrete component and wire them together:

```rust
use std::sync::Arc;
use imece_core::inference::backend::{LlamaCppBackend, LlamaCppKvCache, AsyncLlamaBackend};
use imece_core::inference::kv_cache::KvCacheController;
use imece_core::inference::sandbox_executor::ResilientExecutor;
use imece_core::inference::engine::InferenceEngine;
use imece_core::inference::types::InferenceConfig;

// Step 1: Load the llama.cpp backend (blocking FFI — do this before async code).
//   - model_path: path to a GGUF model file
//   - n_ctx: context window size (0 = use model default)
//   - n_threads: CPU threads for inference
let backend = Arc::new(LlamaCppBackend::load("models/qwen2.5-7b.Q4_K_M.gguf", 4096, 8)?);

// Step 2: Create the Tokio-safe async wrapper.
//   AsyncLlamaBackend offloads blocking llama_decode calls (~50-500ms)
//   to spawn_blocking, preventing Tokio runtime stalls.
let async_backend = AsyncLlamaBackend::new(Arc::clone(&backend));

// Step 3: Create the KV-Cache manager from the same Arc<LlamaCppBackend>.
//   The Arc ensures the llama context pointer remains valid as long as
//   either the async_backend or kv_cache holds a reference.
//   - seq_id: 0 (primary generation sequence)
//   - initial_position: 0 (empty cache)
let kv_cache = LlamaCppKvCache::new(Arc::clone(&backend), 0, 0);

// Step 4: Wrap the KV-Cache manager in a controller (adds rollback logic + telemetry).
let kv_controller = KvCacheController::new(kv_cache);

// Step 5: Create the sandboxed executor.
//   ResilientExecutor probes namespace support at startup and falls back
//   to ProcessExecutor when unprivileged user namespaces are blocked.
let executor = ResilientExecutor::new();

// Step 6: Configure the inference engine.
let config = InferenceConfig {
    max_tokens: 2048,
    temperature: 0.7,
    max_rollback_retries: 3,
    ..Default::default()
};

// Step 7: Assemble the engine — all four components are injected here.
//   Generic parameters are inferred: InferenceEngine<AsyncLlamaBackend, LlamaCppKvCache, ResilientExecutor>
let mut engine = InferenceEngine::new(async_backend, kv_controller, executor, config);

// Run inference.
let session = engine.run("Write a Python function to sort a list").await?;
println!("Output: {}", session.final_text);
println!("Rollbacks: {}", session.total_rollbacks);
```

#### Injecting the Engine into an `InferenceAgent` (Actor Swarm)

To use the engine inside the multi-agent swarm (Module 3), wrap it in an
`InferenceAgent` and spawn it via `SwarmEngine`:

```rust
use imece_core::actor::engine::SwarmEngine;

// Build the engine as shown above...
let engine = InferenceEngine::new(async_backend, kv_controller, executor, config);

// Create the swarm and get the outbox sender for streaming.
let mut swarm = SwarmEngine::new(64, 32);
let outbox_tx = swarm.outbox_sender();

// Wrap the engine in an InferenceAgent.
// The agent streams TextChunk envelopes via outbox_tx during generation.
let inference_agent = InferenceAgent::new(engine, outbox_tx);

// Spawn the agent — it now runs in its own Tokio task.
let inference_id = swarm.spawn(inference_agent);
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
use imece_core::embedding::config::{EmbeddingServiceConfig, VoyageNanoConfig, MrlDimension, OutputPrecision};

let config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
    model_dir: "models/voyage-4-nano-onnx".into(),
    mrl_dimension: MrlDimension::D256,
    output_precision: OutputPrecision::Int8,
    num_threads: 4,
    max_length: 512,
});

let backend = config.create_backend()?;

// Embed a query (automatically prepends the query task prompt)
let query_emb = backend.embed_query("How does Rust prevent data races?")?;
let embedding_f32 = query_emb.to_f32(); // Array1<f32> for MemoryStore

// Embed a document (automatically prepends the document task prompt)
let doc_emb = backend.embed_document("Rust uses an ownership system...")?;
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
    │   ├── mod.rs          # Memory module root
    │   ├── node.rs         # MemoryNode (x, τ, ρ, e) definition
    │   ├── store.rs        # In-memory brute-force vector store
    │   ├── lance_store.rs  # LanceDB persistent vector store
    │   ├── chain.rs        # DMCE algorithm & APT truncation
    │   └── error.rs        # Memory error types
    ├── inference/
    │   ├── mod.rs           # Inference module root
    │   ├── types.rs         # Token, GenerationState, RollbackTarget, InferenceConfig
    │   ├── kv_cache.rs      # KvCacheManager trait & KvCacheController
    │   ├── engine.rs        # InferenceEngine — the execution sandbox loop
    │   ├── executor.rs      # ActionExecutor trait & ProcessExecutor
    │   ├── sandbox_executor.rs  # BubblejailExecutor & ResilientExecutor
    │   ├── backend.rs       # LlamaCppBackend, LlamaCppKvCache, AsyncLlamaBackend
    │   ├── ffi.rs           # Raw llama.cpp C-API FFI bindings
    │   └── error.rs         # Inference error types
    ├── actor/
    │   ├── mod.rs           # Actor module root
    │   ├── types.rs         # AgentId, Signal, Envelope, MessagePayload
    │   ├── agent.rs         # Agent trait, AgentHandle, spawn_agent
    │   ├── engine.rs        # SwarmEngine — central message router
    │   └── escalation.rs    # EscalationPipeline & AnalysisStage trait
    └── embedding/
        ├── mod.rs           # Embedding module root
        ├── backend.rs       # EmbeddingBackend trait, EmbeddingOutput, math utilities
        ├── config.rs        # EmbeddingServiceConfig, VoyageNanoConfig
        ├── engine.rs        # VoyageNanoEngine implements EmbeddingBackend
        └── error.rs         # Embedding error types (BackendError)
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

1. **Proactive KV-Cache Rollback Error Prevention**
   The "Time Travel" rollback protocol can currently enter error loops when repeated rollbacks fail at the same position. This is mitigated by a configurable retry limit (`max_rollback_retries`), but a more robust solution is planned: injecting pre-emptive context about the error *cause* into the LLM prompt before the rollback point, so the model avoids regenerating the same faulty pattern.

2. **Python Wrapper (PyIMECE)**
   A Python binding via PyO3/maturin will be provided so that AI/ML engineers can use IMECE Core from Python with a familiar API, while still benefiting from Rust's performance for the underlying inference, memory, and actor operations.

3. **Batch ONNX Inference for Embeddings**
   The current embedding engine processes texts one-at-a-time. Implementing true batch inference at the ONNX level (padding + batched forward pass) would significantly improve throughput when indexing large document collections.

4. **Persistent Agent State**
   Agents currently lose state on shutdown. Serializable agent checkpoints (backed by LanceDB or a lightweight KV store) would enable long-running agent sessions that survive process restarts.

5. **seccomp-bpf for Sandbox Hardening**
   The `BubblejailExecutor` currently uses namespace isolation. Adding a seccomp-bpf syscall filter would further restrict the kernel attack surface of sandboxed code execution.

6. **macOS Sandbox Support**
   The `ActionExecutor` trait is designed for cross-platform sandboxing. A `SeatbeltExecutor` using macOS's `sandbox-exec` API is planned for native macOS support without Docker.

7. **Windows Job Objects Executor**
   Similarly, a `JobObjectExecutor` using Windows Job Objects API would provide native process isolation on Windows without WSL2.

---

## License

AGPL v3 — see [LICENSE](LICENSE) for full text.
