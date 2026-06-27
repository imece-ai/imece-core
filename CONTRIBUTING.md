# Contributing to IMECE Core

First off, thank you for considering contributing to IMECE Core! It's people like you that make IMECE such a great local-first AI agent framework.

> **Our Mission:** "Local-First Autonomous Agent Framework. Zero API dependency. Zero cloud lock-in. Full sovereignty."

## 🛑 IMPORTANT: Contributor License Agreement (CLA)

IMECE Core uses a **Dual-Licensing** model (AGPL-3.0 for open-source, Commercial for proprietary use).

Because of this, **all contributors must sign our Contributor License Agreement (CLA)** before we can accept any pull requests. This ensures we maintain the necessary copyright ownership to offer commercial licenses, which funds the continued development of this framework.

When you open your first Pull Request, a CLA bot will automatically comment with instructions on how to sign the agreement. **Your PR cannot be merged until the CLA is signed.**

---

## 🛠️ Development Environment Setup

IMECE Core is written in Rust and relies on `llama.cpp` and `ONNX Runtime`.

### Prerequisites

1. **Rust** 1.91+ (2021 edition)
2. **CMake** 3.14+ (Required to build `llama.cpp` from source)
3. **C++ Compiler** (GCC / Clang)
4. **Git**

### Building the Project

We have feature flags to control the build process.

```bash
# Clone the repository
git clone https://github.com/imece-ai/imece-core.git
cd imece-core

# Basic build (CPU only, no llama.cpp)
cargo build

# Build with llama.cpp backend (Downloads and compiles llama.cpp automatically)
cargo build --features llama_backend

# Build with CUDA support (requires Nvidia Toolkit)
cargo build --features llama_backend,cuda
```

> **Note:** Compiling `llama.cpp` can take a while and consumes about ~1.5 GB of RAM per parallel build job. If your system freezes during compilation, lower your build jobs: `IMECE_BUILD_JOBS=1 cargo build --features llama_backend`

### Running Tests

Before submitting a PR, make sure all tests pass:

```bash
cargo test --all-features
```

If you are writing new code, please add corresponding unit tests!

---

## 📝 How to Contribute

### 1. Find an Issue
Look for issues tagged with `good first issue` or `help wanted`. If you want to work on something else, please **open an issue first** to discuss it before spending hours coding. We don't want your hard work to be rejected because it doesn't align with the roadmap!

### 2. Fork and Branch
1. Fork the repository.
2. Create a branch for your feature: `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-123`

### 3. Coding Standards (Rust)
* **Format:** Run `cargo fmt` before committing.
* **Linting:** Run `cargo clippy --all-features -- -D warnings` and fix any warnings.
* **Documentation:** Add `///` rustdoc comments to all new public structs, traits, and functions.
* **Unsafe Code:** Avoid `unsafe` unless absolutely necessary (mostly for FFI with llama.cpp/ONNX). Document *why* `unsafe` was used and how memory safety is guaranteed.

### 4. Commit Messages
Write clear, concise commit messages. We prefer the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format:
* `feat: added new chunking algorithm`
* `fix: resolved memory leak in DMCE engine`
* `docs: updated README with new examples`

### 5. Submit a Pull Request
Push your branch to your fork and open a Pull Request against the `main` branch of IMECE Core. 
* Fill out the Pull Request template provided.
* Ensure CI checks pass.
* **Sign the CLA** when prompted by the bot.

---

## 🛡️ Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to licensing@imece.ai.

Once again, thank you for your contributions! 🦀⚡
