use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // docs.rs environment guard: skip C++ build and git clone
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:warning=IMECE: docs.rs build detected. Skipping llama.cpp build.");
        return;
    }

    // Register custom cfg so `#[cfg(imece_gpu_available)]` doesn't warn.
    println!("cargo::rustc-check-cfg=cfg(imece_gpu_available)");

    // Gate: only build llama.cpp when the feature is enabled.
    if env::var("CARGO_FEATURE_LLAMA_BACKEND").is_err() {
        return;
    }

    // ── Rerun triggers ──────────────────────────────────────────
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=IMECE_GPU_LAYERS");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── Source resolution ───────────────────────────────────────
    // Priority: LLAMA_CPP_DIR env var > git clone into OUT_DIR
    let llama_dir = if let Ok(dir) = env::var("LLAMA_CPP_DIR") {
        let p = PathBuf::from(dir);
        assert!(
            p.join("CMakeLists.txt").exists(),
            "LLAMA_CPP_DIR does not contain CMakeLists.txt"
        );
        p
    } else {
        let cloned = out_dir.join("llama.cpp");
        if !cloned.join("CMakeLists.txt").exists() {
            let status = Command::new("git")
                .args([
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "b5604", // Pin a known-good tag to prevent upstream API breaks
                    "https://github.com/ggerganov/llama.cpp.git",
                ])
                .arg(&cloned)
                .status()
                .expect("Failed to clone llama.cpp — is git installed?");
            assert!(status.success(), "git clone failed");
        }
        cloned
    };

    // ── CMake build ─────────────────────────────────────────────
    let mut cmake_config = cmake::Config::new(&llama_dir);
    cmake_config
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF"); // Static linking: no LD_LIBRARY_PATH needed

    // ── GPU Backend Selection ────────────────────────────────────────
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        cmake_config.define("GGML_CUDA", "ON");
        println!("cargo:rustc-cfg=imece_gpu_available");
        println!("cargo:warning=IMECE: Building llama.cpp with CUDA support");
    } else {
        println!("cargo:warning=IMECE: Building llama.cpp for CPU-only inference");
    }

    // ── Safe parallel job count ───────────────────────────────────
    // llama.cpp's C++ TUs can consume ~1.5 GB RAM each during compilation.
    // On low-RAM machines, unlimited parallelism triggers the OOM killer
    // and crashes the terminal/IDE. We auto-detect available RAM and
    // calculate a safe job count, with an explicit override available.
    //
    //   IMECE_BUILD_JOBS=4  cargo build --features llama_backend   # manual override
    //
    let jobs = resolve_build_jobs();
    println!("cargo:warning=IMECE: CMake parallel jobs = {jobs}");
    env::set_var("NUM_JOBS", jobs.to_string());

    let dst = cmake_config.build();

    // ── Link search paths ───────────────────────────────────────
    for subdir in &["lib", "lib64", "lib/static", "build/src", "build"] {
        let p = dst.join(subdir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }

    // ── Link libraries ──────────────────────────────────────────
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    // C++ runtime (llama.cpp is C++)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // System dependencies
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");

    // CUDA libraries (only when cuda feature is active)
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
    }
}

// ---------------------------------------------------------------------------
// Build Job Auto-Detection
// ---------------------------------------------------------------------------

/// Determine the maximum safe number of parallel CMake compilation jobs.
///
/// Resolution order:
/// 1. `IMECE_BUILD_JOBS` env var — explicit override.
/// 2. Auto-detect from `/proc/meminfo` (Linux) — 1 job per 1.5 GB available RAM,
///    capped at the number of CPU cores.
/// 3. Fallback — 1 (safe on any machine).
fn resolve_build_jobs() -> usize {
    println!("cargo:rerun-if-env-changed=IMECE_BUILD_JOBS");

    // Priority 1: explicit override.
    if let Ok(val) = env::var("IMECE_BUILD_JOBS") {
        if let Ok(n) = val.parse::<usize>() {
            return n.max(1);
        }
    }

    // Priority 2: auto-detect available RAM (Linux / macOS).
    if let Some(jobs) = detect_safe_jobs_from_system() {
        return jobs;
    }

    // Priority 3: safe fallback.
    1
}

/// Calculate a safe job count (~1.5 GB per compiler process) based on
/// system RAM. Supports Linux and macOS.
fn detect_safe_jobs_from_system() -> Option<usize> {
    let avail_bytes = if cfg!(target_os = "linux") {
        // Read MemAvailable from /proc/meminfo
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        let avail_kb = contents
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?;
        avail_kb * 1024
    } else if cfg!(target_os = "macos") {
        // Run sysctl hw.memsize
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse::<u64>().ok()?
    } else {
        return None;
    };

    let avail_gb = avail_bytes / (1024 * 1024 * 1024);
    let ram_based = (avail_gb as f64 / 1.5).floor() as usize;

    let cpu_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    // At least 1, at most CPU count.
    Some(ram_based.clamp(1, cpu_count))
}
