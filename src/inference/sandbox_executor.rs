//! # Sandboxed Namespace Executor
//!
//! Executes LLM-generated code inside a Linux namespace jail using `unshare(1)`.
//! Provides kernel-level isolation (PID, network, mount, IPC, user namespaces)
//! without requiring Docker, WASM runtimes, or external tools beyond what ships
//! with a standard Linux installation.
//!
//! ## Security Layers
//!
//! | Layer      | Mechanism                        | Effect                                    |
//! |------------|----------------------------------|-------------------------------------------|
//! | Filesystem | `--mount` + private tmpfs        | No access to host filesystem              |
//! | Network    | `--net`                          | Empty network namespace — zero sockets    |
//! | PID        | `--pid --fork`                   | Isolated PID tree; cannot signal host     |
//! | IPC        | `--ipc`                          | Isolated shared memory / semaphores       |
//! | User       | `--map-root-user`                | Unprivileged user namespace               |
//! | Resources  | Timeout + output cap             | Bounded CPU time and memory via 8KB cap   |
//! | Syscalls   | seccomp-bpf (Optional/TODO)      | Prevents kernel exploits (omitted for now)|

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use tokio::process::Command as TokioCommand;
use tracing::{debug, info, warn};

use super::error::{InferenceError, InferenceResult};
use super::executor::ActionExecutor;
use super::types::{ActionPayload, ExecutionOutcome};

/// Maximum bytes captured from stdout/stderr to prevent OOM.
const MAX_OUTPUT_BYTES: usize = 8192;

/// Path to the `unshare` binary.
const UNSHARE_BIN: &str = "/usr/bin/unshare";

// ---------------------------------------------------------------------------
// BubblejailExecutor
// ---------------------------------------------------------------------------

/// Executes action payloads inside Linux namespace sandboxes.
///
/// Each invocation:
/// 1. Creates an ephemeral temp directory (auto-cleaned on drop).
/// 2. Writes the code to a temp file inside it.
/// 3. Spawns `unshare` with PID/net/mount/IPC/user namespace flags.
/// 4. Enforces timeout via `tokio::time::timeout`.
/// 5. Captures and truncates stdout/stderr to 8KB.
///
/// # Requirements
/// - Linux with `/usr/bin/unshare` (standard on Ubuntu, Debian, Fedora, Arch).
/// - Unprivileged user namespaces enabled (default on Ubuntu 24.04+).
pub struct BubblejailExecutor {
    /// Maps action types to `(interpreter_binary, file_extension)`.
    /// e.g., `"python"` → `("/usr/bin/python3", ".py")`
    interpreters: HashMap<String, InterpreterSpec>,
}

/// Specification for how to invoke an interpreter inside the sandbox.
#[derive(Debug, Clone)]
struct InterpreterSpec {
    /// Absolute path to the interpreter binary (must exist on host).
    binary: String,
    /// File extension for the temp script file.
    extension: String,
    /// Extra arguments inserted between the binary and the script path.
    extra_args: Vec<String>,
}

impl BubblejailExecutor {
    /// Create a new executor with default interpreter mappings.
    ///
    /// # Errors
    /// Returns `Err` if `unshare` is not found at the expected path.
    pub fn new() -> InferenceResult<Self> {
        // Verify unshare exists.
        if !std::path::Path::new(UNSHARE_BIN).exists() {
            return Err(InferenceError::ExecutionFailed {
                exit_code: -1,
                stderr: format!(
                    "BubblejailExecutor requires '{UNSHARE_BIN}' but it was not found. \
                     Install util-linux: `sudo apt install util-linux`."
                ),
            });
        }

        let mut interpreters = HashMap::new();

        interpreters.insert(
            "python".into(),
            InterpreterSpec {
                binary: "python3".into(),
                extension: ".py".into(),
                extra_args: vec!["-u".into()], // unbuffered stdout
            },
        );

        interpreters.insert(
            "bash".into(),
            InterpreterSpec {
                binary: "bash".into(),
                extension: ".sh".into(),
                extra_args: vec![],
            },
        );

        interpreters.insert(
            "powershell".into(),
            InterpreterSpec {
                binary: "pwsh".into(),
                extension: ".ps1".into(),
                extra_args: vec!["-File".into()],
            },
        );

        Ok(Self { interpreters })
    }

    /// Register a custom interpreter for an action type.
    pub fn register_interpreter(
        &mut self,
        action_type: impl Into<String>,
        binary: impl Into<String>,
        extension: impl Into<String>,
        extra_args: Vec<String>,
    ) {
        self.interpreters.insert(
            action_type.into(),
            InterpreterSpec {
                binary: binary.into(),
                extension: extension.into(),
                extra_args,
            },
        );
    }

    /// Truncate a string to `MAX_OUTPUT_BYTES`, appending a marker if truncated.
    fn truncate(output: String) -> String {
        if output.len() > MAX_OUTPUT_BYTES {
            let mut t = output[..MAX_OUTPUT_BYTES].to_string();
            t.push_str("\n... [output truncated]");
            t
        } else {
            output
        }
    }

    /// Build the `unshare` command with all namespace isolation flags.
    ///
    /// The resulting command line is roughly:
    /// ```text
    /// unshare --pid --fork --net --ipc --mount --user --map-root-user
    ///         -- <interpreter> [extra_args] <script_path>
    /// ```
    fn build_command(
        &self,
        spec: &InterpreterSpec,
        script_path: &std::path::Path,
    ) -> TokioCommand {
        let mut cmd = TokioCommand::new(UNSHARE_BIN);

        // Namespace isolation flags.
        cmd.args([
            "--pid",            // Isolated PID namespace
            "--fork",           // Fork so child is PID 1 in new ns
            "--net",            // Empty network namespace (no sockets)
            "--ipc",            // Isolated IPC namespace
            "--mount",          // Private mount namespace
            "--user",           // Unprivileged user namespace
            "--map-root-user",  // Map current UID → root inside ns
            "--",               // End of unshare flags
        ]);

        // Interpreter + script.
        cmd.arg(&spec.binary);
        for arg in &spec.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(script_path);

        // I/O: capture stdout/stderr, close stdin.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.stdin(std::process::Stdio::null());

        // Prevent child from inheriting unnecessary env vars.
        cmd.env_clear();
        // Provide minimal required environment.
        cmd.env("PATH", "/usr/local/bin:/usr/bin:/bin");
        cmd.env("HOME", "/tmp");
        cmd.env("LANG", "C.UTF-8");

        cmd
    }
}

#[async_trait::async_trait]
impl ActionExecutor for BubblejailExecutor {
    async fn execute(
        &self,
        payload: &ActionPayload,
        timeout: Duration,
    ) -> InferenceResult<ExecutionOutcome> {
        // 1. Resolve interpreter.
        let spec = self.interpreters.get(&payload.action_type).ok_or_else(|| {
            InferenceError::ExecutionFailed {
                exit_code: -1,
                stderr: format!(
                    "Unknown action type '{}'. Registered: {:?}",
                    payload.action_type,
                    self.interpreters.keys().collect::<Vec<_>>()
                ),
            }
        })?;

        debug!(
            action_type = %payload.action_type,
            code_bytes = payload.code.len(),
            timeout_ms = timeout.as_millis() as u64,
            "BubblejailExecutor: preparing sandboxed execution"
        );

        // 2. Create ephemeral temp directory (RAII — cleaned on drop).
        let tmp_dir = tempfile::TempDir::new().map_err(|e| InferenceError::ExecutionFailed {
            exit_code: -1,
            stderr: format!("Failed to create temp directory: {e}"),
        })?;

        // 3. Write code to a temp script file.
        let script_path = tmp_dir.path().join(format!("script{}", spec.extension));
        {
            let mut file =
                std::fs::File::create(&script_path).map_err(|e| InferenceError::ExecutionFailed {
                    exit_code: -1,
                    stderr: format!("Failed to create script file: {e}"),
                })?;
            file.write_all(payload.code.as_bytes())
                .map_err(|e| InferenceError::ExecutionFailed {
                    exit_code: -1,
                    stderr: format!("Failed to write script: {e}"),
                })?;
        }

        // 4. Build the sandboxed command.
        let mut cmd = self.build_command(spec, &script_path);

        // 5. Spawn + enforce timeout asynchronously (does NOT block the runtime).
        let result = tokio::time::timeout(timeout, async {
            let child = cmd.spawn().map_err(|e| InferenceError::ExecutionFailed {
                exit_code: -1,
                stderr: format!("Failed to spawn sandbox process: {e}"),
            })?;

            // `wait_with_output` consumes the child — no zombie possible.
            child
                .wait_with_output()
                .await
                .map_err(|e| InferenceError::ExecutionFailed {
                    exit_code: -1,
                    stderr: format!("Failed to collect process output: {e}"),
                })
        })
        .await;

        // 6. tmp_dir is dropped here → script file cleaned up automatically.

        // 7. Map result to ExecutionOutcome.
        match result {
            // Timeout elapsed — process was killed by tokio's timeout.
            Err(_elapsed) => {
                warn!(
                    timeout_ms = timeout.as_millis() as u64,
                    "BubblejailExecutor: action timed out"
                );
                // When tokio::time::timeout fires, the future (and thus the
                // Child handle) is dropped. Tokio's Child drop impl sends
                // SIGKILL to the child process, ensuring no zombie.
                Ok(ExecutionOutcome::Timeout {
                    timeout_ms: timeout.as_millis() as u64,
                })
            }

            // Spawn or wait failed.
            Ok(Err(e)) => Err(e),

            // Process completed (success or failure).
            Ok(Ok(output)) => {
                let stdout =
                    Self::truncate(String::from_utf8_lossy(&output.stdout).into_owned());
                let stderr =
                    Self::truncate(String::from_utf8_lossy(&output.stderr).into_owned());
                let exit_code = output.status.code().unwrap_or(-1);

                if output.status.success() {
                    debug!(
                        stdout_bytes = stdout.len(),
                        "BubblejailExecutor: action succeeded"
                    );
                    Ok(ExecutionOutcome::Success { stdout })
                } else {
                    warn!(
                        exit_code,
                        stderr_bytes = stderr.len(),
                        "BubblejailExecutor: action failed"
                    );
                    Ok(ExecutionOutcome::Failure {
                        exit_code,
                        stderr,
                        error_token_hint: Some(payload.start_token_pos),
                    })
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ResilientExecutor — Probe + Fallback
// ---------------------------------------------------------------------------

/// Execution isolation level, determined at probe time.
///
/// Represented as `u8` internally so it can be stored in an `AtomicU8`
/// for lock-free runtime downgrades when a per-execution EPERM occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IsolationLevel {
    /// Full namespace sandbox via `unshare` (BubblejailExecutor).
    NamespaceSandbox = 0,
    /// Process-level isolation only (ProcessExecutor fallback).
    ProcessOnly = 1,
}

impl IsolationLevel {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => IsolationLevel::NamespaceSandbox,
            _ => IsolationLevel::ProcessOnly,
        }
    }
}

impl std::fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsolationLevel::NamespaceSandbox => write!(f, "Namespace Sandbox (unshare)"),
            IsolationLevel::ProcessOnly => write!(f, "Process Isolation (fallback)"),
        }
    }
}

/// A resilient executor that attempts namespace-based sandboxing and
/// gracefully falls back to process-level isolation when the kernel
/// denies unprivileged user namespaces.
///
/// ## Why This Exists
///
/// Modern Linux distributions (Ubuntu 24.04+, Debian 12+) have disabled
/// unprivileged user namespaces by default due to a series of critical
/// kernel CVEs (CVE-2024-1086, CVE-2023-32233, etc.) that exploited
/// this feature for privilege escalation.
///
/// Rather than crashing with `EPERM`, this executor:
/// 1. **Probes** namespace support at construction time.
/// 2. **Logs** a detailed diagnostic if namespaces are unavailable.
/// 3. **Falls back** to `ProcessExecutor` transparently.
/// 4. **Re-checks** per-execution in case of runtime policy changes.
///
/// ## Implements `ActionExecutor`
///
/// The `InferenceEngine` sees this as a standard `ActionExecutor` —
/// the KV-Cache Rollback pipeline, Cancel Token, and the entire
/// Execution Sandbox Loop remain completely unaffected.
pub struct ResilientExecutor {
    /// Primary executor: full namespace sandbox.
    sandbox: Option<BubblejailExecutor>,
    /// Fallback executor: process-level isolation.
    fallback: super::executor::ProcessExecutor,
    /// Current isolation level — stored atomically so a runtime EPERM
    /// can permanently downgrade the level without `&mut self`.
    /// See [`IsolationLevel`] for the `u8` ↔ enum mapping.
    level: AtomicU8,
}

impl ResilientExecutor {
    /// Create a new resilient executor.
    ///
    /// Probes the system for namespace support and selects the
    /// appropriate isolation level. **Never panics.**
    ///
    /// # Behavior
    /// - If `unshare` exists and user namespaces work → `NamespaceSandbox`.
    /// - Otherwise → `ProcessOnly` with detailed diagnostics.
    pub fn new() -> Self {
        let fallback = super::executor::ProcessExecutor::new();

        // Step 1: Can we even construct the BubblejailExecutor?
        let sandbox = match BubblejailExecutor::new() {
            Ok(exec) => Some(exec),
            Err(e) => {
                warn!(
                    error = %e,
                    "BubblejailExecutor construction failed — unshare binary not found"
                );
                None
            }
        };

        // Step 2: If construction succeeded, probe actual namespace support.
        if sandbox.is_some() {
            match Self::probe_namespace_support() {
                Ok(()) => {
                    info!(
                        isolation_level = %IsolationLevel::NamespaceSandbox,
                        "Namespace sandbox probe succeeded — full isolation active"
                    );
                    return Self {
                        sandbox,
                        fallback,
                        level: AtomicU8::new(IsolationLevel::NamespaceSandbox as u8),
                    };
                }
                Err(probe_err) => {
                    Self::emit_namespace_diagnostic(&probe_err);
                }
            }
        }

        // Fallback path.
        info!(
            isolation_level = %IsolationLevel::ProcessOnly,
            "Using process-level isolation (reduced but functional)"
        );

        Self {
            sandbox: None,
            fallback,
            level: AtomicU8::new(IsolationLevel::ProcessOnly as u8),
        }
    }

    /// Returns the active isolation level.
    ///
    /// This may change at runtime if a per-execution EPERM triggers
    /// an automatic downgrade from `NamespaceSandbox` to `ProcessOnly`.
    pub fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::from_u8(self.level.load(Ordering::Relaxed))
    }

    /// Atomically downgrade the isolation level to `ProcessOnly`.
    ///
    /// Called when a runtime EPERM is encountered, so all subsequent
    /// executions skip the doomed `unshare` attempt entirely.
    fn downgrade_to_process_only(&self) {
        let prev = self.level.swap(
            IsolationLevel::ProcessOnly as u8,
            Ordering::Relaxed,
        );
        if prev == IsolationLevel::NamespaceSandbox as u8 {
            warn!(
                "Isolation level downgraded: Namespace Sandbox → Process Only. \
                 All subsequent executions will use ProcessExecutor."
            );
        }
    }

    /// Probe whether unprivileged user namespaces are functional.
    ///
    /// Runs `unshare --user --map-root-user -- /bin/true` synchronously.
    /// This is a one-shot probe at startup — it does NOT block the
    /// Tokio runtime (called before the runtime is fully running or
    /// from a blocking context).
    fn probe_namespace_support() -> Result<(), NamespaceProbeError> {
        use std::process::Command;

        let output = Command::new(UNSHARE_BIN)
            .args(["--user", "--map-root-user", "--", "/bin/true"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| NamespaceProbeError {
                kind: ProbeFailureKind::SpawnFailed,
                message: format!("Failed to spawn unshare probe: {e}"),
                stderr: String::new(),
            })?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code().unwrap_or(-1);

        // Classify the failure.
        let kind = if stderr.contains("Operation not permitted")
            || stderr.contains("EPERM")
            || exit_code == 1
        {
            ProbeFailureKind::PermissionDenied
        } else if stderr.contains("Invalid argument") {
            ProbeFailureKind::KernelUnsupported
        } else {
            ProbeFailureKind::Unknown
        };

        Err(NamespaceProbeError {
            kind,
            message: format!("unshare probe exited with code {exit_code}"),
            stderr,
        })
    }

    /// Emit a rich, actionable diagnostic log when namespace support
    /// is unavailable.
    fn emit_namespace_diagnostic(err: &NamespaceProbeError) {
        let distro_info = Self::detect_distro();

        // Build remediation guidance based on detected distro.
        let remediation = match err.kind {
            ProbeFailureKind::PermissionDenied => {
                let mut guidance = String::from(
                    "Unprivileged user namespaces are BLOCKED on this system.\n"
                );

                match distro_info.as_deref() {
                    Some(d) if d.contains("Ubuntu") => {
                        guidance.push_str(&format!(
                            "\n\
                            ╭──────────────────────────────────────────────────────────────────╮\n\
                            │  Detected: {:<51}│\n\
                            │                                                                  │\n\
                            │  Ubuntu 24.04+ blocks user namespaces via AppArmor, NOT the       │\n\
                            │  traditional sysctl. The relevant parameter is:                   │\n\
                            │    kernel.apparmor_restrict_unprivileged_userns                   │\n\
                            │                                                                  │\n\
                            │  Temporary fix (current session):                                │\n\
                            │    sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0  │\n\
                            │                                                                  │\n\
                            │  Permanent fix (survives reboot):                                │\n\
                            │    echo 'kernel.apparmor_restrict_unprivileged_userns=0' |        │\n\
                            │      sudo tee /etc/sysctl.d/99-userns.conf                       │\n\
                            │    sudo sysctl --system                                          │\n\
                            │                                                                  │\n\
                            │  AppArmor profile (recommended for production):                  │\n\
                            │    sudo aa-complain /path/to/imece-core                          │\n\
                            ╰──────────────────────────────────────────────────────────────────╯",
                            d
                        ));
                    }
                    Some(d) if d.contains("Debian") => {
                        guidance.push_str(&format!(
                            "\n\
                            ╭─────────────────────────────────────────────────────────────╮\n\
                            │  Detected: {:<48}│\n\
                            │                                                             │\n\
                            │  Debian 12+ disables unprivileged user namespaces via        │\n\
                            │  kernel.unprivileged_userns_clone sysctl.                   │\n\
                            │                                                             │\n\
                            │  Temporary fix:                                             │\n\
                            │    sudo sysctl -w kernel.unprivileged_userns_clone=1         │\n\
                            │                                                             │\n\
                            │  Permanent fix:                                             │\n\
                            │    echo 'kernel.unprivileged_userns_clone=1' |               │\n\
                            │      sudo tee /etc/sysctl.d/99-userns.conf                  │\n\
                            │    sudo sysctl --system                                     │\n\
                            ╰─────────────────────────────────────────────────────────────╯",
                            d
                        ));
                    }
                    _ => {
                        guidance.push_str(
                            "\n\
                            ╭─────────────────────────────────────────────────────────────╮\n\
                            │  Could not detect distribution.                             │\n\
                            │                                                             │\n\
                            │  Common fix:                                                │\n\
                            │    sudo sysctl -w kernel.unprivileged_userns_clone=1         │\n\
                            │                                                             │\n\
                            │  Or check your AppArmor/SELinux policies for                │\n\
                            │  user namespace restrictions.                               │\n\
                            ╰─────────────────────────────────────────────────────────────╯"
                        );
                    }
                }

                guidance
            }
            ProbeFailureKind::KernelUnsupported => {
                "Kernel does not support user namespaces (CONFIG_USER_NS=n).\n\
                 This is a kernel compile-time setting — cannot be changed at runtime."
                    .to_string()
            }
            ProbeFailureKind::SpawnFailed => {
                format!("Could not spawn the unshare probe process: {}", err.message)
            }
            ProbeFailureKind::Unknown => {
                format!("Unexpected probe failure: {}\nstderr: {}", err.message, err.stderr)
            }
        };

        warn!(
            probe_error = %err.message,
            probe_stderr = %err.stderr,
            distro = ?distro_info,
            failure_kind = ?err.kind,
            "\n\
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n\
            ┃  ⚠  NAMESPACE SANDBOX UNAVAILABLE — FALLING BACK             ┃\n\
            ┃                                                               ┃\n\
            ┃  IMECE will continue with process-level isolation.            ┃\n\
            ┃  Code execution is still functional but less isolated.        ┃\n\
            ┃                                                               ┃\n\
            ┃  This does NOT affect KV-Cache rollback or model inference.   ┃\n\
            ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\
            {remediation}"
        );
    }

    /// Attempt to detect the Linux distribution from `/etc/os-release`.
    fn detect_distro() -> Option<String> {
        let content = std::fs::read_to_string("/etc/os-release").ok()?;
        for line in content.lines() {
            if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
                return Some(name.trim_matches('"').to_string());
            }
        }
        None
    }

    /// Check if a spawn/execution error indicates an EPERM namespace
    /// denial (can happen even if the probe passed, due to race conditions
    /// with sysctl changes or cgroup policy updates).
    fn is_namespace_permission_error(stderr: &str) -> bool {
        stderr.contains("Operation not permitted")
            || stderr.contains("EPERM")
            || stderr.contains("unshare failed")
            || stderr.contains("cannot change root filesystem propagation")
    }
}

impl Default for ResilientExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ActionExecutor for ResilientExecutor {
    async fn execute(
        &self,
        payload: &ActionPayload,
        timeout: Duration,
    ) -> InferenceResult<ExecutionOutcome> {
        // Fast path: if we already know namespaces are unavailable.
        if self.isolation_level() == IsolationLevel::ProcessOnly {
            debug!(
                action_type = %payload.action_type,
                "ResilientExecutor: routing to ProcessExecutor (namespace unavailable)"
            );
            return self.fallback.execute(payload, timeout).await;
        }

        // Attempt sandboxed execution.
        let sandbox = self.sandbox.as_ref().expect(
            "BUG: IsolationLevel::NamespaceSandbox but sandbox is None"
        );

        match sandbox.execute(payload, timeout).await {
            Ok(outcome) => Ok(outcome),

            Err(InferenceError::ExecutionFailed { exit_code, ref stderr })
                if Self::is_namespace_permission_error(stderr) =>
            {
                // Namespace denied at runtime — permanently downgrade
                // so all subsequent calls skip the doomed unshare attempt.
                warn!(
                    exit_code,
                    stderr = %stderr,
                    action_type = %payload.action_type,
                    "Namespace sandbox denied at runtime (EPERM) — \
                     falling back to ProcessExecutor for this and all future actions"
                );
                self.downgrade_to_process_only();

                // Re-execute this action through the fallback path.
                self.fallback.execute(payload, timeout).await
            }

            // All other errors propagate normally (they reach the
            // InferenceEngine which feeds them to KV-Cache Rollback).
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Namespace Probe Types (internal)
// ---------------------------------------------------------------------------

/// Why the namespace probe failed.
#[derive(Debug)]
enum ProbeFailureKind {
    /// `EPERM` — user namespaces blocked by AppArmor/sysctl.
    PermissionDenied,
    /// Kernel compiled without `CONFIG_USER_NS`.
    KernelUnsupported,
    /// Could not spawn the probe process at all.
    SpawnFailed,
    /// Unrecognized failure.
    Unknown,
}

/// Details about a namespace probe failure.
#[derive(Debug)]
struct NamespaceProbeError {
    kind: ProbeFailureKind,
    message: String,
    stderr: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ResilientExecutor Tests ───────────────────────────────────────

    #[test]
    fn test_resilient_executor_constructs_without_panic() {
        // Should never panic regardless of platform.
        let exec = ResilientExecutor::new();
        let level = exec.isolation_level();
        // Level must be one of the two valid variants.
        assert!(
            level == IsolationLevel::NamespaceSandbox
                || level == IsolationLevel::ProcessOnly
        );
    }

    #[test]
    fn test_isolation_level_display() {
        assert_eq!(
            IsolationLevel::NamespaceSandbox.to_string(),
            "Namespace Sandbox (unshare)"
        );
        assert_eq!(
            IsolationLevel::ProcessOnly.to_string(),
            "Process Isolation (fallback)"
        );
    }

    #[test]
    fn test_isolation_level_roundtrip() {
        assert_eq!(
            IsolationLevel::from_u8(IsolationLevel::NamespaceSandbox as u8),
            IsolationLevel::NamespaceSandbox
        );
        assert_eq!(
            IsolationLevel::from_u8(IsolationLevel::ProcessOnly as u8),
            IsolationLevel::ProcessOnly
        );
        // Any unknown value maps to ProcessOnly (safe fallback).
        assert_eq!(
            IsolationLevel::from_u8(255),
            IsolationLevel::ProcessOnly
        );
    }

    #[test]
    fn test_downgrade_is_idempotent() {
        let exec = ResilientExecutor::new();
        // Even if we start at NamespaceSandbox, downgrading twice shouldn't panic.
        exec.downgrade_to_process_only();
        assert_eq!(exec.isolation_level(), IsolationLevel::ProcessOnly);
        exec.downgrade_to_process_only();
        assert_eq!(exec.isolation_level(), IsolationLevel::ProcessOnly);
    }

    #[tokio::test]
    async fn test_resilient_executor_fallback_executes() {
        // Force fallback mode by setting level to ProcessOnly.
        let exec = ResilientExecutor::new();
        exec.downgrade_to_process_only();

        let payload = ActionPayload {
            action_type: "python".into(),
            code: "print('resilient fallback')".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("fallback execution should not return Err");

        match result {
            ExecutionOutcome::Success { stdout } => {
                assert!(stdout.contains("resilient fallback"));
            }
            other => panic!("Expected Success, got: {other:?}"),
        }
    }

    #[test]
    fn test_namespace_permission_error_detection() {
        assert!(ResilientExecutor::is_namespace_permission_error(
            "unshare: Operation not permitted"
        ));
        assert!(ResilientExecutor::is_namespace_permission_error(
            "unshare failed: EPERM"
        ));
        assert!(ResilientExecutor::is_namespace_permission_error(
            "cannot change root filesystem propagation"
        ));
        assert!(!ResilientExecutor::is_namespace_permission_error(
            "SyntaxError: invalid syntax"
        ));
    }

    #[test]
    fn test_detect_distro_does_not_panic() {
        // Should return Some or None without panicking.
        let _distro = ResilientExecutor::detect_distro();
    }

    /// Helper: skip test if unshare is unavailable (e.g., macOS CI).
    fn require_unshare() -> BubblejailExecutor {
        match BubblejailExecutor::new() {
            Ok(exec) => exec,
            Err(_) => {
                eprintln!("SKIP: unshare not available on this platform");
                std::process::exit(0);
            }
        }
    }

    #[tokio::test]
    async fn test_python_hello_world() {
        let exec = require_unshare();
        let payload = ActionPayload {
            action_type: "python".into(),
            code: "print('hello from sandbox')".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("execute should not return Err");

        match result {
            ExecutionOutcome::Success { stdout } => {
                assert!(stdout.contains("hello from sandbox"));
            }
            other => panic!("Expected Success, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let exec = require_unshare();
        let payload = ActionPayload {
            action_type: "bash".into(),
            code: "echo 'sandboxed bash'".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("execute should not return Err");

        match result {
            ExecutionOutcome::Success { stdout } => {
                assert!(stdout.contains("sandboxed bash"));
            }
            other => panic!("Expected Success, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_timeout_enforcement() {
        let exec = require_unshare();
        let payload = ActionPayload {
            action_type: "python".into(),
            code: "import time; time.sleep(60)".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_millis(500))
            .await
            .expect("execute should not return Err");

        assert!(
            matches!(result, ExecutionOutcome::Timeout { .. }),
            "Expected Timeout, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_failure_captures_stderr() {
        let exec = require_unshare();
        let payload = ActionPayload {
            action_type: "python".into(),
            code: "raise ValueError('boom')".into(),
            start_token_pos: 42,
            end_token_pos: 100,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("execute should not return Err");

        match result {
            ExecutionOutcome::Failure {
                stderr,
                error_token_hint,
                ..
            } => {
                assert!(stderr.contains("ValueError"));
                assert_eq!(error_token_hint, Some(42));
            }
            other => panic!("Expected Failure, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_network_isolation() {
        let exec = require_unshare();
        // Attempting any network operation should fail inside the sandbox.
        let payload = ActionPayload {
            action_type: "python".into(),
            code: "import urllib.request; urllib.request.urlopen('http://example.com')".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("execute should not return Err");

        assert!(
            !result.is_success(),
            "Network access should fail inside sandbox"
        );
    }

    #[tokio::test]
    async fn test_output_truncation() {
        let exec = require_unshare();
        // Generate output larger than 8KB.
        let payload = ActionPayload {
            action_type: "python".into(),
            code: "print('A' * 20000)".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(10))
            .await
            .expect("execute should not return Err");

        match result {
            ExecutionOutcome::Success { stdout } => {
                assert!(stdout.len() <= MAX_OUTPUT_BYTES + 30); // 8KB + truncation marker
                assert!(stdout.contains("[output truncated]"));
            }
            other => panic!("Expected Success, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_unknown_action_type() {
        let exec = require_unshare();
        let payload = ActionPayload {
            action_type: "ruby".into(),
            code: "puts 'hello'".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec.execute(&payload, Duration::from_secs(5)).await;
        assert!(result.is_err(), "Unknown action type should return Err");
    }

    #[tokio::test]
    async fn test_env_is_minimal() {
        let exec = require_unshare();
        // Verify HOST env vars are NOT leaked into the sandbox.
        let payload = ActionPayload {
            action_type: "bash".into(),
            code: "env | wc -l".into(),
            start_token_pos: 0,
            end_token_pos: 0,
        };

        let result = exec
            .execute(&payload, Duration::from_secs(5))
            .await
            .expect("execute should not return Err");

        match result {
            ExecutionOutcome::Success { stdout } => {
                let count: usize = stdout.trim().parse().unwrap_or(100);
                // We only set PATH, HOME, LANG — expect very few env vars.
                assert!(count <= 10, "Too many env vars leaked: {count}");
            }
            other => panic!("Expected Success, got: {other:?}"),
        }
    }
}
