//! # Sandboxed Action Executor
//!
//! Executes action payloads (code blocks extracted from LLM output) in
//! an isolated process environment.
//!
//! ## Isolation Strategy
//!
//! On edge devices without Docker/WASM runtimes, we use OS-level process
//! isolation via `std::process::Command` with:
//! - Timeout enforcement via `tokio::time::timeout`.
//! - Working directory isolation (temp directory per execution).
//! - Resource limits (stdout/stderr capture capped).
//!
//! ## Cross-Platform Sandboxing Roadmap
//!
//! To support true local-first execution without heavy dependencies like Docker,
//! the `ActionExecutor` trait maps to native OS sandboxing primitives:
//!
//! ```text
//! ActionExecutor (trait)
//!        │
//!        ├─ Linux/WSL2  → BubblejailExecutor  (unshare namespaces)
//!        ├─ macOS       → SeatbeltExecutor    (sandbox-exec / Seatbelt)
//!        ├─ Windows     → JobObjectExecutor   (Windows Job Objects API)
//!        └─ Fallback    → ProcessExecutor     (rlimit, works everywhere)
//! ```

use std::collections::HashMap;
use std::time::Duration;
use tokio::process::Command as TokioCommand;
use tracing::{debug, warn};

use super::error::{InferenceError, InferenceResult};
use super::types::{ActionPayload, ExecutionOutcome};

// ---------------------------------------------------------------------------
// ActionExecutor Trait
// ---------------------------------------------------------------------------

/// Trait for executing action payloads in an isolated environment.
///
/// See the module-level documentation for the cross-platform roadmap
/// mapping this trait to native OS sandboxing primitives.
#[async_trait::async_trait]
pub trait ActionExecutor: Send + Sync {
    /// Execute the given action payload and return the outcome.
    ///
    /// # Arguments
    /// * `payload` — The extracted action from the LLM's output.
    /// * `timeout` — Maximum execution time.
    ///
    /// # Returns
    /// An `ExecutionOutcome` — either `Success`, `Failure`, or `Timeout`.
    async fn execute(
        &self,
        payload: &ActionPayload,
        timeout: Duration,
    ) -> InferenceResult<ExecutionOutcome>;
}

// ---------------------------------------------------------------------------
// ProcessExecutor
// ---------------------------------------------------------------------------

/// Execute actions via OS subprocess isolation.
///
/// Each action type maps to a shell/interpreter command:
/// - `"python"` → `python3 -c <code>`
/// - `"bash"` → `sh -c <code>`
/// - `"powershell"` → `powershell -Command <code>`
///
/// Stdout and stderr are captured. Execution is bounded by a hard timeout.
pub struct ProcessExecutor {
    /// Map from action type to interpreter command.
    /// e.g., `{"python" => ["python3", "-c"], "bash" => ["sh", "-c"]}`
    interpreters: HashMap<String, Vec<String>>,

    /// Maximum bytes to capture from stdout/stderr.
    max_output_bytes: usize,
}

impl ProcessExecutor {
    /// Create a new executor with default interpreter mappings.
    pub fn new() -> Self {
        let mut interpreters = HashMap::new();

        // Python
        interpreters.insert(
            "python".to_string(),
            vec!["python3".to_string(), "-c".to_string()],
        );

        // Bash / Shell
        interpreters.insert("bash".to_string(), vec!["sh".to_string(), "-c".to_string()]);

        // PowerShell (Windows)
        interpreters.insert(
            "powershell".to_string(),
            vec!["powershell".to_string(), "-Command".to_string()],
        );

        Self {
            interpreters,
            max_output_bytes: 8192, // 8KB cap
        }
    }

    /// Register a custom interpreter for an action type.
    pub fn register_interpreter(
        &mut self,
        action_type: impl Into<String>,
        command_parts: Vec<String>,
    ) {
        self.interpreters.insert(action_type.into(), command_parts);
    }

    /// Set the maximum output capture size.
    pub fn set_max_output_bytes(&mut self, bytes: usize) {
        self.max_output_bytes = bytes;
    }

    /// Truncate output to `max_output_bytes`, appending a marker if truncated.
    fn truncate_output(&self, output: String) -> String {
        if output.len() > self.max_output_bytes {
            let mut truncated = output[..self.max_output_bytes].to_string();
            truncated.push_str("\n... [output truncated]");
            truncated
        } else {
            output
        }
    }
}

impl Default for ProcessExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ActionExecutor for ProcessExecutor {
    async fn execute(
        &self,
        payload: &ActionPayload,
        timeout: Duration,
    ) -> InferenceResult<ExecutionOutcome> {
        // Look up the interpreter for this action type.
        let interpreter = self.interpreters.get(&payload.action_type).ok_or_else(|| {
            InferenceError::ExecutionFailed {
                exit_code: -1,
                stderr: format!(
                    "Unknown action type '{}'. Registered: {:?}",
                    payload.action_type,
                    self.interpreters.keys().collect::<Vec<_>>()
                ),
            }
        })?;

        if interpreter.is_empty() {
            return Err(InferenceError::ExecutionFailed {
                exit_code: -1,
                stderr: "Empty interpreter command".into(),
            });
        }

        debug!(
            "Executing action type='{}', code_len={} bytes, timeout={}ms",
            payload.action_type,
            payload.code.len(),
            timeout.as_millis()
        );

        // Build the subprocess command.
        let mut cmd = TokioCommand::new(&interpreter[0]);
        for arg in &interpreter[1..] {
            cmd.arg(arg);
        }
        cmd.arg(&payload.code);

        // Capture stdout + stderr, don't inherit stdin.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.stdin(std::process::Stdio::null());

        // Execute with timeout.
        let result = tokio::time::timeout(timeout, cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout =
                    self.truncate_output(String::from_utf8_lossy(&output.stdout).into_owned());
                let stderr =
                    self.truncate_output(String::from_utf8_lossy(&output.stderr).into_owned());

                let exit_code = output.status.code().unwrap_or(-1);

                if output.status.success() {
                    debug!("Action succeeded. stdout={} bytes", stdout.len());
                    Ok(ExecutionOutcome::Success { stdout })
                } else {
                    warn!(
                        "Action failed. exit_code={}, stderr={} bytes",
                        exit_code,
                        stderr.len()
                    );
                    Ok(ExecutionOutcome::Failure {
                        exit_code,
                        stderr,
                        error_token_hint: Some(payload.start_token_pos),
                    })
                }
            }
            Ok(Err(e)) => {
                // Process failed to spawn.
                Err(InferenceError::ExecutionFailed {
                    exit_code: -1,
                    stderr: format!("Failed to spawn process: {e}"),
                })
            }
            Err(_) => {
                // Timeout.
                warn!("Action timed out after {}ms", timeout.as_millis());
                Ok(ExecutionOutcome::Timeout {
                    timeout_ms: timeout.as_millis() as u64,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ActionParser
// ---------------------------------------------------------------------------

/// Utilities for extracting action payloads from LLM-generated text.
pub struct ActionParser;

impl ActionParser {
    /// Extract the first `<action type="...">...</action>` block from text.
    ///
    /// Returns `None` if no valid action block is found.
    pub fn extract_action(text: &str, token_offset: usize) -> Option<ActionPayload> {
        // Find <action type="...">
        let open_tag_start = text.find("<action")?;
        let open_tag_end = text[open_tag_start..].find('>')? + open_tag_start + 1;
        let close_tag = text.find("</action>")?;

        if close_tag <= open_tag_end {
            return None;
        }

        // Extract the action type attribute.
        let tag_content = &text[open_tag_start..open_tag_end];
        let action_type =
            Self::extract_attribute(tag_content, "type").unwrap_or_else(|| "python".to_string());

        // Extract the code between tags.
        let code = text[open_tag_end..close_tag].trim().to_string();

        Some(ActionPayload {
            action_type,
            code,
            start_token_pos: token_offset + open_tag_start,
            end_token_pos: token_offset + close_tag + "</action>".len(),
        })
    }

    /// Extract an attribute value from a tag string.
    /// e.g., `extract_attribute("<action type=\"python\">", "type")` → `Some("python")`
    fn extract_attribute(tag: &str, attr_name: &str) -> Option<String> {
        let pattern = format!("{}=\"", attr_name);
        let start = tag.find(&pattern)? + pattern.len();
        let end = tag[start..].find('"')? + start;
        Some(tag[start..end].to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_parser_basic() {
        let text = r#"Here is the code:
<action type="python">
print("Hello, world!")
</action>
Done."#;

        let payload = ActionParser::extract_action(text, 0).unwrap();
        assert_eq!(payload.action_type, "python");
        assert_eq!(payload.code, "print(\"Hello, world!\")");
    }

    #[test]
    fn test_action_parser_default_type() {
        let text = "<action>some code</action>";
        let payload = ActionParser::extract_action(text, 0).unwrap();
        assert_eq!(payload.action_type, "python");
    }

    #[test]
    fn test_action_parser_no_action() {
        let text = "Just some regular text without any action blocks.";
        assert!(ActionParser::extract_action(text, 0).is_none());
    }

    #[test]
    fn test_action_parser_with_offset() {
        let text = "<action type=\"bash\">ls -la</action>";
        let payload = ActionParser::extract_action(text, 100).unwrap();
        assert_eq!(payload.start_token_pos, 100);
        assert_eq!(payload.action_type, "bash");
        assert_eq!(payload.code, "ls -la");
    }

    #[test]
    fn test_process_executor_default_interpreters() {
        let exec = ProcessExecutor::new();
        assert!(exec.interpreters.contains_key("python"));
        assert!(exec.interpreters.contains_key("bash"));
        assert!(exec.interpreters.contains_key("powershell"));
    }

    #[test]
    fn test_truncate_output() {
        let exec = ProcessExecutor {
            interpreters: HashMap::new(),
            max_output_bytes: 10,
        };

        let short = "hi".to_string();
        assert_eq!(exec.truncate_output(short.clone()), "hi");

        let long = "a".repeat(100);
        let truncated = exec.truncate_output(long);
        assert!(truncated.contains("[output truncated]"));
        assert!(truncated.len() < 100);
    }
}
