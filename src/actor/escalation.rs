//! # Escalation Pipeline — Typestate-Driven Analysis
//!
//! Implements a composable pipeline of progressively expensive analysis
//! stages for peripheral agents (Reviewer, Planner, etc.). Each stage
//! returns a [`Verdict`] that structurally determines whether to:
//!
//! - **Pass/Fail** → Short-circuit with 0 tokens consumed.
//! - **Uncertain** → Escalate to the next stage with accumulated context.
//!
//! The final stage may dispatch an isolated LLM inference request via
//! [`InferenceGateway`](super::stages::llm::InferenceGateway).
//!
//! ## Design Principles
//!
//! 1. **Zero-cost early exit:** Regex/syntax stages run in <1ms.
//! 2. **KV-Cache isolation:** LLM escalation uses a separate `seq_id`.
//! 3. **Deadlock-free:** LLM responses use `oneshot` channels, not inbox polling.
//! 4. **Interrupt-compatible:** All `.await` points are raced against
//!    `signal_rx` by `spawn_agent`'s `select!` loop.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Verdict — the universal output of every analysis stage
// ---------------------------------------------------------------------------

/// The result of any analysis stage in the escalation pipeline.
///
/// This enum drives pipeline control flow:
/// - `Pass` / `Fail` → pipeline terminates immediately.
/// - `Uncertain` → context is forwarded to the next stage.
#[derive(Debug)]
pub enum Verdict {
    /// Definitive pass — no issues found.
    ///
    /// `confidence` ∈ [0.0, 1.0] indicates how certain the stage is.
    /// A confidence of 1.0 means the stage is absolutely certain there
    /// are no issues (e.g., empty content, all checks passed).
    Pass { confidence: f32 },

    /// Definitive fail — issues found that may warrant interrupt/rollback.
    Fail { issues: Vec<ReviewIssue> },

    /// Insufficient information to decide. Escalate to next stage.
    ///
    /// The [`EscalationContext`] carries accumulated hints from prior
    /// stages so that the next stage (especially LLM) has richer context.
    Uncertain { ctx: EscalationContext },
}

// ---------------------------------------------------------------------------
// ReviewIssue
// ---------------------------------------------------------------------------

/// A single issue found during review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewIssue {
    /// How severe this issue is.
    pub severity: IssueSeverity,
    /// Classification of the issue.
    pub category: IssueCategory,
    /// Human-readable description.
    pub description: String,
    /// Byte offset in the reviewed content, if applicable.
    pub location: Option<usize>,
}

/// Severity levels for review issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Must trigger interrupt and rollback.
    Critical,
    /// Should be logged but does not trigger interrupt.
    Warning,
    /// Informational — no action needed.
    Info,
}

/// Classification of issue types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    PanicCall,
    TodoMacro,
    UnsafeBlock,
    UnbalancedDelimiters,
    SecurityAntiPattern,
    OutputTooLong,
    /// Only produced by the LLM stage.
    SemanticIssue,
    /// Catch-all for user-defined checks.
    Custom,
}

// ---------------------------------------------------------------------------
// EscalationContext
// ---------------------------------------------------------------------------

/// Context accumulated by earlier pipeline stages, carried forward
/// to more expensive stages.
///
/// When the LLM stage receives this, it constructs a prompt from
/// the content, hints, and criteria — respecting `max_tokens`.
#[derive(Debug, Clone)]
pub struct EscalationContext {
    /// The original content under review.
    pub content: String,

    /// Hints from prior stages.
    ///
    /// Example: `"Brace count {=5, }=4 — might be inside string literal"`
    pub hints: Vec<String>,

    /// Hard token budget for the LLM call.
    ///
    /// Enforced by the `InferenceGateway` before dispatch.
    /// Default: 128 (enough for a structured yes/no + reasoning).
    pub max_tokens: u16,

    /// The original review criteria requested by the caller.
    pub criteria: Vec<String>,
}

impl EscalationContext {
    /// Create a new escalation context from the reviewed content.
    pub fn new(content: String, criteria: Vec<String>) -> Self {
        Self {
            content,
            hints: Vec::new(),
            max_tokens: 128,
            criteria,
        }
    }

    /// Add a hint from a prior stage.
    pub fn add_hint(&mut self, hint: impl Into<String>) {
        self.hints.push(hint.into());
    }
}

// ---------------------------------------------------------------------------
// AnalysisStage Trait
// ---------------------------------------------------------------------------

/// A single analysis stage in the escalation pipeline.
///
/// Stages are composable and ordered by cost:
/// 0. Heuristic (regex/substring) — ~0ms, 0 tokens
/// 1. Syntax (structural checks) — ~0ms, 0 tokens
/// 2. LLM (semantic analysis) — ~100ms+, N tokens
///
/// Implementations must be `Send + Sync` to work inside Tokio tasks.
#[async_trait::async_trait]
pub trait AnalysisStage: Send + Sync {
    /// Human-readable name for tracing/diagnostics.
    fn name(&self) -> &'static str;

    /// Run this analysis stage.
    ///
    /// # Arguments
    /// * `content` — The text being reviewed.
    /// * `criteria` — What to check for (e.g., "syntax", "security").
    /// * `prior_hints` — Hints accumulated from earlier stages.
    ///
    /// # Returns
    /// A [`Verdict`] that determines pipeline control flow.
    async fn analyze(
        &self,
        content: &str,
        criteria: &[String],
        prior_hints: &[String],
    ) -> Verdict;
}

// ---------------------------------------------------------------------------
// EscalationPipeline
// ---------------------------------------------------------------------------

/// An ordered sequence of [`AnalysisStage`]s that execute progressively.
///
/// The first stage to return `Pass` or `Fail` short-circuits the pipeline.
/// `Uncertain` propagates hints to the next stage.
///
/// ```text
/// Stage0 ──Pass──▶ DONE (0 tokens)
///   │
///   Uncertain
///   ▼
/// Stage1 ──Fail──▶ DONE (0 tokens)
///   │
///   Uncertain
///   ▼
/// Stage2 (LLM) ──▶ DONE (N tokens)
/// ```
pub struct EscalationPipeline {
    stages: Vec<Box<dyn AnalysisStage>>,
}

impl EscalationPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the pipeline. Stages execute in insertion order.
    pub fn add_stage(mut self, stage: impl AnalysisStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Run the full pipeline against the given content.
    ///
    /// Returns the first definitive `Pass`/`Fail`, or a low-confidence
    /// `Pass` if all stages are uncertain (conservative — don't block
    /// generation on inconclusive analysis).
    pub async fn evaluate(&self, content: &str, criteria: &[String]) -> Verdict {
        let mut hints: Vec<String> = Vec::new();

        for stage in &self.stages {
            let verdict = stage.analyze(content, criteria, &hints).await;

            match verdict {
                Verdict::Pass { .. } | Verdict::Fail { .. } => {
                    tracing::debug!(
                        "Pipeline resolved at stage '{}' with {:?}",
                        stage.name(),
                        std::mem::discriminant(&verdict),
                    );
                    return verdict;
                }
                Verdict::Uncertain { ctx } => {
                    tracing::debug!(
                        "Stage '{}' uncertain — {} hint(s), escalating",
                        stage.name(),
                        ctx.hints.len(),
                    );
                    hints = ctx.hints;
                }
            }
        }

        // All stages uncertain — conservative pass with very low confidence.
        tracing::warn!(
            "All {} pipeline stages were uncertain — conservative pass",
            self.stages.len()
        );
        Verdict::Pass { confidence: 0.1 }
    }

    /// Number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for EscalationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A test stage that always passes.
    struct AlwaysPass;

    #[async_trait::async_trait]
    impl AnalysisStage for AlwaysPass {
        fn name(&self) -> &'static str {
            "AlwaysPass"
        }
        async fn analyze(&self, _: &str, _: &[String], _: &[String]) -> Verdict {
            Verdict::Pass { confidence: 1.0 }
        }
    }

    /// A test stage that always fails.
    struct AlwaysFail;

    #[async_trait::async_trait]
    impl AnalysisStage for AlwaysFail {
        fn name(&self) -> &'static str {
            "AlwaysFail"
        }
        async fn analyze(&self, _: &str, _: &[String], _: &[String]) -> Verdict {
            Verdict::Fail {
                issues: vec![ReviewIssue {
                    severity: IssueSeverity::Critical,
                    category: IssueCategory::Custom,
                    description: "Test failure".into(),
                    location: None,
                }],
            }
        }
    }

    /// A test stage that is always uncertain.
    struct AlwaysUncertain;

    #[async_trait::async_trait]
    impl AnalysisStage for AlwaysUncertain {
        fn name(&self) -> &'static str {
            "AlwaysUncertain"
        }
        async fn analyze(&self, content: &str, criteria: &[String], _: &[String]) -> Verdict {
            Verdict::Uncertain {
                ctx: EscalationContext::new(content.to_string(), criteria.to_vec()),
            }
        }
    }

    #[tokio::test]
    async fn test_pipeline_short_circuits_on_pass() {
        let pipeline = EscalationPipeline::new()
            .add_stage(AlwaysPass)
            .add_stage(AlwaysFail); // Should never run.

        let verdict = pipeline.evaluate("fn main() {}", &[]).await;
        assert!(matches!(verdict, Verdict::Pass { confidence } if confidence == 1.0));
    }

    #[tokio::test]
    async fn test_pipeline_short_circuits_on_fail() {
        let pipeline = EscalationPipeline::new()
            .add_stage(AlwaysFail)
            .add_stage(AlwaysPass); // Should never run.

        let verdict = pipeline.evaluate("panic!()", &[]).await;
        assert!(matches!(verdict, Verdict::Fail { .. }));
    }

    #[tokio::test]
    async fn test_pipeline_escalates_through_uncertain() {
        let pipeline = EscalationPipeline::new()
            .add_stage(AlwaysUncertain)
            .add_stage(AlwaysPass);

        let verdict = pipeline.evaluate("ambiguous code", &[]).await;
        assert!(matches!(verdict, Verdict::Pass { .. }));
    }

    #[tokio::test]
    async fn test_pipeline_all_uncertain_conservative_pass() {
        let pipeline = EscalationPipeline::new()
            .add_stage(AlwaysUncertain)
            .add_stage(AlwaysUncertain);

        let verdict = pipeline.evaluate("??", &[]).await;
        assert!(matches!(verdict, Verdict::Pass { confidence } if confidence < 0.5));
    }

    #[test]
    fn test_escalation_context_hints() {
        let mut ctx = EscalationContext::new("content".into(), vec!["syntax".into()]);
        ctx.add_hint("Brace mismatch suspected");
        assert_eq!(ctx.hints.len(), 1);
        assert_eq!(ctx.max_tokens, 128);
    }
}
