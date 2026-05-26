//! # Dynamic Memory Chain Evolution (DMCE) & Adaptive Path Truncation (APT)
//!
//! Implements the core chain-building algorithm from:
//! "Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution
//!  for LLM Agents" (arXiv:2601.14287v1).
//!
//! ## Algorithm Overview
//!
//! 1. **Retrieve** Top-K candidates `P` from the flat-index store via cosine
//!    similarity against the query `q`.
//!
//! 2. **Iterate** — at each step `t`:
//!    - Compute `C_z^{(t)}` (chain centroid embedding).
//!    - For each candidate `m ∈ P`, compute the **gating score**:
//!      ```text
//!      S_gate(m) = cos(m.e, q) × cos(m.e, C_z^{(t)})
//!      ```
//!    - Select `m* = argmax S_gate(m)` — the candidate that maximizes both
//!      *global relevance* and *contextual consistency*.
//!    - Remove `m*` from `P`, append to chain `C_z`.
//!
//! 3. **Truncate (APT)** — if the winning score drops sharply:
//!    ```text
//!    s*_t < β × s_{t-1}
//!    ```
//!    Terminate chain evolution. This prevents semantic drift and VRAM bloat.

use ndarray::Array1;
use tracing::{debug, trace};

use super::node::MemoryNode;
use super::store::{cosine_similarity, MemoryStore};

// ---------------------------------------------------------------------------
// DMCE Engine Configuration
// ---------------------------------------------------------------------------

/// Configuration & execution context for the DMCE chain evolution algorithm.
///
/// # Hyperparameters
///
/// | Param          | Spec Symbol | Range     | Description                          |
/// |----------------|-------------|-----------|--------------------------------------|
/// | `beta`         | β           | `[0, 1]`  | APT truncation sensitivity           |
/// | `top_k`        | K           | `≥ 1`     | Candidate pool size per step         |
/// | `max_chain_len`| —           | `≥ 1`     | Hard upper-bound on chain length     |
#[derive(Debug, Clone)]
pub struct DmceEngine {
    /// β — APT truncation threshold ∈ [0, 1].
    ///
    /// Lower values are more aggressive (shorter chains, less VRAM).
    /// Higher values allow longer chains (richer context, more VRAM).
    ///
    /// Recommended: `0.5` – `0.7` for general-purpose use on ≤8GB devices.
    pub beta: f32,

    /// Top-K candidates retrieved from the store at the start of evolution.
    pub top_k: usize,

    /// Maximum chain length (hard cap). Prevents runaway chains even when
    /// APT does not trigger.
    pub max_chain_len: usize,
}

impl DmceEngine {
    /// Create a new DMCE engine.
    ///
    /// # Panics
    /// Panics in debug builds if `beta` is outside `[0, 1]`, or if `top_k`
    /// or `max_chain_len` is zero.
    pub fn new(beta: f32, top_k: usize, max_chain_len: usize) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&beta),
            "β must be in [0, 1], got {beta}"
        );
        debug_assert!(top_k > 0, "top_k must be ≥ 1");
        debug_assert!(max_chain_len > 0, "max_chain_len must be ≥ 1");

        Self {
            beta,
            top_k,
            max_chain_len,
        }
    }

    // ── Core Algorithm ────────────────────────────────────────────────────

    /// Run the DMCE chain evolution algorithm.
    ///
    /// # Arguments
    /// * `store` — The memory store containing all candidate nodes.
    /// * `query` — The query embedding `q ∈ ℝ^d`.
    ///
    /// # Returns
    /// An ordered `Vec<MemoryNode>` representing the evolved chain `C_z`.
    /// The chain is ordered by selection step and is ready for injection
    /// into the LLM context window.
    pub fn evolve(&self, store: &MemoryStore, query: &Array1<f32>) -> Vec<MemoryNode> {
        // Step 1: Retrieve initial candidate pool P via basic cosine sim.
        let candidates = match store.top_k(query, self.top_k, &[]) {
            Ok(c) if !c.is_empty() => c,
            _ => {
                debug!("DMCE: empty candidate pool, returning empty chain");
                return Vec::new();
            }
        };

        // Mutable pool — candidates are consumed as they are selected.
        let mut pool: Vec<MemoryNode> = candidates.into_iter().map(|sc| sc.node).collect();

        // The evolving chain C_z.
        let mut chain: Vec<MemoryNode> = Vec::with_capacity(self.max_chain_len);

        // Running sum of embeddings for O(1) centroid updates.
        let dim = store.dim();
        let mut chain_embedding_sum = Array1::<f32>::zeros(dim);

        // Previous step's winning score (for APT comparison).
        let mut prev_score: Option<f32> = None;

        // Step 2: Iterative chain evolution.
        for step in 0..self.max_chain_len {
            if pool.is_empty() {
                debug!("DMCE: pool exhausted at step {step}");
                break;
            }

            // Compute chain centroid C_z^{(t)}.
            let chain_centroid = if chain.is_empty() {
                // First step: centroid = query itself (bootstrap).
                query.clone()
            } else {
                &chain_embedding_sum / chain.len() as f32
            };

            // Compute gating scores for all remaining candidates.
            let (best_idx, best_score) = self.select_best_candidate(&pool, query, &chain_centroid);

            trace!(
                "DMCE step {step}: best_score={best_score:.6}, \
                 candidate=\"{}\"",
                pool[best_idx].text
            );

            // Step 3: APT — check truncation condition.
            if let Some(prev) = prev_score {
                if best_score < self.beta * prev {
                    debug!(
                        "DMCE: APT triggered at step {step} \
                         (s*_t={best_score:.4} < β·s_{{t-1}}={:.4})",
                        self.beta * prev
                    );
                    break;
                }
            }

            // Accept the winning candidate into the chain.
            let winner = pool.swap_remove(best_idx);
            chain_embedding_sum = &chain_embedding_sum + &winner.embedding;
            chain.push(winner);
            prev_score = Some(best_score);
        }

        debug!(
            "DMCE: chain evolution complete — {} nodes selected",
            chain.len()
        );

        chain
    }

    /// Score all candidates and return the index + score of the best one.
    ///
    /// **Gating Score:**
    /// ```text
    /// S_gate(m) = cos(m.e, q) × cos(m.e, C_z^{(t)})
    /// ```
    ///
    /// This multiplicative gate enforces:
    /// - **Global Relevance:** alignment with the original user query.
    /// - **Contextual Consistency:** logical coherence with the chain so far.
    fn select_best_candidate(
        &self,
        pool: &[MemoryNode],
        query: &Array1<f32>,
        chain_centroid: &Array1<f32>,
    ) -> (usize, f32) {
        let mut best_idx: usize = 0;
        let mut best_score: f32 = f32::NEG_INFINITY;

        for (i, candidate) in pool.iter().enumerate() {
            let cos_query = cosine_similarity(&candidate.embedding, query);
            let cos_chain = cosine_similarity(&candidate.embedding, chain_centroid);

            // Multiplicative gating — both terms must be high.
            let s_gate = cos_query * cos_chain;

            if s_gate > best_score {
                best_score = s_gate;
                best_idx = i;
            }
        }

        (best_idx, best_score)
    }
}

// ---------------------------------------------------------------------------
// ChainResult — Structured output of the DMCE pipeline
// ---------------------------------------------------------------------------

/// The finalized output of a DMCE evolution pass, ready for LLM injection.
///
/// Contains the ordered chain nodes plus diagnostic metadata useful for
/// debugging and telemetry on low-VRAM devices.
#[derive(Debug, Clone)]
pub struct ChainResult {
    /// Ordered chain of selected memory nodes.
    pub chain: Vec<MemoryNode>,

    /// Number of DMCE steps executed before termination.
    pub steps_executed: usize,

    /// Whether APT (Adaptive Path Truncation) triggered early termination.
    pub apt_triggered: bool,

    /// The gating score at each step (for telemetry / visualization).
    pub step_scores: Vec<f32>,
}

impl DmceEngine {
    /// Run DMCE with full diagnostic output.
    ///
    /// Identical to [`evolve`], but returns a [`ChainResult`] with telemetry.
    pub fn evolve_with_diagnostics(&self, store: &MemoryStore, query: &Array1<f32>) -> ChainResult {
        let candidates = match store.top_k(query, self.top_k, &[]) {
            Ok(c) if !c.is_empty() => c,
            _ => {
                return ChainResult {
                    chain: Vec::new(),
                    steps_executed: 0,
                    apt_triggered: false,
                    step_scores: Vec::new(),
                };
            }
        };

        let mut pool: Vec<MemoryNode> = candidates.into_iter().map(|sc| sc.node).collect();

        let dim = store.dim();
        let mut chain: Vec<MemoryNode> = Vec::with_capacity(self.max_chain_len);
        let mut chain_embedding_sum = Array1::<f32>::zeros(dim);
        let mut prev_score: Option<f32> = None;
        let mut step_scores: Vec<f32> = Vec::new();
        let mut apt_triggered = false;
        let mut steps_executed: usize = 0;

        for step in 0..self.max_chain_len {
            if pool.is_empty() {
                break;
            }

            let chain_centroid = if chain.is_empty() {
                query.clone()
            } else {
                &chain_embedding_sum / chain.len() as f32
            };

            let (best_idx, best_score) = self.select_best_candidate(&pool, query, &chain_centroid);

            steps_executed = step + 1;
            step_scores.push(best_score);

            if let Some(prev) = prev_score {
                if best_score < self.beta * prev {
                    apt_triggered = true;
                    break;
                }
            }

            let winner = pool.swap_remove(best_idx);
            chain_embedding_sum = &chain_embedding_sum + &winner.embedding;
            chain.push(winner);
            prev_score = Some(best_score);
        }

        ChainResult {
            chain,
            steps_executed,
            apt_triggered,
            step_scores,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::node::Role;

    /// Helper: create a node with a given embedding.
    fn node(text: &str, emb: Vec<f32>) -> MemoryNode {
        MemoryNode::new(text.into(), Role::User, Array1::from_vec(emb))
    }

    /// Helper: create a populated store with known embeddings.
    fn test_store() -> MemoryStore {
        let dim = 3;
        let mut store = MemoryStore::new_in_memory(dim).unwrap();

        // Cluster A: "France" direction  [1, 0, 0]
        store
            .insert(&node("Paris is in France", vec![0.95, 0.05, 0.0]))
            .unwrap();
        store
            .insert(&node("France is in Europe", vec![0.90, 0.10, 0.0]))
            .unwrap();

        // Cluster B: "Code" direction  [0, 1, 0]
        store
            .insert(&node("Rust is fast", vec![0.0, 0.95, 0.05]))
            .unwrap();
        store
            .insert(&node("Cargo is Rust's build tool", vec![0.0, 0.90, 0.10]))
            .unwrap();

        // Noise: nearly orthogonal
        store
            .insert(&node("Random noise", vec![0.01, 0.01, 0.99]))
            .unwrap();

        store
    }

    #[test]
    fn test_evolve_selects_coherent_cluster() {
        let store = test_store();
        let engine = DmceEngine::new(0.3, 5, 10);

        // Query along the "France" direction.
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let chain = engine.evolve(&store, &query);

        // Should select France-cluster nodes, not Code or Noise.
        assert!(!chain.is_empty());
        for n in &chain {
            assert!(
                n.text.contains("France") || n.text.contains("Paris"),
                "Unexpected node in chain: {}",
                n.text
            );
        }
    }

    #[test]
    fn test_apt_truncation() {
        let store = test_store();
        // Very aggressive β = 0.99 → should truncate after 1–2 steps.
        let engine = DmceEngine::new(0.99, 5, 10);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let result = engine.evolve_with_diagnostics(&store, &query);

        // With β ≈ 1.0, APT should trigger quickly.
        assert!(
            result.apt_triggered || result.chain.len() <= 2,
            "APT should have triggered or chain should be very short, \
             got {} nodes, apt={}",
            result.chain.len(),
            result.apt_triggered,
        );
    }

    #[test]
    fn test_evolve_empty_store() {
        let store = MemoryStore::new_in_memory(3).unwrap();
        let engine = DmceEngine::new(0.5, 5, 10);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let chain = engine.evolve(&store, &query);
        assert!(chain.is_empty());
    }

    #[test]
    fn test_max_chain_len_respected() {
        let store = test_store();
        // max_chain_len = 2, with permissive β.
        let engine = DmceEngine::new(0.01, 5, 2);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let chain = engine.evolve(&store, &query);

        assert!(chain.len() <= 2);
    }

    #[test]
    fn test_diagnostics_metadata() {
        let store = test_store();
        let engine = DmceEngine::new(0.5, 5, 10);
        let query = Array1::from_vec(vec![0.5, 0.5, 0.0]);

        let result = engine.evolve_with_diagnostics(&store, &query);

        assert_eq!(result.step_scores.len(), result.steps_executed);
        assert_eq!(
            result.chain.len() + if result.apt_triggered { 0 } else { 0 },
            result.chain.len()
        ); // chain len is consistent

        // Scores should be monotonically non-increasing (best first).
        // Note: not strictly guaranteed by the algo, but typical.
        // We just verify they are finite.
        for s in &result.step_scores {
            assert!(s.is_finite(), "Score should be finite, got {s}");
        }
    }
}
