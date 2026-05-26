//! # LanceDB Memory Store
//!
//! Production-grade persistent memory store leveraging LanceDB (Rust-native)
//! for efficient vector similarity search via the Lance columnar format.
//! Replaces the brute-force `MemoryStore` when node counts cross the
//! threshold where ANN indexing becomes beneficial.
//!
//! ## Architecture
//!
//! LanceDB stores data in the Lance columnar format on local disk (or cloud
//! object stores). Each memory node is stored as a row in a single table
//! `memory_nodes` with the following Arrow schema:
//!
//! | Column      | Arrow Type                         | Description                       |
//! |-------------|------------------------------------|-----------------------------------|
//! | `uuid`      | `Utf8`                             | UUID v4 string identifier         |
//! | `text`      | `Utf8`                             | Raw text payload `x`              |
//! | `role`      | `UInt8`                            | Encoded role `ρ` (0/1/2)          |
//! | `timestamp` | `UInt64`                           | Unix epoch seconds `τ`            |
//! | `embedding` | `FixedSizeList<Float32>(dim)`      | Dense vector `e ∈ ℝ^d`           |
//!
//! Vector search uses LanceDB's built-in L2/Cosine nearest-neighbor engine,
//! with optional IVF-PQ indexing for datasets exceeding ~50k nodes.
//!
//! ## Advantages over SQLite-VSS
//!
//! - **Pure Rust** — no C library dependency chain (Faiss, SQLite extensions).
//! - **Zero-copy Arrow IPC** — minimal serialization overhead.
//! - **Columnar compression** — Lance format compresses embeddings ~4×
//!   compared to SQLite BLOB storage.
//! - **Native ANN indexing** — IVF-PQ built into the storage engine.
//! - **Async-first** — natural fit for the Tokio-based Actor subsystem.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use ndarray::Array1;
use tracing::{debug, info};
use uuid::Uuid;

use super::error::{MemoryError, MemoryResult};
use super::node::{MemoryNode, Role};
use super::store::ScoredNode;

// ---------------------------------------------------------------------------
// Table & Column Constants
// ---------------------------------------------------------------------------

const TABLE_NAME: &str = "memory_nodes";
const COL_UUID: &str = "uuid";
const COL_TEXT: &str = "text";
const COL_ROLE: &str = "role";
const COL_TIMESTAMP: &str = "timestamp";
const COL_EMBEDDING: &str = "embedding";

// ---------------------------------------------------------------------------
// LanceMemoryStore
// ---------------------------------------------------------------------------

/// Persistent memory database `M` using LanceDB for nearest-neighbor search.
///
/// This is the production-grade backend for the Chain-of-Memory subsystem,
/// designed for datasets that exceed the brute-force threshold (~10k nodes).
/// For smaller workloads, the in-memory `MemoryStore` remains the fastest
/// option.
pub struct LanceMemoryStore {
    /// Kept alive to ensure the LanceDB connection outlives the table.
    /// Prefixed with `_` because the value is never read directly —
    /// only the `table` field is used for operations.
    _db: lancedb::Connection,
    table: lancedb::Table,
    dim: usize,
    /// Keep temp dir alive for in-memory-like instances.
    _tmp: Option<tempfile::TempDir>,
}

impl LanceMemoryStore {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build the Arrow schema for the memory_nodes table.
    fn build_schema(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(COL_UUID, DataType::Utf8, false),
            Field::new(COL_TEXT, DataType::Utf8, false),
            Field::new(COL_ROLE, DataType::UInt8, false),
            Field::new(COL_TIMESTAMP, DataType::UInt64, false),
            Field::new(
                COL_EMBEDDING,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]))
    }

    /// Create or open a LanceDB database at the given filesystem path.
    ///
    /// If the table `memory_nodes` already exists it will be opened;
    /// otherwise a new empty table is created with the correct schema.
    pub async fn new<P: AsRef<Path>>(path: P, dim: usize) -> MemoryResult<Self> {
        let db = lancedb::connect(path.as_ref().to_str().unwrap_or("imece_memory"))
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("LanceDB connect failed: {}", e)))?;

        let table = Self::open_or_create_table(&db, dim).await?;

        info!("LanceDB memory store initialized. Dimension: {}", dim);

        Ok(Self {
            _db: db,
            table,
            dim,
            _tmp: None,
        })
    }

    /// Create an ephemeral LanceDB instance backed by a temporary directory.
    ///
    /// The database is automatically cleaned up when the `LanceMemoryStore`
    /// is dropped. Ideal for tests and short-lived agent sessions.
    pub async fn new_in_memory(dim: usize) -> MemoryResult<Self> {
        let tmp = tempfile::tempdir()
            .map_err(|e| MemoryError::DatabaseError(format!("Failed to create temp dir: {}", e)))?;

        let db = lancedb::connect(tmp.path().to_str().unwrap_or("imece_tmp"))
            .execute()
            .await
            .map_err(|e| {
                MemoryError::DatabaseError(format!("LanceDB temp connect failed: {}", e))
            })?;

        let table = Self::open_or_create_table(&db, dim).await?;

        info!("Ephemeral LanceDB store initialized. Dimension: {}", dim);

        Ok(Self {
            _db: db,
            table,
            dim,
            _tmp: Some(tmp),
        })
    }

    /// Open an existing table or create a new empty one.
    async fn open_or_create_table(
        db: &lancedb::Connection,
        dim: usize,
    ) -> MemoryResult<lancedb::Table> {
        let schema = Self::build_schema(dim);

        // Check if the table already exists.
        let table_names = db
            .table_names()
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Failed to list tables: {}", e)))?;

        if table_names.iter().any(|n| n == TABLE_NAME) {
            db.open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(|e| MemoryError::DatabaseError(format!("Open table failed: {}", e)))
        } else {
            // Create an empty RecordBatch with zero rows to bootstrap the table.
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

            db.create_table(
                TABLE_NAME,
                Box::new(reader) as Box<dyn RecordBatchReader + Send>,
            )
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Create table failed: {}", e)))
        }
    }

    // ── Arrow Conversion Helpers ──────────────────────────────────────────

    /// Convert a single `MemoryNode` into a one-row `RecordBatch`.
    fn node_to_batch(node: &MemoryNode, schema: &Arc<Schema>) -> MemoryResult<RecordBatch> {
        let uuid_arr = Arc::new(StringArray::from(vec![node.id.to_string()]));
        let text_arr = Arc::new(StringArray::from(vec![node.text.clone()]));
        let role_arr = Arc::new(UInt8Array::from(vec![node.role.as_u8()]));
        let ts_arr = Arc::new(UInt64Array::from(vec![node.timestamp]));

        let emb_values: Vec<f32> = node
            .embedding
            .as_slice()
            .expect("embedding not contiguous")
            .to_vec();
        let values_arr = Float32Array::from(emb_values);
        let emb_arr = Arc::new(
            FixedSizeListArray::try_new_from_values(values_arr, node.dim() as i32).map_err(
                |e| MemoryError::Serialization(format!("Embedding array build failed: {}", e)),
            )?,
        );

        RecordBatch::try_new(
            schema.clone(),
            vec![uuid_arr, text_arr, role_arr, ts_arr, emb_arr],
        )
        .map_err(|e| MemoryError::Serialization(format!("RecordBatch build failed: {}", e)))
    }

    /// Convert multiple `MemoryNode`s into a single `RecordBatch`.
    fn nodes_to_batch(
        nodes: &[MemoryNode],
        dim: usize,
        schema: &Arc<Schema>,
    ) -> MemoryResult<RecordBatch> {
        if nodes.is_empty() {
            return Ok(RecordBatch::new_empty(schema.clone()));
        }

        let uuids: Vec<String> = nodes.iter().map(|n| n.id.to_string()).collect();
        let texts: Vec<String> = nodes.iter().map(|n| n.text.clone()).collect();
        let roles: Vec<u8> = nodes.iter().map(|n| n.role.as_u8()).collect();
        let timestamps: Vec<u64> = nodes.iter().map(|n| n.timestamp).collect();

        let uuid_arr = Arc::new(StringArray::from(uuids));
        let text_arr = Arc::new(StringArray::from(texts));
        let role_arr = Arc::new(UInt8Array::from(roles));
        let ts_arr = Arc::new(UInt64Array::from(timestamps));

        // Flatten all embeddings into a single contiguous f32 buffer.
        let mut flat_emb: Vec<f32> = Vec::with_capacity(nodes.len() * dim);
        for node in nodes {
            let slice = node.embedding.as_slice().expect("embedding not contiguous");
            flat_emb.extend_from_slice(slice);
        }
        let values_arr = Float32Array::from(flat_emb);
        let emb_arr = Arc::new(
            FixedSizeListArray::try_new_from_values(values_arr, dim as i32).map_err(|e| {
                MemoryError::Serialization(format!("Batch embedding build failed: {}", e))
            })?,
        );

        RecordBatch::try_new(
            schema.clone(),
            vec![uuid_arr, text_arr, role_arr, ts_arr, emb_arr],
        )
        .map_err(|e| MemoryError::Serialization(format!("Batch build failed: {}", e)))
    }

    // ── Insertion ─────────────────────────────────────────────────────────

    /// Insert a single memory node into the LanceDB table.
    pub async fn insert(&mut self, node: &MemoryNode) -> MemoryResult<()> {
        if node.dim() != self.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dim,
                got: node.dim(),
            });
        }

        let schema = Self::build_schema(self.dim);
        let batch = Self::node_to_batch(node, &schema)?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        self.table
            .add(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Insert failed: {}", e)))?;

        debug!("Inserted memory node: {}", node.id);
        Ok(())
    }

    /// Batch-insert multiple memory nodes in a single write.
    ///
    /// Significantly faster than individual inserts due to fewer I/O round-trips
    /// and better columnar compression on larger batches.
    pub async fn insert_batch(&mut self, nodes: &[MemoryNode]) -> MemoryResult<()> {
        for node in nodes {
            if node.dim() != self.dim {
                return Err(MemoryError::DimensionMismatch {
                    expected: self.dim,
                    got: node.dim(),
                });
            }
        }

        if nodes.is_empty() {
            return Ok(());
        }

        let schema = Self::build_schema(self.dim);
        let batch = Self::nodes_to_batch(nodes, self.dim, &schema)?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        self.table
            .add(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Batch insert failed: {}", e)))?;

        debug!("Batch-inserted {} memory nodes", nodes.len());
        Ok(())
    }

    // ── Retrieval (Vector Search) ─────────────────────────────────────────

    /// Retrieve the Top-K most similar nodes to `query` via LanceDB vector search.
    ///
    /// Uses L2 distance internally; results are converted to an approximate
    /// cosine similarity score for API compatibility with the in-memory store.
    ///
    /// # Arguments
    /// * `query` — Query embedding `q ∈ ℝ^d`.
    /// * `top_k` — Maximum number of candidates to return.
    /// * `exclude_ids` — Node UUIDs to skip (already consumed by the chain).
    pub async fn top_k(
        &self,
        query: &Array1<f32>,
        top_k: usize,
        exclude_ids: &[Uuid],
    ) -> MemoryResult<Vec<ScoredNode>> {
        if query.len() != self.dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let query_vec: Vec<f32> = query
            .as_slice()
            .expect("query embedding not contiguous")
            .to_vec();

        // Request extra candidates to compensate for post-filter exclusions.
        let fetch_k = top_k + exclude_ids.len();

        let results = self
            .table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| MemoryError::DatabaseError(format!("Query build failed: {}", e)))?
            .limit(fetch_k)
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Query execute failed: {}", e)))?
            .try_collect::<Vec<RecordBatch>>()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Result collection failed: {}", e)))?;

        let mut scored: Vec<ScoredNode> = Vec::new();

        for batch in &results {
            let uuid_col = batch
                .column_by_name(COL_UUID)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing uuid column".into()))?;

            let text_col = batch
                .column_by_name(COL_TEXT)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing text column".into()))?;

            let role_col = batch
                .column_by_name(COL_ROLE)
                .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing role column".into()))?;

            let ts_col = batch
                .column_by_name(COL_TIMESTAMP)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing timestamp column".into()))?;

            let emb_col = batch
                .column_by_name(COL_EMBEDDING)
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing embedding column".into()))?;

            // LanceDB appends a `_distance` column with L2 distances.
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for row in 0..batch.num_rows() {
                let uuid_str = uuid_col.value(row);
                let id = Uuid::parse_str(uuid_str).unwrap_or_else(|_| Uuid::new_v4());

                // Exclude filter
                if exclude_ids.contains(&id) {
                    continue;
                }

                let text = text_col.value(row).to_string();
                let role = Role::from_u8(role_col.value(row)).unwrap_or(Role::User);
                let timestamp = ts_col.value(row);

                // Reconstruct embedding from FixedSizeList.
                let emb_values = emb_col.value(row);
                let float_arr = emb_values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("embedding values must be Float32");
                let embedding = Array1::from_vec(float_arr.values().to_vec());

                // Convert L2 distance → approximate cosine similarity.
                // For normalized vectors: L2² = 2 - 2·cos_sim → cos_sim = 1 - L2²/2
                let distance = dist_col.map(|d| d.value(row)).unwrap_or(0.0);
                let score = 1.0 - (distance / 2.0);

                let node = MemoryNode::from_raw(id, text, timestamp, role, embedding);

                scored.push(ScoredNode { node, score });
            }
        }

        // Truncate to requested top_k after exclusion filtering.
        scored.truncate(top_k);
        Ok(scored)
    }

    // ── Metadata ──────────────────────────────────────────────────────────

    /// Count the total number of memory nodes in the table.
    pub async fn len(&self) -> MemoryResult<usize> {
        let count = self
            .table
            .count_rows(None)
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Count failed: {}", e)))?;
        Ok(count)
    }

    /// Whether the store is empty.
    pub async fn is_empty(&self) -> MemoryResult<bool> {
        Ok(self.len().await? == 0)
    }

    /// Expected embedding dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ── Indexing ──────────────────────────────────────────────────────────

    /// Create an IVF-PQ vector index on the embedding column.
    ///
    /// Should be called once the table contains a meaningful number of rows
    /// (typically >10k) to accelerate ANN search. For small tables, LanceDB
    /// performs an efficient flat scan automatically.
    pub async fn create_index(&self) -> MemoryResult<()> {
        use lancedb::index::Index;

        self.table
            .create_index(&[COL_EMBEDDING], Index::Auto)
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Index creation failed: {}", e)))?;

        info!("Created IVF-PQ index on embedding column");
        Ok(())
    }

    // ── Deletion ──────────────────────────────────────────────────────────

    /// Delete a memory node by its UUID.
    pub async fn delete(&self, id: &Uuid) -> MemoryResult<()> {
        let filter = format!("{} = '{}'", COL_UUID, id);
        self.table
            .delete(&filter)
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Delete failed: {}", e)))?;

        debug!("Deleted memory node: {}", id);
        Ok(())
    }

    /// Delete all rows from the memory table (full reset).
    pub async fn clear(&self) -> MemoryResult<()> {
        self.table
            .delete("true")
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Clear failed: {}", e)))?;

        debug!("Cleared all memory nodes");
        Ok(())
    }

    // ── Full Scan (for DMCE fallback) ─────────────────────────────────────

    /// Retrieve ALL nodes from the table.
    ///
    /// Used by the DMCE engine when it needs the full candidate pool without
    /// vector pre-filtering. This should only be called on small stores
    /// (< threshold) as a fallback path.
    pub async fn all_nodes(&self) -> MemoryResult<Vec<MemoryNode>> {
        let results = self
            .table
            .query()
            .execute()
            .await
            .map_err(|e| MemoryError::DatabaseError(format!("Full scan failed: {}", e)))?
            .try_collect::<Vec<RecordBatch>>()
            .await
            .map_err(|e| {
                MemoryError::DatabaseError(format!("Full scan collection failed: {}", e))
            })?;

        let mut nodes: Vec<MemoryNode> = Vec::new();

        for batch in &results {
            let uuid_col = batch
                .column_by_name(COL_UUID)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing uuid column".into()))?;

            let text_col = batch
                .column_by_name(COL_TEXT)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing text column".into()))?;

            let role_col = batch
                .column_by_name(COL_ROLE)
                .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing role column".into()))?;

            let ts_col = batch
                .column_by_name(COL_TIMESTAMP)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing timestamp column".into()))?;

            let emb_col = batch
                .column_by_name(COL_EMBEDDING)
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| MemoryError::DatabaseError("Missing embedding column".into()))?;

            for row in 0..batch.num_rows() {
                let uuid_str = uuid_col.value(row);
                let id = Uuid::parse_str(uuid_str).unwrap_or_else(|_| Uuid::new_v4());
                let text = text_col.value(row).to_string();
                let role = Role::from_u8(role_col.value(row)).unwrap_or(Role::User);
                let timestamp = ts_col.value(row);

                let emb_values = emb_col.value(row);
                let float_arr = emb_values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("embedding values must be Float32");
                let embedding = Array1::from_vec(float_arr.values().to_vec());

                nodes.push(MemoryNode::from_raw(id, text, timestamp, role, embedding));
            }
        }

        Ok(nodes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::node::Role;

    fn make_node(text: &str, emb: Vec<f32>) -> MemoryNode {
        MemoryNode::new(text.into(), Role::User, Array1::from_vec(emb))
    }

    #[tokio::test]
    async fn test_lance_store_create_and_insert() {
        let dim = 4;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let node = make_node("hello", vec![0.1, 0.2, 0.3, 0.4]);
        store.insert(&node).await.unwrap();

        assert_eq!(store.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_lance_store_dimension_mismatch() {
        let mut store = LanceMemoryStore::new_in_memory(4).await.unwrap();
        let bad_node = make_node("wrong dim", vec![0.1, 0.2]); // dim=2 vs expected 4

        assert!(store.insert(&bad_node).await.is_err());
    }

    #[tokio::test]
    async fn test_lance_store_batch_insert() {
        let dim = 3;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let nodes = vec![
            make_node("alpha", vec![1.0, 0.0, 0.0]),
            make_node("beta", vec![0.0, 1.0, 0.0]),
            make_node("gamma", vec![0.0, 0.0, 1.0]),
        ];

        store.insert_batch(&nodes).await.unwrap();
        assert_eq!(store.len().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_lance_store_top_k_search() {
        let dim = 3;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let nodes = vec![
            make_node("close", vec![0.9, 0.1, 0.0]),
            make_node("far", vec![0.0, 0.0, 1.0]),
            make_node("medium", vec![0.5, 0.5, 0.0]),
        ];
        store.insert_batch(&nodes).await.unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = store.top_k(&query, 2, &[]).await.unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // The closest node should be "close" (most aligned with [1,0,0]).
        assert_eq!(results[0].node.text, "close");
    }

    #[tokio::test]
    async fn test_lance_store_top_k_excludes() {
        let dim = 2;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let n1 = make_node("alpha", vec![1.0, 0.0]);
        let n2 = make_node("beta", vec![0.9, 0.1]);
        store.insert(&n1).await.unwrap();
        store.insert(&n2).await.unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0]);
        let results = store.top_k(&query, 10, &[n1.id]).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node.text, "beta");
    }

    #[tokio::test]
    async fn test_lance_store_delete() {
        let dim = 3;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let node = make_node("to delete", vec![0.1, 0.2, 0.3]);
        let node_id = node.id;
        store.insert(&node).await.unwrap();
        assert_eq!(store.len().await.unwrap(), 1);

        store.delete(&node_id).await.unwrap();
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_lance_store_all_nodes() {
        let dim = 3;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let nodes = vec![
            make_node("one", vec![1.0, 0.0, 0.0]),
            make_node("two", vec![0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&nodes).await.unwrap();

        let all = store.all_nodes().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_lance_store_clear() {
        let dim = 3;
        let mut store = LanceMemoryStore::new_in_memory(dim).await.unwrap();

        let nodes = vec![
            make_node("a", vec![1.0, 0.0, 0.0]),
            make_node("b", vec![0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&nodes).await.unwrap();
        assert_eq!(store.len().await.unwrap(), 2);

        store.clear().await.unwrap();
        assert_eq!(store.len().await.unwrap(), 0);
    }
}
