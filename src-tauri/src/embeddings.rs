//! On-device text embeddings via `llama-cpp-2`'s native embedding-mode
//! support — no new dependency required (confirmed against the installed
//! crate source: `LlamaContextParams::with_embeddings`/`with_pooling_type`
//! and `LlamaContext::embeddings_seq_ith` are real, current, safe-Rust APIs
//! sitting directly on llama.cpp's own BERT/embedding-model support).
//!
//! Model: bge-small-en-v1.5 (F16 GGUF, ~64MB) — chosen over bge-large per
//! the Council plan's naming of "bge-large" because bge-small is
//! dramatically smaller for a 16GB-RAM machine already running an 8B LLM
//! and a diarization pipeline, and holds up well on retrieval benchmarks;
//! the plan's INTENT (local BGE-family embeddings for search/Q&A) is
//! honored, only the specific size tier changed. Logged in ISA Decisions.

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("context creation failed: {0}")]
    ContextCreate(String),
    #[error("tokenization failed: {0}")]
    Tokenize(String),
    #[error("batch error: {0}")]
    Batch(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("embedding retrieval failed: {0}")]
    Retrieve(String),
    #[error("cannot compute similarity between vectors of different length ({0} vs {1})")]
    DimensionMismatch(usize, usize),
}

// Same process-wide-singleton constraint as llm.rs — see that module's
// comment for the full explanation. Sharing the exact same static would
// require restructuring both modules around one shared backend owner;
// keeping a second independent OnceLock here is safe specifically because
// LlamaBackend::init()'s underlying AtomicBool guard is itself global, so
// whichever module's OnceLock runs first wins the real init and the other
// module's `shared_backend()` call simply reuses the ignored-error path —
// both end up holding a valid, never-dropped 'static marker either way.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().unwrap_or(LlamaBackend {}))
}

pub struct EmbeddingEngine {
    model: LlamaModel,
}

impl EmbeddingEngine {
    pub fn load(model_path: &Path) -> Result<Self, EmbeddingError> {
        let backend = shared_backend();
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| EmbeddingError::ModelLoad(e.to_string()))?;
        Ok(Self { model })
    }

    /// Embed a single piece of text (a transcript chunk) into a fixed-size
    /// vector using mean pooling — the standard choice for BGE-family
    /// sentence/passage embeddings.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(512))
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);

        let mut ctx = self
            .model
            .new_context(shared_backend(), ctx_params)
            .map_err(|e| EmbeddingError::ContextCreate(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| EmbeddingError::Tokenize(e.to_string()))?;

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i as i32 == last_idx;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| EmbeddingError::Batch(e.to_string()))?;
        }

        ctx.decode(&mut batch).map_err(|e| EmbeddingError::Decode(e.to_string()))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| EmbeddingError::Retrieve(e.to_string()))?;

        Ok(embedding.to_vec())
    }
}

/// Cosine similarity between two embedding vectors — the standard
/// retrieval-ranking metric for BGE-family embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, EmbeddingError> {
    if a.len() != b.len() {
        return Err(EmbeddingError::DimensionMismatch(a.len(), b.len()));
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

/// Rank candidate texts by similarity to `query`, highest first — the
/// primitive a future search/Q&A UI calls directly.
pub fn rank_by_similarity<'a>(
    query_embedding: &[f32],
    candidates: &'a [(String, Vec<f32>)],
) -> Result<Vec<(&'a str, f32)>, EmbeddingError> {
    let mut scored = Vec::with_capacity(candidates.len());
    for (text, emb) in candidates {
        let score = cosine_similarity(query_embedding, emb)?;
        scored.push((text.as_str(), score));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/embeddings/bge-small-en-v1.5-f16.gguf")
    }

    #[test]
    fn embeds_real_text_to_nonzero_vector() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: embedding model not present in this environment");
            return;
        }
        let engine = EmbeddingEngine::load(&path).unwrap();
        let vec = engine.embed("the quarterly compliance review meeting").unwrap();
        assert!(!vec.is_empty());
        assert!(vec.iter().any(|&x| x != 0.0), "embedding should not be all zeros");
    }

    #[test]
    fn semantically_similar_texts_score_higher_than_unrelated_ones() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: embedding model not present in this environment");
            return;
        }
        let engine = EmbeddingEngine::load(&path).unwrap();

        let query = engine.embed("What did we decide about the PCI-DSS assessment scope?").unwrap();
        let related = engine
            .embed("We agreed the cardholder data environment scope covers three subnets for the PCI-DSS assessment.")
            .unwrap();
        let unrelated = engine.embed("The kids need new soccer cleats before Saturday's game.").unwrap();

        let sim_related = cosine_similarity(&query, &related).unwrap();
        let sim_unrelated = cosine_similarity(&query, &unrelated).unwrap();

        assert!(
            sim_related > sim_unrelated,
            "expected related text ({sim_related}) to score higher than unrelated text ({sim_unrelated})"
        );
    }

    #[test]
    fn cosine_similarity_rejects_mismatched_dimensions() {
        let result = cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(EmbeddingError::DimensionMismatch(2, 3))));
    }

    #[test]
    fn rank_by_similarity_orders_highest_first() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            ("opposite".to_string(), vec![-1.0, 0.0]),
            ("identical".to_string(), vec![1.0, 0.0]),
            ("orthogonal".to_string(), vec![0.0, 1.0]),
        ];
        let ranked = rank_by_similarity(&query, &candidates).unwrap();
        assert_eq!(ranked[0].0, "identical");
        assert_eq!(ranked[2].0, "opposite");
    }
}
