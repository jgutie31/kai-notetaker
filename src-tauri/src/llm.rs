//! On-device LLM inference via `llama-cpp-2` (llama.cpp Rust bindings),
//! Metal-accelerated on macOS. Model: Llama-3.1-8B-Instruct, Q4_K_M
//! quantization (~4.9GB) — chosen over the Council plan's first-choice
//! Qwen2.5-14B-Instruct because this machine has 16GB unified RAM and a
//! 14B model (even quantized) needs ~9-10GB just to load; the plan itself
//! named Llama-3.1-8B as the explicit fallback "if VRAM-constrained,"
//! which this machine is (see ISA.md load-bearing fact, 2026-08-05).
//!
//! This module provides raw text completion only — the chunked map-reduce
//! summarization, structured action-item JSON extraction, and embeddings
//! logic (ISC-63 through ISC-67) are a separate future layer built on top
//! of this primitive, not implemented in this pass.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
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
    #[error("prompt exceeds context window: {tokens} tokens, context is {n_ctx}")]
    PromptTooLong { tokens: usize, n_ctx: u32 },
}

// llama.cpp's backend is a strict process-wide singleton: `LlamaBackend`'s
// `Drop` impl flips a single global "initialized" flag from true to false
// and panics (`unreachable!()`) if that flag was already false — meaning
// AT MOST ONE `LlamaBackend` value may ever exist and be dropped over the
// life of the process. Constructing a second one (even as a zero-field
// marker struct) and letting it drop independently corrupts that
// invariant. The correct pattern, confirmed against the crate's own
// Drop logic, is a single `'static` instance that is never dropped —
// storing it in a `OnceLock` inside a `static` achieves exactly that
// (its destructor never runs for program-duration statics). Every
// `LlmEngine` borrows this same instance rather than owning one.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("llama backend init failed"))
}

pub struct LlmEngine {
    model: LlamaModel,
}

impl LlmEngine {
    /// Load the model. `n_gpu_layers` controls how many transformer layers
    /// are offloaded to Metal — pass a large number (e.g. 1000) to offload
    /// everything llama.cpp will allow, or 0 for CPU-only.
    pub fn load(model_path: &Path, n_gpu_layers: u32) -> Result<Self, LlmError> {
        let backend = shared_backend();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| LlmError::ModelLoad(e.to_string()))?;

        Ok(Self { model })
    }

    /// Run a single completion for `prompt`, generating up to `max_tokens`
    /// new tokens (stopping early on an end-of-generation token). Greedy
    /// sampling (deterministic) — appropriate for structured extraction
    /// tasks (summaries, action items) where reproducibility matters more
    /// than creative variance.
    pub fn complete(&self, prompt: &str, max_tokens: usize, n_ctx: u32) -> Result<String, LlmError> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
        let mut ctx = self
            .model
            .new_context(shared_backend(), ctx_params)
            .map_err(|e| LlmError::ContextCreate(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| LlmError::Tokenize(e.to_string()))?;

        if tokens.len() >= n_ctx as usize {
            return Err(LlmError::PromptTooLong { tokens: tokens.len(), n_ctx });
        }

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i as i32 == last_idx;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| LlmError::Batch(e.to_string()))?;
        }
        ctx.decode(&mut batch).map_err(|e| LlmError::Decode(e.to_string()))?;

        let mut sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(new_token, &mut decoder, true, None)
                .map_err(|e| LlmError::Decode(e.to_string()))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(new_token, n_cur, &[0], true)
                .map_err(|e| LlmError::Batch(e.to_string()))?;
            ctx.decode(&mut batch).map_err(|e| LlmError::Decode(e.to_string()))?;
            n_cur += 1;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    }

    #[test]
    fn loads_real_model_and_completes_a_simple_prompt() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: LLM model not present in this environment ({path:?})");
            return;
        }

        let engine = LlmEngine::load(&path, 1000).unwrap();
        let prompt = "<|start_header_id|>user<|end_header_id|>\n\nReply with exactly the word: hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let output = engine.complete(prompt, 20, 512).unwrap();

        assert!(!output.trim().is_empty(), "expected non-empty completion");
        // Loose check — a real 8B model won't be perfectly literal, but
        // should produce something recognizable given a direct instruction.
        assert!(
            output.to_lowercase().contains("hello"),
            "expected 'hello' somewhere in output, got: {output:?}"
        );
    }

    #[test]
    fn rejects_prompt_longer_than_context_window() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: LLM model not present in this environment");
            return;
        }
        let engine = LlmEngine::load(&path, 1000).unwrap();
        let huge_prompt = "word ".repeat(2000); // far more tokens than a tiny n_ctx
        let result = engine.complete(&huge_prompt, 10, 64);
        assert!(matches!(result, Err(LlmError::PromptTooLong { .. })));
    }
}
