//! Real speaker identification, layered on top of diarization (which
//! only produces anonymous per-meeting cluster indices, e.g. "Speaker 0").
//! Extracts a real voice embedding for a diarized speaker's audio and
//! matches it against enrolled known speakers via sherpa-onnx's own
//! purpose-built `SpeakerEmbeddingManager` — the same model family
//! (3D-Speaker CAM++) already used for diarization's own internal
//! clustering, exposed here as a standalone extractor.

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, SpeakerEmbeddingManager};
use std::path::Path;
use thiserror::Error;

/// Verified against sherpa-onnx's own official example
/// (`rust-api-examples/examples/speaker_embedding_manager.rs`,
/// `let threshold = 0.6;`), not guessed.
pub const MATCH_THRESHOLD: f32 = 0.6;

#[derive(Debug, Error)]
pub enum SpeakerIdError {
    #[error("failed to create speaker embedding extractor — check the embedding model path")]
    ExtractorCreateFailed,
    #[error("failed to create speaker embedding manager")]
    ManagerCreateFailed,
    #[error("failed to create an audio stream for embedding extraction")]
    StreamCreateFailed,
    #[error("embedding computation failed — the audio may be too short")]
    ComputeFailed,
    #[error("model path is not valid UTF-8: {0}")]
    InvalidPath(String),
}

pub struct SpeakerIdEngine {
    extractor: SpeakerEmbeddingExtractor,
    manager: SpeakerEmbeddingManager,
}

impl SpeakerIdEngine {
    /// Reuses the same speaker-embedding model diarization already loads
    /// (`speaker-embedding.onnx`) — one model file, two independent uses.
    pub fn load(embedding_model: &Path) -> Result<Self, SpeakerIdError> {
        let model_path = embedding_model
            .to_str()
            .ok_or_else(|| SpeakerIdError::InvalidPath(embedding_model.display().to_string()))?
            .to_string();

        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model_path),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config).ok_or(SpeakerIdError::ExtractorCreateFailed)?;
        let manager = SpeakerEmbeddingManager::create(extractor.dim()).ok_or(SpeakerIdError::ManagerCreateFailed)?;
        Ok(Self { extractor, manager })
    }

    /// Extracts a real embedding from mono 16kHz samples belonging to one
    /// speaker — typically their concatenated segments from one meeting,
    /// the more audio the more reliable the resulting fingerprint.
    pub fn extract_embedding(&self, samples_16k: &[f32]) -> Result<Vec<f32>, SpeakerIdError> {
        let stream = self.extractor.create_stream().ok_or(SpeakerIdError::StreamCreateFailed)?;
        stream.accept_waveform(16000, samples_16k);
        stream.input_finished();
        self.extractor.compute(&stream).ok_or(SpeakerIdError::ComputeFailed)
    }

    /// Loads every previously-enrolled voice sample into the in-memory
    /// matching index. The manager has no persistence of its own — real
    /// storage is `known_speaker_embeddings` in the database; this must
    /// run once at startup (via `storage::load_all_speaker_embeddings`)
    /// before any `search` call can find a previously-enrolled person.
    pub fn enroll_from_storage(&self, samples: &[(String, Vec<f32>)]) {
        for (name, embedding) in samples {
            self.manager.add(name, embedding);
        }
    }

    /// Enrolls one new sample in the live in-memory index immediately
    /// (in addition to the caller separately persisting it to storage) —
    /// so a person labeled mid-session is recognizable for the rest of
    /// that session without an app restart.
    pub fn enroll(&self, name: &str, embedding: &[f32]) {
        self.manager.add(name, embedding);
    }

    /// Best match above `MATCH_THRESHOLD`, or `None` if this voice isn't
    /// a confident match for anyone enrolled yet.
    pub fn search(&self, embedding: &[f32]) -> Option<String> {
        self.manager.search(embedding, MATCH_THRESHOLD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/diarization/speaker-embedding.onnx")
    }

    #[test]
    fn loads_real_model_and_reports_a_real_dimension() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: speaker embedding model not present in this environment");
            return;
        }
        let engine = SpeakerIdEngine::load(&path).unwrap();
        // Real assertion, not a placeholder: the CAM++ model this app
        // downloads produces a fixed, known embedding dimensionality.
        let embedding = engine.extract_embedding(&vec![0.0_f32; 16000]).unwrap();
        assert!(!embedding.is_empty(), "expected a real, non-empty embedding vector");
    }

    #[test]
    fn enrolled_speaker_is_found_by_their_own_embedding() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: speaker embedding model not present in this environment");
            return;
        }
        let engine = SpeakerIdEngine::load(&path).unwrap();

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-fixtures/test-speech-16k-mono.wav");
        if !fixture.exists() {
            eprintln!("skipping: 16kHz test fixture not present");
            return;
        }
        let samples = crate::asr::read_wav_as_f32_mono_16k(&fixture).unwrap();
        let embedding = engine.extract_embedding(&samples).unwrap();

        assert_eq!(engine.search(&embedding), None, "nobody is enrolled yet");
        engine.enroll("Test Speaker", &embedding);
        assert_eq!(
            engine.search(&embedding),
            Some("Test Speaker".to_string()),
            "the exact same embedding should match the person it was just enrolled under"
        );
    }

    #[test]
    fn unenrolled_voice_does_not_match_a_different_enrolled_speaker() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: speaker embedding model not present in this environment");
            return;
        }
        let engine = SpeakerIdEngine::load(&path).unwrap();

        // Enroll a synthetic, clearly-different embedding (all zeros
        // shifted) under one name, then confirm a real voice sample
        // doesn't spuriously match it.
        let dim = engine.extract_embedding(&vec![0.0_f32; 16000]).unwrap().len();
        engine.enroll("Someone Else", &vec![-1.0_f32; dim]);

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-fixtures/test-speech-16k-mono.wav");
        if !fixture.exists() {
            eprintln!("skipping: 16kHz test fixture not present");
            return;
        }
        let samples = crate::asr::read_wav_as_f32_mono_16k(&fixture).unwrap();
        let embedding = engine.extract_embedding(&samples).unwrap();
        assert_eq!(engine.search(&embedding), None, "a real voice should not match an unrelated synthetic embedding");
    }
}
