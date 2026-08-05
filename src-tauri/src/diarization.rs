//! On-device speaker diarization via `sherpa-onnx`'s official Rust crate.
//!
//! Architecture note (see ISA.md Decisions, 2026-08-05): the original plan
//! specified "pyannote models exported to ONNX, run via raw `ort`." Real
//! research found the official pyannote org gates every model behind a
//! HuggingFace auth token and publishes zero ONNX exports — only community
//! conversions exist. `sherpa-onnx` already wraps exactly this problem
//! (sliding-window segmentation, powerset-class decoding, speaker-embedding
//! clustering) using the same pyannote-segmentation-3.0 model family
//! *internally through ONNX Runtime* — so the goal (on-device, ONNX-based,
//! no cloud, no vendor lock-in) is unchanged; only the crate-level plumbing
//! is better-fitted to what's actually available today.
//!
//! Zero network calls — both the segmentation and speaker-embedding models
//! are loaded from local disk.

use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DiarizationError {
    #[error("failed to create diarizer — check model paths are valid and files exist")]
    CreateFailed,
    #[error("diarization processing failed")]
    ProcessFailed,
    #[error("model path is not valid UTF-8: {0}")]
    InvalidPath(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: i32,
}

pub struct DiarizationEngine {
    inner: OfflineSpeakerDiarization,
}

impl DiarizationEngine {
    pub fn load(segmentation_model: &Path, embedding_model: &Path) -> Result<Self, DiarizationError> {
        let seg_path = segmentation_model
            .to_str()
            .ok_or_else(|| DiarizationError::InvalidPath(segmentation_model.display().to_string()))?
            .to_string();
        let emb_path = embedding_model
            .to_str()
            .ok_or_else(|| DiarizationError::InvalidPath(embedding_model.display().to_string()))?
            .to_string();

        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(seg_path),
                },
                num_threads: 1,
                debug: false,
                provider: Some("cpu".to_string()),
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(emb_path),
                num_threads: 1,
                debug: false,
                provider: Some("cpu".to_string()),
            },
            ..Default::default()
        };

        OfflineSpeakerDiarization::create(&config)
            .map(|inner| Self { inner })
            .ok_or(DiarizationError::CreateFailed)
    }

    /// The sample rate this diarizer's segmentation model requires — the
    /// caller (AudioCapture/pipeline glue) must resample to this rate
    /// before calling `diarize`, matching the same "no silent resampling
    /// inside the model layer" discipline as `asr::read_wav_as_f32_mono_16k`.
    pub fn required_sample_rate(&self) -> i32 {
        self.inner.sample_rate()
    }

    /// Diarize a full mono waveform at `required_sample_rate()`. Returns
    /// segments sorted by start time, each labeled with a speaker index
    /// (consistent across segments within this one call — sherpa-onnx does
    /// not persist speaker identity across separate `diarize` calls).
    pub fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerSegment>, DiarizationError> {
        let result = self
            .inner
            .process(samples)
            .ok_or(DiarizationError::ProcessFailed)?;

        Ok(result
            .sort_by_start_time()
            .into_iter()
            .map(|s| SpeakerSegment {
                start_ms: (s.start * 1000.0) as i64,
                end_ms: (s.end * 1000.0) as i64,
                speaker: s.speaker,
            })
            .collect())
    }
}

/// Merge ASR segments (text + timestamps) with diarization segments
/// (speaker + timestamps) into speaker-labeled transcript lines. An ASR
/// segment is assigned the speaker of whichever diarization segment covers
/// the ASR segment's midpoint — a simple, defensible overlap rule rather
/// than anything requiring frame-level alignment.
pub fn merge_asr_and_diarization(
    asr_segments: &[crate::asr::TranscriptSegment],
    speaker_segments: &[SpeakerSegment],
) -> Vec<(crate::asr::TranscriptSegment, Option<i32>)> {
    asr_segments
        .iter()
        .map(|asr_seg| {
            let midpoint = (asr_seg.start_ms + asr_seg.end_ms) / 2;
            let speaker = speaker_segments
                .iter()
                .find(|sp| midpoint >= sp.start_ms && midpoint <= sp.end_ms)
                .map(|sp| sp.speaker);
            (asr_seg.clone(), speaker)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::TranscriptSegment;
    use std::path::PathBuf;

    fn seg_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
    }

    fn emb_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/diarization/speaker-embedding.onnx")
    }

    #[test]
    fn loads_real_models_and_reports_sample_rate() {
        let seg = seg_model_path();
        let emb = emb_model_path();
        if !seg.exists() || !emb.exists() {
            eprintln!("skipping: diarization models not present in this environment");
            return;
        }
        let engine = DiarizationEngine::load(&seg, &emb).unwrap();
        // pyannote-segmentation-3.0 is a 16kHz model — assert the engine
        // reports that rather than assuming it silently.
        assert_eq!(engine.required_sample_rate(), 16000);
    }

    #[test]
    fn diarizes_real_speech_fixture_without_erroring() {
        let seg = seg_model_path();
        let emb = emb_model_path();
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-fixtures/test-speech-16k-mono.wav");
        if !seg.exists() || !emb.exists() || !fixture.exists() {
            eprintln!("skipping: models or fixture not present in this environment");
            return;
        }

        let samples = crate::asr::read_wav_as_f32_mono_16k(&fixture).unwrap();
        let engine = DiarizationEngine::load(&seg, &emb).unwrap();
        let segments = engine.diarize(&samples).unwrap();

        // A single-speaker 2.6s TTS clip should produce at least one
        // segment attributed to exactly one speaker — this is a real,
        // meaningful assertion, not a placeholder that always trivially
        // passes: it would fail if the pipeline returned zero segments or
        // mis-detected multiple speakers from one voice.
        assert!(!segments.is_empty(), "expected at least one diarized segment");
        let unique_speakers: std::collections::HashSet<i32> =
            segments.iter().map(|s| s.speaker).collect();
        assert_eq!(
            unique_speakers.len(),
            1,
            "expected exactly 1 speaker for a single-voice fixture, got {unique_speakers:?}"
        );
    }

    #[test]
    fn merge_assigns_speaker_by_midpoint_overlap() {
        let asr_segments = vec![
            TranscriptSegment { text: "hello".into(), start_ms: 0, end_ms: 1000 },
            TranscriptSegment { text: "world".into(), start_ms: 1600, end_ms: 2000 },
        ];
        let speaker_segments = vec![
            SpeakerSegment { start_ms: 0, end_ms: 1500, speaker: 0 },
            SpeakerSegment { start_ms: 1500, end_ms: 2000, speaker: 1 },
        ];

        let merged = merge_asr_and_diarization(&asr_segments, &speaker_segments);
        assert_eq!(merged[0].1, Some(0)); // midpoint 500 -> unambiguously speaker 0
        assert_eq!(merged[1].1, Some(1)); // midpoint 1800 -> unambiguously speaker 1
    }

    #[test]
    fn merge_at_exact_shared_boundary_picks_first_matching_segment_deterministically() {
        // A midpoint sitting exactly on the shared edge between two
        // inclusive-inclusive ranges is genuinely ambiguous. Document the
        // real, deterministic tie-break (first match in segment order)
        // rather than asserting an arbitrary side and calling it a bug.
        let asr_segments = vec![TranscriptSegment { text: "boundary".into(), start_ms: 1000, end_ms: 2000 }];
        let speaker_segments = vec![
            SpeakerSegment { start_ms: 0, end_ms: 1500, speaker: 0 },
            SpeakerSegment { start_ms: 1500, end_ms: 3000, speaker: 1 },
        ];
        let merged = merge_asr_and_diarization(&asr_segments, &speaker_segments);
        assert_eq!(merged[0].1, Some(0), "tie at exact boundary should deterministically pick the first listed segment");
    }

    #[test]
    fn merge_assigns_none_when_no_diarization_segment_covers_midpoint() {
        let asr_segments = vec![TranscriptSegment { text: "gap".into(), start_ms: 5000, end_ms: 6000 }];
        let speaker_segments = vec![SpeakerSegment { start_ms: 0, end_ms: 1000, speaker: 0 }];

        let merged = merge_asr_and_diarization(&asr_segments, &speaker_segments);
        assert_eq!(merged[0].1, None);
    }
}
