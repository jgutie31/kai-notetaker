//! Ties the whole local pipeline together: a finished recording (WAV file
//! at `CANONICAL_SAMPLE_RATE`) goes in, a fully-populated meeting record
//! (transcript, summary, action items, embeddings) comes out in the
//! database. This is the piece that was missing before — without it,
//! recordings were just WAV files nobody ever looked at again.

use crate::asr::AsrEngine;
use crate::audio_capture::{self, CANONICAL_SAMPLE_RATE, PIPELINE_SAMPLE_RATE};
use crate::audit_log::AuditLog;
use crate::diarization::{self, DiarizationEngine};
use crate::embeddings::EmbeddingEngine;
use crate::llm::LlmEngine;
use crate::{storage, summarization};
use rusqlite::Connection;
use serde_json::json;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("wav read error: {0}")]
    Wav(#[from] hound::Error),
    #[error("audio format error: {0}")]
    AudioFormat(String),
    #[error("resample error: {0}")]
    Resample(#[from] crate::audio_capture::AudioCaptureError),
    #[error("asr error: {0}")]
    Asr(#[from] crate::asr::AsrError),
    #[error("diarization error: {0}")]
    Diarization(#[from] crate::diarization::DiarizationError),
    #[error("summarization error: {0}")]
    Summarization(#[from] crate::summarization::SummarizationError),
    #[error("storage error: {0}")]
    Storage(#[from] storage::StorageError),
    #[error("audit log error: {0}")]
    Audit(#[from] crate::audit_log::AuditLogError),
}

/// The four heavy local models, loaded once at app startup and reused
/// across every recording — loading any one of them takes real seconds,
/// so doing it per-recording would make every meeting's processing start
/// with an avoidable multi-second stall.
pub struct PipelineEngines {
    pub asr: AsrEngine,
    pub diarization: DiarizationEngine,
    pub llm: LlmEngine,
    pub embedding: EmbeddingEngine,
}

fn read_wav_mono_at_canonical_rate(path: &Path) -> Result<Vec<f32>, PipelineError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != CANONICAL_SAMPLE_RATE {
        return Err(PipelineError::AudioFormat(format!(
            "expected mono {CANONICAL_SAMPLE_RATE}Hz, got {} channel(s) at {}Hz",
            spec.channels, spec.sample_rate
        )));
    }
    let samples: Result<Vec<f32>, hound::Error> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };
    Ok(samples?)
}

/// Run the full pipeline for one meeting. On any failure, the meeting is
/// marked `failed` with the error message rather than left stuck at
/// `processing` forever — callers don't need to remember to do this
/// themselves.
pub fn process_meeting(
    conn: &Connection,
    audit: &AuditLog,
    engines: &PipelineEngines,
    meeting_id: i64,
    audio_path: &Path,
) -> Result<(), PipelineError> {
    match process_meeting_inner(conn, audit, engines, meeting_id, audio_path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = storage::mark_meeting_failed(conn, meeting_id, &e.to_string());
            Err(e)
        }
    }
}

fn process_meeting_inner(
    conn: &Connection,
    audit: &AuditLog,
    engines: &PipelineEngines,
    meeting_id: i64,
    audio_path: &Path,
) -> Result<(), PipelineError> {
    let samples_48k = read_wav_mono_at_canonical_rate(audio_path)?;
    let samples_16k = audio_capture::one_shot_resample(&samples_48k, CANONICAL_SAMPLE_RATE, PIPELINE_SAMPLE_RATE)?;

    let asr_segments = engines.asr.transcribe(&samples_16k)?;
    let speaker_segments = engines.diarization.diarize(&samples_16k)?;
    let merged = diarization::merge_asr_and_diarization(&asr_segments, &speaker_segments);

    for (i, (segment, speaker)) in merged.iter().enumerate() {
        storage::insert_transcript_segment(
            conn,
            meeting_id,
            i as i64,
            *speaker,
            segment.start_ms,
            segment.end_ms,
            &segment.text,
        )?;
    }

    let labeled_transcript: String = merged
        .iter()
        .map(|(segment, speaker)| {
            let label = speaker.map(|s| format!("Speaker {s}")).unwrap_or_else(|| "Unknown".to_string());
            format!("{label}: {}", segment.text)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let summary = summarization::summarize_meeting(&engines.llm, &labeled_transcript, 4096)?;
    storage::insert_summary(conn, meeting_id, &summary.meeting_summary)?;
    for item in &summary.action_items {
        storage::insert_action_item(conn, meeting_id, &item.description, item.owner.as_deref(), item.due_date.as_deref())?;
    }

    for (segment, _speaker) in &merged {
        if segment.text.trim().is_empty() {
            continue;
        }
        match engines.embedding.embed(&segment.text) {
            Ok(vector) => storage::insert_embedding(conn, meeting_id, &segment.text, &vector)?,
            Err(e) => eprintln!("embedding failed for a transcript segment (non-fatal): {e}"),
        }
    }

    let title: String = summary
        .meeting_summary
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ");
    let title = if title.is_empty() { "Untitled meeting".to_string() } else { title };
    storage::mark_meeting_ready(conn, meeting_id, &title)?;

    audit.append(
        "meeting_processed",
        "system:pipeline",
        json!({ "meeting_id": meeting_id, "segment_count": merged.len(), "action_item_count": summary.action_items.len() }),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit_log::AuditLog;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn model_paths_exist() -> bool {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base.join("models/ggml-base.bin").exists()
            && base.join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx").exists()
            && base.join("models/diarization/speaker-embedding.onnx").exists()
            && base.join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf").exists()
            && base.join("models/embeddings/bge-small-en-v1.5-f16.gguf").exists()
    }

    #[test]
    fn full_pipeline_end_to_end_on_real_48k_fixture() {
        if !model_paths_exist() {
            eprintln!("skipping: not all pipeline models present in this environment");
            return;
        }

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixture = base.join("test-fixtures/test-speech-48k-mono.wav");
        if !fixture.exists() {
            eprintln!("skipping: 48kHz pipeline test fixture not present (see test-fixtures/README or generate via say+afconvert)");
            return;
        }

        let engines = PipelineEngines {
            asr: AsrEngine::load(&base.join("models/ggml-base.bin"), true).unwrap(),
            diarization: DiarizationEngine::load(
                &base.join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                &base.join("models/diarization/speaker-embedding.onnx"),
            )
            .unwrap(),
            llm: LlmEngine::load(&base.join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), 1000).unwrap(),
            embedding: EmbeddingEngine::load(&base.join("models/embeddings/bge-small-en-v1.5-f16.gguf")).unwrap(),
        };

        let conn = Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();
        let meeting_id = storage::create_meeting(&conn, fixture.to_str().unwrap(), 3).unwrap();

        let audit_tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(audit_tmp.path()).ok();
        let audit = AuditLog::new(audit_tmp.path());

        process_meeting(&conn, &audit, &engines, meeting_id, &fixture).unwrap();

        let detail = storage::get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.status, "ready");
        assert!(!detail.transcript.is_empty(), "expected real transcript segments");
        assert!(detail.summary.is_some() && !detail.summary.as_ref().unwrap().trim().is_empty());

        let lower = detail.transcript.iter().map(|s| s.text.to_lowercase()).collect::<Vec<_>>().join(" ");
        assert!(
            lower.contains("test") || lower.contains("record") || lower.contains("transcri") || lower.contains("pipeline"),
            "transcript didn't contain expected fixture content: {lower}"
        );

        assert!(audit.verify_chain().is_ok());
    }

    #[test]
    fn failed_step_marks_meeting_failed_not_stuck_processing() {
        let conn = Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();
        let meeting_id = storage::create_meeting(&conn, "/nonexistent/path.wav", 10).unwrap();

        let audit_tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(audit_tmp.path()).ok();
        let audit = AuditLog::new(audit_tmp.path());

        // Deliberately skip real engine construction (would be slow and
        // is irrelevant to this test) by calling the WAV-read failure
        // path directly through the public entry point with a bogus path
        // — engines are never touched because the read fails first.
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !model_paths_exist() {
            eprintln!("skipping: models not present, cannot construct engines for this test either");
            return;
        }
        let engines = PipelineEngines {
            asr: AsrEngine::load(&base.join("models/ggml-base.bin"), true).unwrap(),
            diarization: DiarizationEngine::load(
                &base.join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                &base.join("models/diarization/speaker-embedding.onnx"),
            )
            .unwrap(),
            llm: LlmEngine::load(&base.join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), 1000).unwrap(),
            embedding: EmbeddingEngine::load(&base.join("models/embeddings/bge-small-en-v1.5-f16.gguf")).unwrap(),
        };

        let result = process_meeting(&conn, &audit, &engines, meeting_id, Path::new("/nonexistent/path.wav"));
        assert!(result.is_err());

        let detail = storage::get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.status, "failed");
        assert!(detail.error_message.is_some());
    }
}
