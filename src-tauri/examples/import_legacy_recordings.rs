//! One-off migration: import the old screenshot-based notetaker's real
//! audio recordings (never the screenshots — those live in a separate
//! directory entirely) into kai-notetaker's own pipeline and encrypted
//! database, so this stops being two separate systems holding real KCG
//! meeting audio. Run once, then the `meeting-watcher` PULSE.toml job
//! gets disabled.
//!
//! `cargo run --example import_legacy_recordings`

use kai_notetaker_lib::asr::AsrEngine;
use kai_notetaker_lib::audit_log::AuditLog;
use kai_notetaker_lib::diarization::DiarizationEngine;
use kai_notetaker_lib::embeddings::EmbeddingEngine;
use kai_notetaker_lib::llm::LlmEngine;
use kai_notetaker_lib::pipeline::{self, PipelineEngines};
use kai_notetaker_lib::speaker_id::SpeakerIdEngine;
use kai_notetaker_lib::storage;
use std::path::PathBuf;

fn main() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let old_recordings_dir =
        PathBuf::from(&home).join(".claude/PAI/MEMORY/WORK/meeting-notetaker-agent/recordings");
    let app_data_dir =
        PathBuf::from(&home).join("Library/Application Support/com.kairoscompliance.kainotetaker");
    let new_recordings_dir = app_data_dir.join("recordings");
    let db_path = app_data_dir.join("kai-notetaker.sqlite3");
    let audit_path = app_data_dir.join("audit-log.jsonl");

    // Diagnostic mode: `cargo run --example import_legacy_recordings -- inspect <meeting_id>`
    // Reuses this already-Keychain-approved binary identity rather than a
    // new example (a new binary path triggers a fresh macOS authorization
    // prompt — confirmed the hard way earlier this session).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("inspect") {
        let meeting_id: i64 = args.get(2).expect("usage: inspect <meeting_id>").parse().expect("meeting_id must be an integer");
        let conn = storage::open_connection(&db_path).expect("open real encrypted db");
        let detail = storage::get_meeting_detail(&conn, meeting_id).expect("get meeting detail");
        println!("id={} title={:?} status={} duration={}s", detail.id, detail.title, detail.status, detail.duration_secs);
        println!("summary: {:?}", detail.summary);
        println!("action_items: {} total", detail.action_items.len());
        for item in &detail.action_items {
            println!("  - {:?} (owner={:?}, due={:?})", item.description, item.owner, item.due_date);
        }
        let distinct_speakers: std::collections::BTreeSet<Option<i32>> =
            detail.transcript.iter().map(|s| s.speaker).collect();
        println!("transcript: {} segments, distinct speaker labels: {:?}", detail.transcript.len(), distinct_speakers);
        for seg in &detail.transcript {
            println!("  [{}ms-{}ms] speaker={:?}: {}", seg.start_ms, seg.end_ms, seg.speaker, seg.text);
        }
        return;
    }

    // Clean-summaries mode: strips a leading LLM preamble (e.g. "Here is a
    // combined meeting summary of 6 sentences:") from already-stored
    // summaries — real bug Jeremiah reported after the fact. The prompt
    // fix (summarization.rs) only affects summaries generated from here
    // on; this is the one-off cleanup for what's already in the database.
    if args.get(1).map(String::as_str) == Some("clean-summaries") {
        let conn = storage::open_connection(&db_path).expect("open real encrypted db");
        let mut stmt = conn.prepare("SELECT meeting_id, summary_text FROM meeting_summaries").unwrap();
        let rows: Vec<(i64, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        drop(stmt);

        let mut cleaned = 0;
        for (meeting_id, text) in rows {
            let lower = text.to_lowercase();
            if !lower.starts_with("here is") && !lower.starts_with("here's") {
                continue;
            }
            // The preamble always ends with a colon before the real
            // content — strip through the first "colon then newline"
            // within a reasonable prefix window, then trim leading
            // whitespace/newlines from what remains.
            let search_window = &text[..text.len().min(120)];
            if let Some(colon_pos) = search_window.find(':') {
                let stripped = text[colon_pos + 1..].trim_start();
                if !stripped.is_empty() {
                    conn.execute(
                        "UPDATE meeting_summaries SET summary_text = ?1 WHERE meeting_id = ?2",
                        rusqlite::params![stripped, meeting_id],
                    )
                    .unwrap();
                    println!("cleaned meeting_id={meeting_id}: {:?} -> {:?}", &text[..colon_pos.min(text.len())], &stripped[..stripped.len().min(60)]);
                    cleaned += 1;
                }
            }
        }
        println!("cleaned {cleaned} summaries");
        return;
    }

    // Reprocess mode: `cargo run --example import_legacy_recordings -- reprocess <meeting_id>`
    // Re-runs the full pipeline (ASR/diarization/summarization) against
    // the SAME already-imported audio file, using whatever fixes have
    // landed in the pipeline code since the original import — e.g. the
    // diarization clustering threshold fix and the ASR no_context fix.
    // Clears the meeting's existing transcript/summary/action-items/
    // embeddings first so the new run doesn't append alongside stale data.
    if args.get(1).map(String::as_str) == Some("reprocess") {
        let meeting_id: i64 = args.get(2).expect("usage: reprocess <meeting_id>").parse().expect("meeting_id must be an integer");
        let conn = storage::open_connection(&db_path).expect("open real encrypted db");
        let existing = storage::get_meeting_detail(&conn, meeting_id).expect("get meeting detail");
        let audio_path = existing.audio_path.expect("meeting has no audio_path to reprocess");

        conn.execute("DELETE FROM transcript_segments WHERE meeting_id = ?1", [meeting_id]).unwrap();
        conn.execute("DELETE FROM meeting_summaries WHERE meeting_id = ?1", [meeting_id]).unwrap();
        conn.execute("DELETE FROM action_items WHERE meeting_id = ?1", [meeting_id]).unwrap();
        conn.execute("DELETE FROM meeting_embeddings WHERE meeting_id = ?1", [meeting_id]).unwrap();
        // Old diarization runs can assign different speaker indices than the new run —
        // stale labels keyed on (meeting_id, speaker_index) would otherwise attach to
        // the wrong voice after reprocessing.
        conn.execute("DELETE FROM meeting_speaker_labels WHERE meeting_id = ?1", [meeting_id]).unwrap();
        conn.execute("UPDATE meetings SET status = 'processing', title = NULL WHERE id = ?1", [meeting_id]).unwrap();

        println!("loading engines (this takes a while — 4 real models)...");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let engines = PipelineEngines {
            asr: AsrEngine::load(&manifest_dir.join("models/ggml-base.bin"), true).expect("load ASR"),
            diarization: DiarizationEngine::load(
                &manifest_dir.join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                &manifest_dir.join("models/diarization/speaker-embedding.onnx"),
            )
            .expect("load diarization"),
            llm: LlmEngine::load(&manifest_dir.join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), 1000)
                .expect("load LLM"),
            embedding: EmbeddingEngine::load(&manifest_dir.join("models/embeddings/bge-small-en-v1.5-f16.gguf"))
                .expect("load embeddings"),
            speaker_id: SpeakerIdEngine::load(&manifest_dir.join("models/diarization/speaker-embedding.onnx"))
                .expect("load speaker id engine"),
        };
        println!("engines loaded. reprocessing meeting_id={meeting_id} ({audio_path})...");

        let audit = AuditLog::new(&audit_path);
        match pipeline::process_meeting(&conn, &audit, &engines, meeting_id, std::path::Path::new(&audio_path)) {
            Ok(()) => println!("OK: reprocessed meeting_id={meeting_id}"),
            Err(e) => eprintln!("FAILED reprocessing meeting_id={meeting_id}: {e}"),
        }
        return;
    }

    std::fs::create_dir_all(&new_recordings_dir).expect("create recordings dir");

    println!("loading engines (this takes a while — 4 real models)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let engines = PipelineEngines {
        asr: AsrEngine::load(&manifest_dir.join("models/ggml-base.bin"), true).expect("load ASR"),
        diarization: DiarizationEngine::load(
            &manifest_dir.join("models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            &manifest_dir.join("models/diarization/speaker-embedding.onnx"),
        )
        .expect("load diarization"),
        llm: LlmEngine::load(&manifest_dir.join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), 1000)
            .expect("load LLM"),
        embedding: EmbeddingEngine::load(&manifest_dir.join("models/embeddings/bge-small-en-v1.5-f16.gguf"))
            .expect("load embeddings"),
        speaker_id: SpeakerIdEngine::load(&manifest_dir.join("models/diarization/speaker-embedding.onnx"))
            .expect("load speaker id engine"),
    };
    println!("engines loaded.");

    let conn = storage::open_connection(&db_path).expect("open real encrypted db");
    storage::ensure_schema(&conn).expect("ensure schema");
    let audit = AuditLog::new(&audit_path);

    println!("current meetings in the real database:");
    for m in storage::list_meetings(&conn).expect("list meetings") {
        println!("  id={} title={:?} status={} duration={}s", m.id, m.title, m.status, m.duration_secs);
    }

    // Crash recovery: a previous run of this same script can leave a
    // meeting row stuck at 'processing' if the process aborted mid-way
    // (a real llama.cpp assertion abort skips Rust's own failure-handling
    // wrapper entirely, since it terminates the process rather than
    // unwinding). Any non-ready row whose audio_path is under this run's
    // own recordings directory is a leftover from a prior attempt at THIS
    // migration, not real user data to protect — clear it so the retry
    // below starts fresh instead of skipping (idempotency check further
    // down only trusts a 'ready' status).
    let stale_ids: Vec<i64> = conn
        .prepare("SELECT id FROM meetings WHERE status != 'ready' AND audio_path LIKE ?1")
        .unwrap()
        .query_map([format!("{}%", new_recordings_dir.display())], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    for id in stale_ids {
        println!("cleaning up stale non-ready meeting row from a previous attempt: id={id}");
        conn.execute("DELETE FROM transcript_segments WHERE meeting_id = ?1", [id]).unwrap();
        conn.execute("DELETE FROM meeting_summaries WHERE meeting_id = ?1", [id]).unwrap();
        conn.execute("DELETE FROM action_items WHERE meeting_id = ?1", [id]).unwrap();
        conn.execute("DELETE FROM meeting_embeddings WHERE meeting_id = ?1", [id]).unwrap();
        conn.execute("DELETE FROM meeting_speaker_labels WHERE meeting_id = ?1", [id]).unwrap();
        conn.execute("DELETE FROM meetings WHERE id = ?1", [id]).unwrap();
    }

    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&old_recordings_dir)
        .expect("read old recordings dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "wav").unwrap_or(false))
        .collect();
    wavs.sort();

    println!("found {} recordings to import", wavs.len());

    let mut succeeded = 0;
    let mut failed = 0;

    for src in &wavs {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let dest_name = format!("imported-{stem}.wav");
        let dest = new_recordings_dir.join(&dest_name);

        let already_ready: bool = conn
            .query_row(
                "SELECT 1 FROM meetings WHERE audio_path = ?1 AND status = 'ready'",
                [dest.to_str().unwrap()],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if already_ready {
            println!("--- skipping {stem}: already imported and ready ---");
            succeeded += 1;
            continue;
        }

        println!("--- importing {stem} ---");

        if let Err(e) = std::fs::copy(src, &dest) {
            eprintln!("FAILED to copy {stem}: {e}");
            failed += 1;
            continue;
        }

        let duration_secs = match hound::WavReader::open(&dest) {
            Ok(reader) => {
                let spec = reader.spec();
                let n = reader.len() as u64;
                n / spec.channels as u64 / spec.sample_rate as u64
            }
            Err(e) => {
                eprintln!("FAILED to read wav spec for {stem}: {e}");
                failed += 1;
                continue;
            }
        };

        let meeting_id = match storage::create_meeting(&conn, dest.to_str().unwrap(), duration_secs) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("FAILED to create meeting row for {stem}: {e}");
                failed += 1;
                continue;
            }
        };

        match pipeline::process_meeting(&conn, &audit, &engines, meeting_id, &dest) {
            Ok(()) => {
                println!("OK: {stem} -> meeting_id={meeting_id}, duration={duration_secs}s");
                succeeded += 1;
            }
            Err(e) => {
                eprintln!("FAILED processing {stem}: {e}");
                failed += 1;
            }
        }
    }

    println!("\n=== import complete: {succeeded} succeeded, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
}
