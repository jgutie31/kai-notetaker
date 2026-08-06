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
