mod asr;
mod audio_capture;
mod audit_log;
mod cloud_sync_gate;
mod diarization;
mod embeddings;
mod frontier;
mod llm;
mod retention;
mod summarization;

use audit_log::AuditLog;
use retention::RetentionPolicy;
use rusqlite::Connection;
use std::sync::Mutex;
use std::time::Instant;
use tauri::{Manager, State};

/// Holds the in-progress recording (if any) across separate command
/// invocations from the frontend (start → ... → stop are two distinct
/// calls). `cpal::Stream` is confirmed `Send + Sync` on macOS via the
/// crate's own compile-time assertion, so this is sound to hold in
/// Tauri's managed state without any unsafe wrapper.
#[derive(Default)]
struct RecordingState(Mutex<Option<(audio_capture::RecordingSession, Instant)>>);

#[tauri::command]
fn list_audio_devices() -> Result<Vec<audio_capture::InputDeviceInfo>, String> {
    audio_capture::list_input_devices().map_err(|e| e.to_string())
}

#[tauri::command]
fn start_recording(app: tauri::AppHandle, state: State<RecordingState>) -> Result<(), String> {
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let recording_id = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();

    let session = audio_capture::RecordingSession::start(&data_dir, &recording_id)
        .map_err(|e| e.to_string())?;

    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    if guard.is_some() {
        return Err("a recording is already in progress".to_string());
    }
    *guard = Some((session, Instant::now()));
    Ok(())
}

#[tauri::command]
fn switch_recording_device(device_name: String, state: State<RecordingState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let (session, _) = guard.as_mut().ok_or("no recording in progress")?;
    session.switch_device(&device_name).map_err(|e| e.to_string())
}

#[derive(serde::Serialize)]
struct StopRecordingResult {
    path: String,
    duration_secs: u64,
}

#[tauri::command]
fn stop_recording(state: State<RecordingState>) -> Result<StopRecordingResult, String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let (session, started_at) = guard.take().ok_or("no recording in progress")?;
    let elapsed = started_at.elapsed().as_secs();
    let path = session.stop_and_write().map_err(|e| e.to_string())?;
    Ok(StopRecordingResult {
        path: path.display().to_string(),
        duration_secs: elapsed,
    })
}

/// Placeholder schema sufficient for the retention scheduler to run against
/// on app startup. This is intentionally NOT the real SQLCipher-encrypted
/// storage layer (see ISA.md Feature: StorageLayer, deferred to a follow-up
/// session) — it exists so ISC-25 (retention actually scheduled on startup)
/// is real and testable now rather than aspirational.
fn ensure_placeholder_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meetings (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS transcripts (id INTEGER PRIMARY KEY, meeting_id INTEGER);
         CREATE TABLE IF NOT EXISTS action_items (id INTEGER PRIMARY KEY, meeting_id INTEGER);
         CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, meeting_id INTEGER);",
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(RecordingState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices,
            start_recording,
            switch_recording_device,
            stop_recording
        ])
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("resolve app data dir");
            std::fs::create_dir_all(&data_dir).expect("create app data dir");

            let db_path = data_dir.join("kai-notetaker.sqlite3");
            let audit_path = data_dir.join("audit-log.jsonl");

            tauri::async_runtime::spawn(async move {
                // Sweep once shortly after launch, then on a fixed interval.
                // Real interval will be tuned once actual usage patterns
                // exist; every-6-hours is a reasonable v1 default that
                // still satisfies "not only on manual trigger" (ISC-25).
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(6 * 60 * 60));
                loop {
                    interval.tick().await;

                    let conn = match Connection::open(&db_path) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("retention sweep: failed to open db: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = ensure_placeholder_schema(&conn) {
                        eprintln!("retention sweep: schema setup failed: {e}");
                        continue;
                    }

                    let audit = AuditLog::new(&audit_path);
                    let policy = RetentionPolicy::default_policy();
                    match retention::retention_sweep(&conn, &audit, policy) {
                        Ok(count) if count > 0 => {
                            println!("retention sweep: deleted {count} expired meeting(s)");
                        }
                        Ok(_) => {}
                        Err(e) => eprintln!("retention sweep failed: {e}"),
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
