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
use tauri::Manager;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
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
        .invoke_handler(tauri::generate_handler![greet])
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
