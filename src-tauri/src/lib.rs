// `pub` on the modules an examples/ binary needs (MeetingImport: storage,
// pipeline, audit_log, and the four engine modules) — examples compile
// against this crate as an external dependency, so they only see items
// re-exported at this level.
pub mod asr;
mod audio_capture;
pub mod audit_log;
mod cloud_sync_gate;
pub mod diarization;
pub mod embeddings;
mod frontier;
mod keychain;
pub mod llm;
pub mod model_provisioning;
pub mod pipeline;
mod retention;
pub mod storage;
mod summarization;

use audit_log::AuditLog;
use pipeline::PipelineEngines;
use retention::RetentionPolicy;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{Manager, State};

/// Holds the in-progress recording (if any) across separate command
/// invocations from the frontend (start → ... → stop are two distinct
/// calls). `cpal::Stream` is confirmed `Send + Sync` on macOS via the
/// crate's own compile-time assertion, so this is sound to hold in
/// Tauri's managed state without any unsafe wrapper.
#[derive(Default)]
struct RecordingState(Mutex<Option<(audio_capture::RecordingSession, Instant)>>);

/// The four heavy local models, loaded once in a background OS thread at
/// startup (not blocking the window from appearing) and shared across
/// every recording thereafter. `None` until loading finishes.
#[derive(Default, Clone)]
struct EnginesState(Arc<Mutex<Option<Arc<PipelineEngines>>>>);

struct AppPaths {
    data_dir: PathBuf,
}

/// Loads the four heavy models from `models_dir` in a background OS
/// thread and populates `engines_state` on success. Shared by app startup
/// (models already present) and by `download_missing_models` (models
/// just finished downloading) — both cases converge on the same
/// "models are on disk now, go load them" moment.
fn spawn_engine_loading(models_dir: PathBuf, engines_state: Arc<Mutex<Option<Arc<PipelineEngines>>>>) {
    std::thread::spawn(move || {
        if !model_provisioning::missing_models(&models_dir).is_empty() {
            println!("models not yet provisioned — waiting for first-run download to complete");
            return;
        }

        let asr = match asr::AsrEngine::load(&models_dir.join("ggml-base.bin"), true) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("failed to load ASR engine: {e}");
                return;
            }
        };
        let diarization = match diarization::DiarizationEngine::load(
            &models_dir.join("diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            &models_dir.join("diarization/speaker-embedding.onnx"),
        ) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("failed to load diarization engine: {e}");
                return;
            }
        };
        let llm = match llm::LlmEngine::load(&models_dir.join("llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), 1000) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("failed to load LLM engine: {e}");
                return;
            }
        };
        let embedding = match embeddings::EmbeddingEngine::load(&models_dir.join("embeddings/bge-small-en-v1.5-f16.gguf")) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("failed to load embedding engine: {e}");
                return;
            }
        };

        *engines_state.lock().unwrap() = Some(Arc::new(PipelineEngines { asr, diarization, llm, embedding }));
        println!("all pipeline engines loaded and ready");
    });
}

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
    meeting_id: i64,
}

#[tauri::command]
fn stop_recording(
    state: State<RecordingState>,
    engines_state: State<EnginesState>,
    paths: State<AppPaths>,
) -> Result<StopRecordingResult, String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let (session, started_at) = guard.take().ok_or("no recording in progress")?;
    let elapsed = started_at.elapsed().as_secs();
    let path = session.stop_and_write().map_err(|e| e.to_string())?;

    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let audit_path = paths.data_dir.join("audit-log.jsonl");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::ensure_schema(&conn).map_err(|e| e.to_string())?;
    let meeting_id = storage::create_meeting(&conn, &path.display().to_string(), elapsed).map_err(|e| e.to_string())?;

    // Heavy CPU/GPU-bound work — a real OS thread, not an async task, so
    // it never blocks Tokio's worker pool or the UI thread. Waits for
    // engine loading to finish if it somehow hasn't already (startup
    // loading is normally much faster than a real meeting's length).
    let engines_handle = engines_state.0.clone();
    let audio_path = path.clone();
    std::thread::spawn(move || {
        let engines = loop {
            if let Some(e) = engines_handle.lock().unwrap().clone() {
                break e;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        };
        let conn = match storage::open_connection(&db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("pipeline: failed to open db for meeting {meeting_id}: {e}");
                return;
            }
        };
        let audit = AuditLog::new(&audit_path);
        if let Err(e) = pipeline::process_meeting(&conn, &audit, &engines, meeting_id, &audio_path) {
            eprintln!("pipeline processing failed for meeting {meeting_id}: {e}");
        }
    });

    Ok(StopRecordingResult {
        path: path.display().to_string(),
        duration_secs: elapsed,
        meeting_id,
    })
}

#[tauri::command]
fn list_meetings(paths: State<AppPaths>) -> Result<Vec<storage::MeetingListItem>, String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::ensure_schema(&conn).map_err(|e| e.to_string())?;
    storage::list_meetings(&conn).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_meeting_detail(meeting_id: i64, paths: State<AppPaths>) -> Result<storage::MeetingDetail, String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::get_meeting_detail(&conn, meeting_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn check_missing_models(paths: State<AppPaths>) -> Vec<String> {
    let models_dir = model_provisioning::resolve_models_dir(&paths.data_dir);
    model_provisioning::missing_models(&models_dir)
        .into_iter()
        .map(|spec| spec.name.to_string())
        .collect()
}

/// Downloads every missing model in a background thread, emitting
/// `model-download-progress` events the frontend listens for. Returns
/// immediately — the actual download can take minutes (the LLM alone is
/// ~4.6GB) and must not block the command/UI thread. Real downloads
/// always target `$APPDATA/models`, not the dev-fallback source tree.
#[tauri::command]
fn download_missing_models(app: tauri::AppHandle, paths: State<AppPaths>, engines: State<EnginesState>) {
    use tauri::Emitter;

    let models_dir = paths.data_dir.join("models");
    let engines_state = engines.0.clone();
    let missing: Vec<model_provisioning::ModelSpec> =
        model_provisioning::missing_models(&models_dir).into_iter().cloned().collect();

    std::thread::spawn(move || {
        for spec in &missing {
            let app_for_progress = app.clone();
            let model_name = spec.name.to_string();
            let result = model_provisioning::download_model(spec, &models_dir, |downloaded, total| {
                let _ = app_for_progress.emit(
                    "model-download-progress",
                    serde_json::json!({ "model": model_name, "downloaded": downloaded, "total": total }),
                );
            });
            if let Err(e) = result {
                let _ = app.emit(
                    "model-download-error",
                    serde_json::json!({ "model": spec.name, "error": e.to_string() }),
                );
                return;
            }
        }
        let _ = app.emit("model-download-complete", ());
        spawn_engine_loading(models_dir, engines_state);
    });
}

#[tauri::command]
fn rename_meeting(meeting_id: i64, title: String, paths: State<AppPaths>) -> Result<(), String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::rename_meeting(&conn, meeting_id, &title).map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_meeting(meeting_id: i64, paths: State<AppPaths>) -> Result<(), String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::delete_meeting(&conn, meeting_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn undelete_meeting(meeting_id: i64, paths: State<AppPaths>) -> Result<(), String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::undelete_meeting(&conn, meeting_id).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(RecordingState::default())
        .manage(EnginesState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices,
            start_recording,
            switch_recording_device,
            stop_recording,
            list_meetings,
            get_meeting_detail,
            check_missing_models,
            download_missing_models,
            rename_meeting,
            delete_meeting,
            undelete_meeting
        ])
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("resolve app data dir");
            std::fs::create_dir_all(&data_dir).expect("create app data dir");
            app.manage(AppPaths { data_dir: data_dir.clone() });

            let db_path = data_dir.join("kai-notetaker.sqlite3");
            let audit_path = data_dir.join("audit-log.jsonl");

            // Real schema, created up front so list_meetings/get_meeting_detail
            // never race against a not-yet-created table.
            {
                let conn = storage::open_connection(&db_path).expect("open db at startup");
                storage::ensure_schema(&conn).expect("create schema at startup");
            }

            // Load the four heavy models in a background OS thread so the
            // window appears immediately rather than stalling on multi-
            // second model loads.
            let engines_state = app.state::<EnginesState>().0.clone();
            let models_dir = model_provisioning::resolve_models_dir(&data_dir);
            spawn_engine_loading(models_dir, engines_state);

            tauri::async_runtime::spawn(async move {
                // Sweep once shortly after launch, then on a fixed interval.
                // Real interval will be tuned once actual usage patterns
                // exist; every-6-hours is a reasonable v1 default that
                // still satisfies "not only on manual trigger" (ISC-25).
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(6 * 60 * 60));
                loop {
                    interval.tick().await;

                    let conn = match storage::open_connection(&db_path) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("retention sweep: failed to open db: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = storage::ensure_schema(&conn) {
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
