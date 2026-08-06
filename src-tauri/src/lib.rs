// `pub` on the modules an examples/ binary needs (MeetingImport: storage,
// pipeline, audit_log, and the four engine modules) — examples compile
// against this crate as an external dependency, so they only see items
// re-exported at this level.
pub mod asr;
mod audio_capture;
pub mod audit_log;
mod calendar;
mod cloud_sync_gate;
pub mod diarization;
pub mod embeddings;
mod frontier;
mod keychain;
pub mod llm;
pub mod model_provisioning;
mod oauth;
pub mod pipeline;
mod retention;
pub mod speaker_id;
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
fn spawn_engine_loading(models_dir: PathBuf, data_dir: PathBuf, engines_state: Arc<Mutex<Option<Arc<PipelineEngines>>>>) {
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
            None,
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
        let speaker_id = match speaker_id::SpeakerIdEngine::load(&models_dir.join("diarization/speaker-embedding.onnx")) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("failed to load speaker id engine: {e}");
                return;
            }
        };
        match storage::open_connection(&data_dir.join("kai-notetaker.sqlite3"))
            .and_then(|conn| storage::load_all_speaker_embeddings(&conn))
        {
            Ok(samples) => speaker_id.enroll_from_storage(&samples),
            Err(e) => eprintln!("failed to load enrolled speakers at startup (non-fatal): {e}"),
        }

        *engines_state.lock().unwrap() = Some(Arc::new(PipelineEngines { asr, diarization, llm, embedding, speaker_id }));
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
        if let Err(e) = pipeline::process_meeting(&conn, &audit, &engines, meeting_id, &audio_path, None) {
            eprintln!("pipeline processing failed for meeting {meeting_id}: {e}");
        }
    });

    Ok(StopRecordingResult {
        path: path.display().to_string(),
        duration_secs: elapsed,
        meeting_id,
    })
}

/// Re-runs the full pipeline for one already-processed meeting, this time
/// telling diarization exactly how many real people were on the call
/// instead of letting it guess from a voice-similarity threshold. Exists
/// because threshold-based clustering can badly over-split a real call
/// (Jeremiah's real 3-person Smithville call produced up to 12 distinct
/// raw speaker indices) — sherpa-onnx's `num_clusters` mode forces exactly
/// the given count and is a real, officially-supported clustering mode,
/// not a guess-and-check workaround.
#[tauri::command]
fn reprocess_meeting_with_speaker_count(
    meeting_id: i64,
    num_speakers: i32,
    paths: State<AppPaths>,
    engines: State<EnginesState>,
) -> Result<(), String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    let detail = storage::get_meeting_detail(&conn, meeting_id).map_err(|e| e.to_string())?;
    let audio_path = detail.audio_path.ok_or("this meeting has no audio to reprocess")?;

    let models_dir = model_provisioning::resolve_models_dir(&paths.data_dir);
    let fresh_diarization = diarization::DiarizationEngine::load(
        &models_dir.join("diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        &models_dir.join("diarization/speaker-embedding.onnx"),
        Some(num_speakers),
    )
    .map_err(|e| e.to_string())?;

    storage::clear_meeting_processing_data(&conn, meeting_id).map_err(|e| e.to_string())?;
    storage::mark_meeting_processing(&conn, meeting_id).map_err(|e| e.to_string())?;

    let engines_arc = engines.0.lock().map_err(|_| "engines lock poisoned".to_string())?.clone().ok_or("models are still loading — try again shortly")?;
    let db_path = db_path.clone();
    let audit_path = paths.data_dir.join("audit-log.jsonl");
    std::thread::spawn(move || {
        let conn = match storage::open_connection(&db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("reprocess: failed to open db for meeting {meeting_id}: {e}");
                return;
            }
        };
        let audit = AuditLog::new(&audit_path);
        if let Err(e) = pipeline::process_meeting(&conn, &audit, &engines_arc, meeting_id, std::path::Path::new(&audio_path), Some(&fresh_diarization)) {
            eprintln!("reprocess with known speaker count failed for meeting {meeting_id}: {e}");
        }
    });

    Ok(())
}

/// Stores the Microsoft app registration's client ID so it only needs to
/// be entered once, then runs the full interactive OAuth consent flow
/// (opens the user's browser, waits for the redirect, exchanges the code,
/// stores tokens). Blocks the calling command for up to 3 minutes — fine
/// for a rare, explicit "Connect" click, not something polled.
#[tauri::command]
fn connect_microsoft_calendar(client_id: String) -> Result<(), String> {
    oauth::store_client_id(calendar::MICROSOFT_PROVIDER_ID, &client_id).map_err(|e| e.to_string())?;
    // Fixed port, not random: this app only ever runs one connect flow at
    // a time, and a fixed port makes the one-time "add http://localhost
    // as a redirect URI" Azure step unambiguous to describe — Microsoft
    // ignores the port for matching anyway (verified against their own
    // docs), so this isn't load-bearing for correctness, just clarity.
    calendar::connect_microsoft(&client_id, 53682).map_err(|e| e.to_string())
}

#[tauri::command]
fn is_microsoft_calendar_connected() -> Result<bool, String> {
    calendar::is_microsoft_connected().map_err(|e| e.to_string())
}

#[derive(serde::Serialize)]
struct UpcomingMeetingPayload {
    subject: String,
    start: String,
    end: String,
    attendees: Vec<String>,
    join_url: Option<String>,
}

#[tauri::command]
fn list_upcoming_meetings(hours_ahead: i64) -> Result<Vec<UpcomingMeetingPayload>, String> {
    let client_id = oauth::load_client_id(calendar::MICROSOFT_PROVIDER_ID)
        .map_err(|e| e.to_string())?
        .ok_or("Microsoft calendar isn't connected yet")?;
    calendar::list_upcoming_meetings(&client_id, hours_ahead)
        .map(|meetings| {
            meetings
                .into_iter()
                .map(|m| UpcomingMeetingPayload { subject: m.subject, start: m.start, end: m.end, attendees: m.attendees, join_url: m.join_url })
                .collect()
        })
        .map_err(|e| e.to_string())
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
    let data_dir = paths.data_dir.clone();
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
        spawn_engine_loading(models_dir, data_dir, engines_state);
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

#[tauri::command]
fn list_known_speakers(paths: State<AppPaths>) -> Result<Vec<String>, String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    Ok(storage::list_known_speakers(&conn).map_err(|e| e.to_string())?.into_iter().map(|(_, name)| name).collect())
}

/// Labels one or more specific transcript segments. Scoped to exact
/// segment ids (not a whole raw diarization speaker index) by default,
/// because clustering can — and on real long calls, does — merge two
/// different real people into the same index; an index-wide label would
/// then silently mislabel whichever person didn't type the name. Set
/// `apply_to_whole_speaker: true` to opt into the old, simpler behavior
/// (label every segment sharing the first selected segment's raw index)
/// for the common case where diarization got that index right.
///
/// `remember: true` also extracts a real voice sample from the selected
/// audio (just the selected segments' own ranges, or the whole index's
/// ranges when `apply_to_whole_speaker`) and enrolls it — both in the
/// database (survives a restart) and in the live `SpeakerIdEngine`
/// (recognized for the rest of this session immediately). `remember:
/// false` just sets a display label with no persistent identity attached
/// — for a person you don't expect to see again.
#[tauri::command]
fn label_transcript_segments(
    meeting_id: i64,
    segment_ids: Vec<i64>,
    name: String,
    remember: bool,
    apply_to_whole_speaker: bool,
    paths: State<AppPaths>,
    engines: State<EnginesState>,
) -> Result<(), String> {
    if segment_ids.is_empty() {
        return Err("no segments selected".to_string());
    }

    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    let detail = storage::get_meeting_detail(&conn, meeting_id).map_err(|e| e.to_string())?;

    let (ranges, whole_speaker_index): (Vec<(i64, i64)>, Option<i32>) = {
        let selected: Vec<&storage::TranscriptSegmentRow> =
            detail.transcript.iter().filter(|s| segment_ids.contains(&s.id)).collect();
        if selected.is_empty() {
            return Err("selected segments not found in this meeting".to_string());
        }
        if apply_to_whole_speaker {
            let speaker_index = selected[0].speaker.ok_or("selected segment has no diarized speaker")?;
            let ranges = detail
                .transcript
                .iter()
                .filter(|s| s.speaker == Some(speaker_index))
                .map(|s| (s.start_ms, s.end_ms))
                .collect();
            (ranges, Some(speaker_index))
        } else {
            (selected.iter().map(|s| (s.start_ms, s.end_ms)).collect(), None)
        }
    };

    if !remember {
        return match whole_speaker_index {
            Some(speaker_index) => storage::label_meeting_speaker(&conn, meeting_id, speaker_index, None, &name).map_err(|e| e.to_string()),
            None => storage::set_segment_speaker_labels(&conn, &segment_ids, None, &name).map_err(|e| e.to_string()),
        };
    }

    let audio_path = detail.audio_path.ok_or("this meeting has no audio to extract a voice sample from")?;
    let engines_guard = engines.0.lock().map_err(|_| "engines lock poisoned".to_string())?;
    let engines_ref = engines_guard.as_ref().ok_or("models are still loading — try again shortly")?;

    let embedding = pipeline::extract_embedding_for_speaker_ranges(
        std::path::Path::new(&audio_path),
        &ranges,
        &engines_ref.speaker_id,
    )
    .map_err(|e| e.to_string())?;

    let known_speaker_id = storage::get_or_create_known_speaker(&conn, &name).map_err(|e| e.to_string())?;
    storage::add_speaker_embedding_sample(&conn, known_speaker_id, &embedding, Some(meeting_id)).map_err(|e| e.to_string())?;
    match whole_speaker_index {
        Some(speaker_index) => storage::label_meeting_speaker(&conn, meeting_id, speaker_index, Some(known_speaker_id), &name).map_err(|e| e.to_string())?,
        None => storage::set_segment_speaker_labels(&conn, &segment_ids, Some(known_speaker_id), &name).map_err(|e| e.to_string())?,
    }
    engines_ref.speaker_id.enroll(&name, &embedding);

    Ok(())
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
            undelete_meeting,
            list_known_speakers,
            label_transcript_segments,
            reprocess_meeting_with_speaker_count,
            connect_microsoft_calendar,
            is_microsoft_calendar_connected,
            list_upcoming_meetings
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
            spawn_engine_loading(models_dir, data_dir.clone(), engines_state);

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
