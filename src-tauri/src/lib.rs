// `pub` on the modules an examples/ binary needs (MeetingImport: storage,
// pipeline, audit_log, and the four engine modules) — examples compile
// against this crate as an external dependency, so they only see items
// re-exported at this level.
pub mod asr;
mod audio_capture;
pub mod audit_log;
mod auto_join;
mod calendar;
mod cloud_sync_gate;
pub mod diarization;
pub mod embeddings;
mod frontier;
mod google;
mod keychain;
pub mod llm;
pub mod model_provisioning;
mod oauth;
pub mod pipeline;
mod retention;
mod silence_monitor;
pub mod speaker_id;
pub mod storage;
mod summarization;
mod zoom;

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

/// Tracks the meeting AutoJoinRecording is currently capturing, if any —
/// distinct from `RecordingState` itself so auto-stop only ever ends a
/// recording *it* started, never a manually-started one that happens to
/// still be running when a poll cycle fires (Jeremiah's real requirement:
/// "make sure there IS an auto-stop when the call ends").
#[derive(Default)]
struct AutoRecordingState(Mutex<Option<AutoRecordingMarker>>);

struct AutoRecordingMarker {
    subject: String,
    /// The meeting's real end time — parsed once at trigger time so the
    /// stop check never has to re-fetch or re-parse anything.
    end: chrono::DateTime<chrono::Utc>,
}

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
    auto_recording_state: State<AutoRecordingState>,
) -> Result<StopRecordingResult, String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let (session, started_at) = guard.take().ok_or("no recording in progress")?;
    let elapsed = started_at.elapsed().as_secs();
    let path = session.stop_and_write().map_err(|e| e.to_string())?;

    // Clear the auto-recording marker on ANY stop — manual click or
    // auto-stop — so a manually-stopped auto-triggered meeting doesn't
    // get a second, redundant auto-stop attempt on the next poll cycle.
    if let Ok(mut marker_guard) = auto_recording_state.0.lock() {
        *marker_guard = None;
    }

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

/// Google Calendar's equivalent of `connect_microsoft_calendar` — same
/// store-then-consent shape, same blocking-for-up-to-3-minutes behavior,
/// different provider. Distinct loopback port per provider purely so two
/// connect flows started in quick succession can't collide on the same
/// socket.
#[tauri::command]
fn connect_google_calendar(client_id: String) -> Result<(), String> {
    oauth::store_client_id(google::GOOGLE_PROVIDER_ID, &client_id).map_err(|e| e.to_string())?;
    google::connect_google(&client_id, 53683).map_err(|e| e.to_string())
}

#[tauri::command]
fn is_google_calendar_connected() -> Result<bool, String> {
    google::is_google_connected().map_err(|e| e.to_string())
}

/// Zoom's equivalent. The client ID here must come from a Zoom app
/// registered with **Use Public Client OAuth** enabled — a standard
/// (confidential) Zoom app's client ID will fail the token exchange,
/// since this app deliberately sends no client secret (ISC-192).
#[tauri::command]
fn connect_zoom(client_id: String) -> Result<(), String> {
    oauth::store_client_id(zoom::ZOOM_PROVIDER_ID, &client_id).map_err(|e| e.to_string())?;
    zoom::connect_zoom(&client_id, 53684).map_err(|e| e.to_string())
}

#[tauri::command]
fn is_zoom_connected() -> Result<bool, String> {
    zoom::is_zoom_connected().map_err(|e| e.to_string())
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

/// Every provider this app can poll. Adding a fourth is one entry here
/// plus its own module — `run_auto_join_cycle` below never names one.
const ALL_PROVIDER_IDS: [&str; 3] =
    [calendar::MICROSOFT_PROVIDER_ID, google::GOOGLE_PROVIDER_ID, zoom::ZOOM_PROVIDER_ID];

/// One fetcher closure per *genuinely connected* provider — zero, one,
/// two, or three of them (ISC-198).
///
/// "Connected" means both a stored client id AND stored tokens: a provider
/// where Jeremiah pasted a client id but abandoned the browser consent has
/// no usable credentials, and including it would fire a guaranteed-failing
/// request every single poll cycle (ISC-199). That predicate lives in
/// `auto_join::ProviderConnection::is_active` so it's unit-testable against
/// a real 0/1/2/3 fixture matrix without writing fake tokens into the real
/// Keychain slots.
///
/// A Keychain read that errors is treated as "not connected" rather than
/// propagated: this runs on a background timer forever, and one transient
/// secure-storage hiccup should cost one cycle, not kill the poller.
fn active_fetchers() -> Vec<auto_join::MeetingFetcher> {
    let states: Vec<auto_join::ProviderConnection> = ALL_PROVIDER_IDS
        .iter()
        .map(|provider_id| auto_join::ProviderConnection {
            provider_id,
            client_id: oauth::load_client_id(provider_id).unwrap_or(None),
            has_tokens: oauth::load_tokens(provider_id).map(|t| t.is_some()).unwrap_or(false),
        })
        .collect();

    auto_join::active_provider_client_ids(&states)
        .into_iter()
        .filter_map(|(provider_id, client_id)| -> Option<auto_join::MeetingFetcher> {
            if provider_id == calendar::MICROSOFT_PROVIDER_ID {
                Some(Box::new(move || {
                    calendar::list_upcoming_meetings(&client_id, auto_join::FETCH_WINDOW_HOURS).map_err(|e| e.to_string())
                }))
            } else if provider_id == google::GOOGLE_PROVIDER_ID {
                Some(Box::new(move || {
                    google::list_upcoming_meetings(&client_id, auto_join::FETCH_WINDOW_HOURS).map_err(|e| e.to_string())
                }))
            } else if provider_id == zoom::ZOOM_PROVIDER_ID {
                // No hours-ahead argument: Zoom's list endpoint has no
                // time-window parameters at all (see `zoom::list_upcoming_meetings`).
                Some(Box::new(move || zoom::list_upcoming_meetings(&client_id).map_err(|e| e.to_string())))
            } else {
                eprintln!("auto-join: no fetcher is registered for provider '{provider_id}' — skipping it this cycle");
                None
            }
        })
        .collect()
}

/// One AutoJoinRecording poll cycle: the side-effecting half of the
/// feature. All the decision logic lives in `auto_join` (pure, unit
/// tested); this function only supplies real inputs and carries out the
/// results — open the link, start the recording, write the log row.
///
/// Every failure path here logs and returns. Nothing panics: this runs
/// forever in a background task, and a panic would take a real recording
/// down with it.
fn run_auto_join_cycle(app: &tauri::AppHandle, db_path: &std::path::Path) {
    // Auto-stop check runs FIRST and unconditionally — independent of the
    // enabled toggle or a stored client id. A recording this feature
    // already started must still get stopped even if Jeremiah unticks the
    // box mid-meeting; leaving a mic capturing indefinitely is exactly the
    // real risk he flagged ("make sure there IS an auto-stop when the call
    // ends"). Distinct from the new-trigger path below, which IS gated.
    {
        let due_subject = match app.state::<AutoRecordingState>().0.lock() {
            Ok(guard) => guard
                .as_ref()
                .and_then(|m| auto_join::should_auto_stop(m.end, chrono::Utc::now()).then(|| m.subject.clone())),
            Err(_) => {
                eprintln!("auto-join: auto-recording marker lock poisoned — skipping auto-stop check this cycle");
                None
            }
        };
        if let Some(subject) = due_subject {
            match stop_recording(
                app.state::<RecordingState>(),
                app.state::<EnginesState>(),
                app.state::<AppPaths>(),
                app.state::<AutoRecordingState>(),
            ) {
                Ok(result) => println!("auto-join: auto-stopped '{subject}' — meeting_id={}", result.meeting_id),
                Err(e) => eprintln!("auto-join: failed to auto-stop '{subject}': {e}"),
            }
        }
    }

    // First check, deliberately before touching the database (ISC-173):
    // a user who has never connected ANY calendar gets zero poller
    // activity — no error log, no work, no noise every 60 seconds. This
    // is the per-provider generalization of the old Microsoft-only
    // "no client id, return early" check: the poller now stands down only
    // when nothing at all is connected, not when Microsoft specifically
    // isn't (ISC-198).
    let fetchers = active_fetchers();
    if fetchers.is_empty() {
        return;
    }

    let conn = match storage::open_connection(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("auto-join: failed to open db: {e}");
            return;
        }
    };
    if let Err(e) = storage::ensure_schema(&conn) {
        eprintln!("auto-join: schema setup failed: {e}");
        return;
    }

    // Re-read every cycle, never cached at startup — that's what makes
    // unticking the box take effect within 60s (ISC-166).
    let enabled = match storage::get_auto_join_enabled(&conn) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("auto-join: could not read the auto_join_enabled setting: {e}");
            return;
        }
    };
    if !enabled {
        return;
    }

    let already_recording = match app.state::<RecordingState>().0.lock() {
        Ok(guard) => guard.is_some(),
        Err(_) => {
            eprintln!("auto-join: recording state lock poisoned — skipping this cycle");
            return;
        }
    };

    let decisions = auto_join::poll_cycle(
        enabled,
        &fetchers,
        chrono::Utc::now(),
        already_recording,
        &|event_id| storage::was_already_auto_joined(&conn, event_id).map_err(|e| e.to_string()),
    );

    for (meeting, decision) in decisions {
        if !decision.opens_join_url() {
            continue;
        }
        let Some(join_url) = meeting.join_url.as_deref() else { continue };

        println!("auto-join: {} — opening join link ({decision:?})", meeting.subject);
        if let Err(e) = open::that(join_url) {
            eprintln!("auto-join: failed to open join link for '{}': {e}", meeting.subject);
        }

        if decision.starts_recording() {
            // The exact same command function the Start Recording button
            // calls — not a parallel recording path — so an auto-started
            // meeting produces an identical on-disk artifact and runs the
            // identical downstream pipeline (ISC-171).
            if let Err(e) = start_recording(app.clone(), app.state::<RecordingState>()) {
                eprintln!("auto-join: failed to start recording for '{}': {e}", meeting.subject);
            } else {
                println!("auto-join: started recording for '{}'", meeting.subject);
                // Record what we started so the auto-stop check above can
                // end THIS recording when the meeting's real end time
                // passes — never a manually-started recording, since only
                // this path ever writes the marker (ISC-181).
                if let Some(end) = auto_join::parse_graph_utc(&meeting.end) {
                    if let Ok(mut marker_guard) = app.state::<AutoRecordingState>().0.lock() {
                        *marker_guard = Some(AutoRecordingMarker { subject: meeting.subject.clone(), end });
                    }
                } else {
                    eprintln!(
                        "auto-join: could not parse end time for '{}' — auto-stop will not fire for this meeting, manual stop still works",
                        meeting.subject
                    );
                }
            }
        }

        if decision.should_log() {
            if let Err(e) = storage::log_auto_join(&conn, &meeting.id, &meeting.subject) {
                eprintln!("auto-join: failed to record '{}' in the auto-join log: {e}", meeting.subject);
            }
        }
    }
}

/// One tick of the silence monitor. Deliberately a synchronous function
/// rather than inline async-block code: every mutex guard it takes is
/// confined to this call, so none can be held across an `.await` in the
/// caller's loop.
///
/// Order of checks is load-bearing:
/// 1. **No auto-recording marker → do nothing, ever** (ISC-205). This is
///    the hard scope boundary: a manually-started recording is never
///    monitored and can never be prompted. The tracker is also reset here
///    so a later auto-recording can't inherit a stale silence run.
/// 2. A dialog already on screen → don't stack a second one.
/// 3. No live recording → nothing to measure.
/// 4. Only then: read RMS and feed the tracker.
fn silence_monitor_tick(
    app: &tauri::AppHandle,
    tracker: &Mutex<silence_monitor::SilenceTracker>,
    dialog_open: &Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;

    // ISC-205 / ISC-206: this feature applies ONLY to recordings
    // AutoJoinRecording itself started (the marker is written at
    // auto-trigger time and nowhere else). Jeremiah's explicit scope cut:
    // "No manual recording ideas for now on this functionality request."
    let is_auto_started = match app.state::<AutoRecordingState>().0.lock() {
        Ok(guard) => guard.is_some(),
        Err(_) => return,
    };

    if dialog_open.load(Ordering::SeqCst) {
        return;
    }

    // `None` here means "monitoring doesn't apply right now" — either this
    // isn't an auto-started recording, or nothing is recording at all. Both
    // clear the tracker so a later auto-recording can't inherit a stale
    // silence run.
    let rms: Option<Option<f32>> = if !is_auto_started {
        None
    } else {
        match app.state::<RecordingState>().0.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|(session, _)| session.trailing_rms(silence_monitor::SILENCE_WINDOW_SECS)),
            Err(_) => return,
        }
    };

    // The tracker guard is scoped to this block and dropped before the
    // dialog is shown. Not cosmetic: `prompt_stop_or_continue`'s callback
    // runs on the UI thread and locks this same tracker to handle
    // "Continue", so holding the guard across `.show()` would risk a
    // deadlock against a `std::sync::Mutex` that isn't reentrant.
    let action = {
        let Ok(mut tracker) = tracker.lock() else { return };
        match rms {
            Some(reading) => tracker.observe(reading, chrono::Utc::now()),
            None => {
                tracker.reset();
                return;
            }
        }
    };

    if action != silence_monitor::SilenceAction::Prompt {
        return;
    }

    println!(
        "silence monitor: {}s of continuous silence on an auto-started recording — asking whether to stop",
        silence_monitor::SILENCE_WINDOW_SECS
    );
    prompt_stop_or_continue(app, dialog_open.clone());
}

/// The native OS Stop/Continue prompt.
///
/// Non-blocking `.show()` with a callback (not `.blocking_show()`): this
/// is reached from inside an async interval loop, and blocking there would
/// stall the loop for as long as the dialog sits unanswered — which, given
/// the realistic case is Jeremiah looking at the call rather than at this
/// app, could be a very long time.
///
/// Verified against the installed `tauri-plugin-dialog` 2.7.2 source, not
/// assumed: `MessageDialogButtons::OkCancelCustom(String, String)` and
/// `show<F: FnOnce(bool) + Send + 'static>`, where the `bool` is `true`
/// only when the FIRST custom label was clicked. So `true` = "Stop", and
/// every other outcome — including dismissing the dialog outright — is
/// `false` = "Continue". That default is the safe one: an ignored prompt
/// leaves the recording running rather than silently ending a live meeting.
fn prompt_stop_or_continue(app: &tauri::AppHandle, dialog_open: Arc<std::sync::atomic::AtomicBool>) {
    use std::sync::atomic::Ordering;
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};

    // Set before showing, cleared in the callback — the guard against
    // stacking a second dialog while one is already on screen.
    dialog_open.store(true, Ordering::SeqCst);

    let app_for_callback = app.clone();
    app.dialog()
        .message("No one has spoken for the last minute. Is this meeting over?")
        .title("Still recording?")
        .buttons(MessageDialogButtons::OkCancelCustom("Stop".to_string(), "Continue".to_string()))
        .show(move |stop| {
            if stop {
                // ISC-203: the exact same function every other stop path
                // calls — the Stop Recording button, and ISC-181's
                // calendar-end auto-stop. Not a parallel stop mechanism,
                // so the recording is written, the marker cleared, and the
                // pipeline kicked off identically however it was ended.
                match stop_recording(
                    app_for_callback.state::<RecordingState>(),
                    app_for_callback.state::<EnginesState>(),
                    app_for_callback.state::<AppPaths>(),
                    app_for_callback.state::<AutoRecordingState>(),
                ) {
                    Ok(result) => println!("silence monitor: stopped on request — meeting_id={}", result.meeting_id),
                    Err(e) => eprintln!("silence monitor: failed to stop the recording: {e}"),
                }
            } else {
                // ISC-204: "Continue" (or a dismissed dialog) requires a
                // fresh, full continuous silence window before this can
                // fire again — never an immediate re-prompt on the next
                // 5-second check.
                println!("silence monitor: continuing — a fresh silence window is required before asking again");
                if let Ok(mut tracker) = app_for_callback.state::<SilenceTrackerState>().0.lock() {
                    tracker.reset();
                }
            }
            dialog_open.store(false, Ordering::SeqCst);
        });
}

/// The silence run-length tracker, in managed state so the dialog's
/// callback (which runs on the UI thread, not the monitor loop) can reset
/// it directly when the user chooses "Continue".
#[derive(Default)]
struct SilenceTrackerState(Mutex<silence_monitor::SilenceTracker>);

#[derive(serde::Serialize)]
struct AutoJoinLogEntry {
    event_id: String,
    subject: String,
    /// Unix seconds — formatted for display by the frontend.
    triggered_at: i64,
}

#[tauri::command]
fn set_auto_join_enabled(enabled: bool, paths: State<AppPaths>) -> Result<(), String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::ensure_schema(&conn).map_err(|e| e.to_string())?;
    storage::set_auto_join_enabled(&conn, enabled).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_auto_join_enabled(paths: State<AppPaths>) -> Result<bool, String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::ensure_schema(&conn).map_err(|e| e.to_string())?;
    storage::get_auto_join_enabled(&conn).map_err(|e| e.to_string())
}

#[tauri::command]
fn list_auto_joined_meetings(paths: State<AppPaths>) -> Result<Vec<AutoJoinLogEntry>, String> {
    let db_path = paths.data_dir.join("kai-notetaker.sqlite3");
    let conn = storage::open_connection(&db_path).map_err(|e| e.to_string())?;
    storage::ensure_schema(&conn).map_err(|e| e.to_string())?;
    Ok(storage::list_auto_joined(&conn)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(event_id, triggered_at, subject)| AutoJoinLogEntry { event_id, subject, triggered_at })
        .collect())
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
        // Native OS message dialogs, for the silence-based Stop/Continue
        // prompt (ISC-202). Registered exactly as the plugin's own v2 docs
        // specify.
        .plugin(tauri_plugin_dialog::init())
        .manage(RecordingState::default())
        .manage(EnginesState::default())
        .manage(AutoRecordingState::default())
        .manage(SilenceTrackerState::default())
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
            connect_google_calendar,
            is_google_calendar_connected,
            connect_zoom,
            is_zoom_connected,
            list_upcoming_meetings,
            set_auto_join_enabled,
            get_auto_join_enabled,
            list_auto_joined_meetings
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

            // AutoJoinRecording's poller — same shape as the retention
            // sweep above (own DB connection per cycle, catch-and-log,
            // never panic), just on a 60-second interval and spawned
            // unconditionally so it keeps running whichever tab is open
            // (ISC-167). The cycle body itself is genuinely blocking (a
            // real Graph HTTP call plus SQLite), so it runs on the
            // blocking pool rather than stalling an async worker for the
            // length of a network round trip.
            let auto_join_handle = app.handle().clone();
            let auto_join_db_path = data_dir.join("kai-notetaker.sqlite3");
            tauri::async_runtime::spawn(async move {
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(auto_join::POLL_INTERVAL_SECS));
                loop {
                    interval.tick().await;
                    let app_handle = auto_join_handle.clone();
                    let db_path = auto_join_db_path.clone();
                    if let Err(e) = tauri::async_runtime::spawn_blocking(move || {
                        run_auto_join_cycle(&app_handle, &db_path)
                    })
                    .await
                    {
                        eprintln!("auto-join: poll cycle task failed: {e}");
                    }
                }
            });

            // SilenceBasedStopPrompt's monitor — a SEPARATE loop from the
            // auto-join poller above, on a much finer interval (ISC-201).
            // The two never gate each other: ISC-181's calendar-end
            // auto-stop still runs in the poll cycle and still stops
            // silently and directly when a scheduled end passes, while
            // this loop is the independent, lower-confidence layer for
            // calls that end earlier or later than the calendar said
            // (ISC-206).
            //
            // Each tick is cheap and non-blocking (one buffer read, no
            // network, no SQLite), so unlike the auto-join cycle it does
            // not need the blocking pool.
            let silence_handle = app.handle().clone();
            let dialog_open = Arc::new(std::sync::atomic::AtomicBool::new(false));
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                    silence_monitor::CHECK_INTERVAL_SECS,
                ));
                loop {
                    interval.tick().await;
                    let tracker = silence_handle.state::<SilenceTrackerState>();
                    silence_monitor_tick(&silence_handle, &tracker.0, &dialog_open);
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
