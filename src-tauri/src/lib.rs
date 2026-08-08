// `pub` on the modules an examples/ binary needs (MeetingImport: storage,
// pipeline, audit_log, and the four engine modules) — examples compile
// against this crate as an external dependency, so they only see items
// re-exported at this level.
pub mod asr;
mod audio_capture;
pub mod audit_log;
mod auto_join;
mod calendar;
mod call_mute;
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
mod presence;
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
struct RecordingState(Mutex<Option<ActiveRecording>>);

/// Everything that must survive from START to STOP for one recording.
///
/// Was a bare `(RecordingSession, Instant)` tuple; now a named struct
/// because ISC-248 adds a third member whose position in a tuple would
/// mean nothing at the call sites. The trigger source is the reason:
/// a recording's trigger is only knowable at START, but `create_meeting`
/// only runs at STOP, so the value has to ride along here for the
/// recording's whole lifetime rather than being re-derived (guessed)
/// later from whatever state happens to still be around.
struct ActiveRecording {
    session: audio_capture::RecordingSession,
    started_at: Instant,
    trigger_source: storage::TriggerSource,
    /// The real name of the thing being recorded, when it was already
    /// known at START (ISC-260) — in practice a calendar event's subject.
    /// Rides along for exactly the same reason `trigger_source` does: by
    /// STOP time, when `create_meeting` finally runs, the calendar context
    /// that started this recording may be gone from the poller's view.
    known_title: Option<String>,
}

/// Tracks the meeting AutoJoinRecording is currently capturing, if any —
/// distinct from `RecordingState` itself so auto-stop only ever ends a
/// recording *it* started, never a manually-started one that happens to
/// still be running when a poll cycle fires (Jeremiah's real requirement:
/// "make sure there IS an auto-stop when the call ends").
#[derive(Default)]
struct AutoRecordingState(Mutex<Option<AutoRecordingMarker>>);

struct AutoRecordingMarker {
    subject: String,
    /// Which loop owns ending this recording, and on what signal (ISC-233).
    /// A scheduled meeting carries its real end time, parsed once at trigger
    /// time so the stop check never re-fetches or re-parses. An ad-hoc Teams
    /// call has no end time at all — presence itself ends it.
    stop_trigger: auto_join::AutoStopTrigger,
}

/// The four heavy local models, loaded once in a background OS thread at
/// startup (not blocking the window from appearing) and shared across
/// every recording thereafter. `None` until loading finishes.
#[derive(Default, Clone)]
struct EnginesState(Arc<Mutex<Option<Arc<PipelineEngines>>>>);

struct AppPaths {
    data_dir: PathBuf,
}

/// The background thread handle for `spawn_engine_loading`, so the app's
/// exit handler can wait for in-flight model loading to fully finish
/// before letting the process tear down.
///
/// Real crash, confirmed 2026-08-07: quitting shortly after launch — while
/// this thread is still constructing the Metal-backed Whisper/LLM/embedding
/// contexts — races the loading thread's async Metal resource-set init
/// against ggml's own atexit-time global Metal device teardown on the main
/// thread. `ggml_metal_device_free` -> `ggml_metal_rsets_free` -> aborted
/// while `__ggml_metal_rsets_init_block_invoke` was still mid-flight on the
/// loading thread. Nothing in this app's own Rust code was unsound; the
/// race is entirely inside vendored ggml's C++ lifecycle, so the fix is to
/// never let the process's normal exit path run concurrently with it.
#[derive(Default, Clone)]
struct EngineLoadHandle(Arc<Mutex<Option<std::thread::JoinHandle<()>>>>);

/// `true` only while a `spawn_engine_loading` thread is still running.
/// `None` (nothing ever spawned, or a prior load's handle was already
/// taken/joined) and "spawned but finished" both read as `false` — the
/// exit path should only ever pause for the narrow window where Metal
/// init could genuinely still be in flight.
fn engine_load_still_in_progress(load_handle: &Mutex<Option<std::thread::JoinHandle<()>>>) -> bool {
    load_handle.lock().unwrap().as_ref().map(|h| !h.is_finished()).unwrap_or(false)
}

/// Loads the four heavy models from `models_dir` in a background OS
/// thread and populates `engines_state` on success. Shared by app startup
/// (models already present) and by `download_missing_models` (models
/// just finished downloading) — both cases converge on the same
/// "models are on disk now, go load them" moment.
fn spawn_engine_loading(
    models_dir: PathBuf,
    data_dir: PathBuf,
    engines_state: Arc<Mutex<Option<Arc<PipelineEngines>>>>,
    load_handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
) {
    let handle = std::thread::spawn(move || {
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
    *load_handle.lock().unwrap() = Some(handle);
}

#[tauri::command]
fn list_audio_devices() -> Result<Vec<audio_capture::InputDeviceInfo>, String> {
    audio_capture::list_input_devices().map_err(|e| e.to_string())
}

/// The overlay indicator window's label. Referenced in exactly three places:
/// the one `WebviewWindowBuilder::new` call in `.setup()`, the one lookup in
/// `set_recording_badge_visible`, and the frontend's own
/// `getCurrentWindow().label` check (ISC-242) — which is why it's a constant
/// and not three string literals that can drift apart.
const RECORDING_BADGE_LABEL: &str = "recording-badge";

/// The overlay's fixed logical size — small enough to be a status indicator
/// rather than a window, wide enough for "Recording" plus a `MM:SS` timer
/// without truncating. Fixed for v1; draggable/resizable/configurable is
/// explicitly out of scope.
///
/// Widened from 176 to 216 for MicMuteToggle (ISC-275): the badge now also
/// carries a real mute button, and squeezing it into the old width would
/// have pushed the timer into the label.
const RECORDING_BADGE_SIZE: (f64, f64) = (216.0, 44.0);

/// Inset from the screen edge, matching the breathing room macOS's own
/// screen-recording indicator leaves.
const RECORDING_BADGE_MARGIN: f64 = 24.0;

/// Top-right corner placement, in logical pixels, from the primary monitor's
/// logical size (ISC-238).
///
/// Computed rather than hardcoded because a fixed x-coordinate is a real
/// off-screen hazard, not a hypothetical one: any constant tuned for one
/// display puts the badge partly or entirely off the edge of a narrower one,
/// and an indicator you cannot see fails the single job it exists to do.
///
/// `None` — Tauri could not resolve a primary monitor — falls back to a
/// conservative 1280-wide assumption, which lands on-screen for every common
/// display and merely looks off-center on a wide one. Erring toward visible
/// is the whole point.
///
/// Pure and unit-tested for exactly that reason: the arithmetic is trivial,
/// but getting it wrong is silent, and only observable by launching the app
/// on a specific monitor.
fn recording_badge_position(monitor_logical_size: Option<(f64, f64)>) -> (f64, f64) {
    const FALLBACK_WIDTH: f64 = 1280.0;
    let width = match monitor_logical_size {
        Some((w, _)) if w > 0.0 => w,
        _ => FALLBACK_WIDTH,
    };
    let x = (width - RECORDING_BADGE_SIZE.0 - RECORDING_BADGE_MARGIN).max(0.0);
    (x, RECORDING_BADGE_MARGIN)
}

/// Shows or hides the always-on-top recording indicator (ISC-240).
///
/// The ONE place overlay visibility changes. Called from `start_recording`
/// and `stop_recording` — the two functions every trigger already funnels
/// through (manual button, calendar auto-start, silence-prompt stop, and now
/// presence auto-start/stop), per the same "one real call site" precedent as
/// ISC-171/ISC-203. A fifth trigger added later gets the indicator for free
/// and cannot forget to.
///
/// Every failure here is logged and swallowed. Deliberate: this is a status
/// indicator, and a window that won't show must never be the reason a real
/// meeting fails to record. The recording is the product; the badge is the
/// signal about it.
fn set_recording_badge_visible(app: &tauri::AppHandle, visible: bool) {
    let Some(window) = app.get_webview_window(RECORDING_BADGE_LABEL) else {
        eprintln!("recording badge: window '{RECORDING_BADGE_LABEL}' not found — indicator not updated");
        return;
    };
    let result = if visible { window.show() } else { window.hide() };
    if let Err(e) = result {
        let action = if visible { "show" } else { "hide" };
        eprintln!("recording badge: failed to {action} the indicator: {e}");
    }
}

/// `trigger_source` is an optional IPC argument specifically so the
/// frontend's existing `invoke("start_recording")` — which passes no
/// arguments at all — keeps working untouched (ISC-249). Verified against
/// tauri 2.11.5's `CommandItem::deserialize_option`: a key missing from
/// the JSON payload visits `none`, so an omitted argument really does
/// arrive as `None` rather than an "invalid args" rejection.
///
/// `None` therefore means exactly one thing — a human clicked the button.
/// The internal auto-trigger callers always pass their own value and are
/// never allowed to reach this default (ISC-250).
#[tauri::command]
fn start_recording(
    app: tauri::AppHandle,
    state: State<RecordingState>,
    trigger_source: Option<storage::TriggerSource>,
    known_title: Option<String>,
) -> Result<(), String> {
    let trigger_source = trigger_source.unwrap_or(storage::TriggerSource::Manual);
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let recording_id = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();

    let session = audio_capture::RecordingSession::start(&data_dir, &recording_id)
        .map_err(|e| e.to_string())?;

    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    if guard.is_some() {
        return Err("a recording is already in progress".to_string());
    }
    *guard = Some(ActiveRecording { session, started_at: Instant::now(), trigger_source, known_title });

    // Only after the capture is genuinely live and committed to state — the
    // badge must never claim a recording that an early return above
    // prevented (ISC-243).
    set_recording_badge_visible(&app, true);

    // ISC-277's UI half. The backend flag is unconditionally fresh-`false`
    // per session, so this is purely about the windows: the badge overlay
    // is built once at startup and merely shown/hidden, so its React tree
    // never unmounts and would otherwise still be displaying the PREVIOUS
    // recording's muted state the instant the next one appears.
    {
        use tauri::Emitter;
        let _ = app.emit(MIC_MUTE_CHANGED_EVENT, MicMuteChangedPayload { muted: false });
    }
    Ok(())
}

#[tauri::command]
fn switch_recording_device(device_name: String, state: State<RecordingState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let active = guard.as_mut().ok_or("no recording in progress")?;
    active.session.switch_device(&device_name).map_err(|e| e.to_string())
}

#[derive(serde::Serialize)]
struct StopRecordingResult {
    path: String,
    duration_secs: u64,
    meeting_id: i64,
}

/// Elapsed seconds on the live recording, or `None` when nothing is
/// recording (ISC-241).
///
/// Elapsed is computed here rather than handing the frontend a start
/// timestamp to subtract from: the source of truth is the same monotonic
/// `Instant` `stop_recording` bills the real duration against, so the badge
/// can never drift from the saved recording's length, and a wall-clock jump
/// (sleep/wake, NTP correction) can't make the badge count backwards.
#[derive(serde::Serialize)]
struct RecordingStatusPayload {
    elapsed_secs: u64,
    /// Included so the badge's existing 500ms poll doubles as a
    /// self-correcting backstop for the `mic-mute-changed` event
    /// (ISC-273). The event is what makes the UI instant and poll-free;
    /// this field is what makes a dropped event self-heal within half a
    /// second instead of leaving the overlay lying about whether the mic
    /// is live — which, for a mute indicator, is the one failure mode
    /// that actually matters.
    mic_muted: bool,
}

#[tauri::command]
fn recording_status(state: State<RecordingState>) -> Option<RecordingStatusPayload> {
    let guard = state.0.lock().ok()?;
    let active = guard.as_ref()?;
    Some(RecordingStatusPayload {
        elapsed_secs: active.started_at.elapsed().as_secs(),
        mic_muted: active.session.is_mic_muted(),
    })
}

/// The event both windows listen on so neither has to poll for mute state
/// (ISC-273). Emitted app-wide — the main window and the badge overlay are
/// separate webviews, and a toggle originating in either one (or from the
/// global hotkey, which belongs to neither) must be reflected in both.
const MIC_MUTE_CHANGED_EVENT: &str = "mic-mute-changed";

#[derive(Clone, serde::Serialize)]
struct MicMuteChangedPayload {
    muted: bool,
}

/// The ONE place mic-mute state changes (ISC-274).
///
/// Both entry points — the `toggle_mic_mute` command invoked from a UI
/// click, and the global-shortcut callback registered in `run()` — call
/// exactly this function. Deliberately not duplicated: two copies of
/// "read, flip, store, emit" would be two places for the emit to be
/// forgotten, and a hotkey that muted the mic without telling the badge
/// is precisely the silent-state failure this feature exists to avoid.
/// Same "one real call site" precedent as `set_recording_badge_visible`.
///
/// Takes `&AppHandle` rather than a `State<RecordingState>` so the hotkey
/// callback — which is handed an `AppHandle` and nothing else — can call
/// the identical function without the command's IPC-injected arguments.
///
/// Returns the NEW mute state, or a clear error when nothing is recording.
/// Never panics on that path: pressing a global hotkey with no meeting in
/// progress is an ordinary thing to do by accident, not a bug.
fn toggle_mic_mute_inner(app: &tauri::AppHandle) -> Result<bool, String> {
    use tauri::Emitter;

    let new_muted = {
        let state = app.state::<RecordingState>();
        // `as_ref`, never `take` — this is a toggle on a recording that
        // keeps running, not a stop (ISC-276).
        let guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
        let active = guard.as_ref().ok_or("no recording in progress")?;
        let new_muted = !active.session.is_mic_muted();
        active.session.set_mic_muted(new_muted);
        new_muted
        // Guard dropped here, before the emit below: emitting to every
        // webview while still holding the recording lock would let a
        // frontend listener's follow-up command deadlock against it.
    };

    // TeamsCallMuteSync (ISC-281). Placed here, after the lock is
    // dropped, for the same reason the emit below is: shelling out to
    // `pgrep` and talking to CoreGraphics while holding the recording
    // lock would block every other recording-state caller on an
    // unbounded external process.
    //
    // Best-effort by construction — see the function's own contract.
    sync_teams_call_mute();

    if let Err(e) = app.emit(MIC_MUTE_CHANGED_EVENT, MicMuteChangedPayload { muted: new_muted }) {
        // Logged, not fatal, and deliberately NOT rolled back: the audio
        // gate has already changed, which is the part that matters. The
        // UI can still recover on its next `recording_status` poll.
        eprintln!("mic mute: failed to emit {MIC_MUTE_CHANGED_EVENT}: {e}");
    }
    Ok(new_muted)
}

/// Microsoft Teams' executable name, as `pgrep -x` sees it.
///
/// Read off `CFBundleExecutable` in the installed `/Applications/Microsoft
/// Teams.app/Contents/Info.plist`, not guessed. The bundle identifier
/// (`com.microsoft.teams2`) is deliberately NOT used — `pgrep` matches
/// process names, and passing a bundle id would silently never match.
const TEAMS_EXECUTABLE_NAME: &str = "MSTeams";

/// Ensures the Accessibility-permission remedy is printed at most once per
/// app run, not once per toggle. A user who has not granted it will hit
/// this on every single mute press for the whole meeting; repeating the
/// same paragraph dozens of times buries it instead of surfacing it.
static ACCESSIBILITY_WARNING: std::sync::Once = std::sync::Once::new();

/// Mirror the mute toggle onto the real Teams call by posting Teams' own
/// mute shortcut into Teams' process (ISC-281, ISC-282).
///
/// **Returns `()` on purpose.** This is the failure isolation, expressed
/// in the type rather than in a comment asking the caller to be careful:
/// there is no error value for `toggle_mic_mute_inner` to accidentally
/// propagate with `?`, so no future edit can make a Teams-side problem
/// fail the toggle. By the time this runs, `set_mic_muted` has already
/// succeeded — the recording-side gate is the part that must never be
/// held hostage to whether Teams is open or whether an OS permission
/// Jeremiah has not granted yet happens to be in place.
///
/// Teams not running is a silent no-op, not a warning: most recordings
/// have no Teams at all (in-person meetings, testing the button on its
/// own), and logging that as a problem would train the log to be ignored.
///
/// Sends Cmd+Shift+M — Teams' own mute shortcut — which is deliberately a
/// DIFFERENT combo from kai-notetaker's own Cmd+Option+M hotkey, so the
/// keystroke this posts can never be mistaken for, or re-trigger, the
/// hotkey that caused it.
fn sync_teams_call_mute() {
    let Some(pid) = call_mute::find_running_pid(TEAMS_EXECUTABLE_NAME) else {
        // Teams isn't running. Ordinary, not an error.
        return;
    };

    if !call_mute::accessibility_permission_granted() {
        ACCESSIBILITY_WARNING.call_once(|| {
            eprintln!(
                "mic mute: Teams is running, but kai-notetaker does not have Accessibility \
                 permission, so it cannot mute the actual call — only its own recording. \
                 To fix: System Settings -> Privacy & Security -> Accessibility, then enable \
                 kai-notetaker (add it with + if it isn't listed) and restart the app."
            );
        });
        // Deliberately falls through and still attempts the post. The
        // check is a diagnostic, not a gate: AXIsProcessTrusted can read
        // stale right after the user grants permission, and refusing to
        // try would turn a recoverable state into a hard failure.
    }

    if let Err(e) = call_mute::post_key_combo_to_pid(
        pid,
        core_graphics::event::KeyCode::ANSI_M,
        core_graphics::event::CGEventFlags::CGEventFlagCommand
            | core_graphics::event::CGEventFlags::CGEventFlagShift,
    ) {
        // Swallowed by design — see this function's return type.
        eprintln!("mic mute: failed to post mute shortcut to Teams (pid {pid}): {e}");
    }
}

/// Flip kai-notetaker's own mic capture between recording real audio and
/// recording silence, returning the new state (ISC-273).
///
/// As of TeamsCallMuteSync (ISC-281) this ALSO mutes the real Microsoft
/// Teams call when Teams is running — the two are no longer independent,
/// which was the whole point of Phase 2. The Teams half is best-effort:
/// it never blocks, never rolls back, and never fails this command.
///
/// Teams specifically. Other conferencing apps are untouched, so for
/// those this remains a recording-side gate only.
#[tauri::command]
fn toggle_mic_mute(app: tauri::AppHandle) -> Result<bool, String> {
    toggle_mic_mute_inner(&app)
}

#[tauri::command]
fn stop_recording(
    app: tauri::AppHandle,
    state: State<RecordingState>,
    engines_state: State<EnginesState>,
    paths: State<AppPaths>,
    auto_recording_state: State<AutoRecordingState>,
) -> Result<StopRecordingResult, String> {
    let mut guard = state.0.lock().map_err(|_| "recording state lock poisoned")?;
    let ActiveRecording { session, started_at, trigger_source, known_title } =
        guard.take().ok_or("no recording in progress")?;
    let elapsed = started_at.elapsed().as_secs();
    let path = session.stop_and_write().map_err(|e| e.to_string())?;

    // The recording is over the moment the capture is torn down — hide the
    // indicator before the (much longer) pipeline work below, so the badge
    // tracks "is the mic live", not "is the app busy" (ISC-243).
    set_recording_badge_visible(&app, false);

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
    // ISC-248/ISC-260: both the trigger source and the known title are read
    // back out of the recording's own state, never re-derived here — by stop
    // time the calendar/presence context that started this recording may
    // well be gone.
    let meeting_id = storage::create_meeting(
        &conn,
        &path.display().to_string(),
        elapsed,
        trigger_source,
        known_title.as_deref(),
    )
    .map_err(|e| e.to_string())?;

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

/// One command for all three providers — `microsoft`/`google`/`zoom` — since
/// they already share the exact same OAuth engine and UI shape. Forgets only
/// the stored tokens, not the client ID: reconnecting is then just the
/// browser consent flow, not re-pasting an Azure/Google/Zoom client ID.
#[tauri::command]
fn disconnect_provider(provider: String) -> Result<(), String> {
    oauth::delete_tokens(&provider).map_err(|e| e.to_string())
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
    //
    // Marker-variant aware since ISC-233: a `CalendarEnd` marker keeps
    // exactly ISC-181's behavior, while a `PresenceBased` one is skipped
    // outright here — it has no end time, and the presence loop owns ending
    // it. The two stop mechanisms never cross-trigger.
    {
        let due_subject = match app.state::<AutoRecordingState>().0.lock() {
            Ok(guard) => guard
                .as_ref()
                .and_then(|m| m.stop_trigger.calendar_auto_stop_due(chrono::Utc::now()).then(|| m.subject.clone())),
            Err(_) => {
                eprintln!("auto-join: auto-recording marker lock poisoned — skipping auto-stop check this cycle");
                None
            }
        };
        if let Some(subject) = due_subject {
            match stop_recording(
                app.clone(),
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
            // identical downstream pipeline (ISC-171). The one thing that
            // differs is what gets recorded ABOUT the recording: this site
            // states its own trigger explicitly (ISC-250) and never falls
            // through to the Manual default.
            // ISC-260: the event's real subject is already in hand right
            // here and rides along to `create_meeting` at stop time, so the
            // meeting is named the thing it actually is from the moment its
            // row exists — no LLM guess, no waiting for the pipeline.
            if let Err(e) = start_recording(
                app.clone(),
                app.state::<RecordingState>(),
                Some(storage::TriggerSource::Calendar),
                Some(meeting.subject.clone()),
            ) {
                eprintln!("auto-join: failed to start recording for '{}': {e}", meeting.subject);
            } else {
                println!("auto-join: started recording for '{}'", meeting.subject);
                // Record what we started so the auto-stop check above can
                // end THIS recording when the meeting's real end time
                // passes — never a manually-started recording, since only
                // this path ever writes the marker (ISC-181).
                if let Some(end) = auto_join::parse_graph_utc(&meeting.end) {
                    if let Ok(mut marker_guard) = app.state::<AutoRecordingState>().0.lock() {
                        *marker_guard = Some(AutoRecordingMarker {
                            subject: meeting.subject.clone(),
                            stop_trigger: auto_join::AutoStopTrigger::CalendarEnd(end),
                        });
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

/// One TeamsPresenceAdhocRecording poll cycle: the side-effecting half of
/// the feature, exactly the shape `run_auto_join_cycle` established. All
/// three decisions — is Microsoft connected, should this start, should this
/// stop — are pure functions in `presence`/`auto_join`; this function only
/// supplies real inputs and carries out the results.
///
/// Every failure path logs and returns. Nothing panics or propagates: this
/// runs forever on a 15-second timer, and a network blip, an expired refresh
/// token, or consent revoked in the Azure portal must cost one cycle, not
/// take a live recording down with it (ISC-229's resilience requirement,
/// same rule as the calendar poller).
fn run_presence_cycle(app: &tauri::AppHandle) {
    // ISC-230, before anything else and before any network call: a
    // disconnected — or half-connected — Microsoft provider produces zero
    // presence-poll activity. Same predicate the calendar poller uses
    // (`ProviderConnection::is_active`), not a second copy of the rule, so
    // the two can't drift on what "connected" means. A Keychain read that
    // errors counts as not connected: one transient secure-storage hiccup
    // should cost one cycle, not kill the loop.
    let connection = auto_join::ProviderConnection {
        provider_id: calendar::MICROSOFT_PROVIDER_ID,
        client_id: oauth::load_client_id(calendar::MICROSOFT_PROVIDER_ID).unwrap_or(None),
        has_tokens: oauth::load_tokens(calendar::MICROSOFT_PROVIDER_ID).map(|t| t.is_some()).unwrap_or(false),
    };
    if !connection.is_active() {
        return;
    }
    let Some(client_id) = connection.client_id else { return };

    let access_token = match calendar::microsoft_access_token(&client_id) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("presence: could not obtain a valid access token: {e}");
            return;
        }
    };

    let call_state = match presence::get_presence(&access_token) {
        Ok(p) => {
            // Logged verbatim on purpose: this is the exact raw value
            // ISC-227's [DEFERRED-VERIFY] check needs from Jeremiah's real
            // Teams client to confirm what "Meet Now" actually produces.
            // Reading it out of a log beats guessing.
            println!("presence: availability='{}' activity='{}'", p.availability, p.activity);
            p.call_state()
        }
        Err(e) => {
            eprintln!("presence: /me/presence poll failed: {e}");
            return;
        }
    };

    let already_recording = match app.state::<RecordingState>().0.lock() {
        Ok(guard) => guard.is_some(),
        Err(_) => {
            eprintln!("presence: recording state lock poisoned — skipping this cycle");
            return;
        }
    };

    if presence::should_start(call_state, already_recording) {
        // ISC-231: the exact same command function the Start Recording
        // button and the calendar poller call — not a parallel recording
        // path — so an ad-hoc capture produces an identical on-disk artifact
        // and runs the identical downstream pipeline. Explicit trigger
        // (ISC-250) — an ad-hoc capture must never be filed as Manual.
        // ISC-261: `None`, deliberately — an ad-hoc call has no calendar
        // event, and fetching a Graph `onlineMeeting` object to look for a
        // subject would spend a round-trip to retrieve a generic
        // system-generated string. The deterministic "Ad Hoc Call — …"
        // fallback is both cheaper and more honest.
        match start_recording(
            app.clone(),
            app.state::<RecordingState>(),
            Some(storage::TriggerSource::Presence),
            None,
        ) {
            Ok(()) => {
                println!("presence: detected an ad-hoc Teams call — started recording");
                if let Ok(mut marker_guard) = app.state::<AutoRecordingState>().0.lock() {
                    *marker_guard = Some(AutoRecordingMarker {
                        subject: presence::ADHOC_SUBJECT.to_string(),
                        stop_trigger: auto_join::AutoStopTrigger::PresenceBased,
                    });
                }
            }
            Err(e) => eprintln!("presence: failed to start an ad-hoc recording: {e}"),
        }
        return;
    }

    // ISC-234. The marker is read and the decision made under one lock, then
    // released before `stop_recording` — which takes that same lock to clear
    // the marker, and `std::sync::Mutex` is not reentrant.
    let should_stop = match app.state::<AutoRecordingState>().0.lock() {
        Ok(guard) => presence::should_stop(call_state, guard.as_ref().map(|m| &m.stop_trigger)),
        Err(_) => {
            eprintln!("presence: auto-recording marker lock poisoned — skipping the stop check this cycle");
            return;
        }
    };
    if should_stop {
        match stop_recording(
            app.clone(),
            app.state::<RecordingState>(),
            app.state::<EnginesState>(),
            app.state::<AppPaths>(),
            app.state::<AutoRecordingState>(),
        ) {
            Ok(result) => println!("presence: the ad-hoc call ended — auto-stopped, meeting_id={}", result.meeting_id),
            Err(e) => eprintln!("presence: failed to auto-stop the ad-hoc recording: {e}"),
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
                .map(|active| active.session.trailing_rms(silence_monitor::SILENCE_WINDOW_SECS)),
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
                    app_for_callback.clone(),
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
fn download_missing_models(
    app: tauri::AppHandle,
    paths: State<AppPaths>,
    engines: State<EnginesState>,
    load_handle: State<EngineLoadHandle>,
) {
    use tauri::Emitter;

    let models_dir = paths.data_dir.join("models");
    let data_dir = paths.data_dir.clone();
    let engines_state = engines.0.clone();
    let load_handle = load_handle.0.clone();
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
        spawn_engine_loading(models_dir, data_dir, engines_state, load_handle);
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

// ---------------------------------------------------------------------
// Startup orphan-recording recovery (ISC-217 … ISC-221)
//
// RecordingDurability makes a crashed recording's WAV file *survive* on
// disk. That file is still useless until something notices it and runs
// the pipeline over it — which is what this scan does, exactly once per
// launch. It is safe to run at startup precisely because the app is
// single-instance: at the moment `setup()` runs, no `RecordingSession`
// can exist, so any auto-generated-pattern `.wav` without a `meetings`
// row is necessarily a leftover from a previous, dead process.
// ---------------------------------------------------------------------

/// True only for filenames the app itself generates in `start_recording`
/// (`chrono::Utc::now().format("%Y%m%dT%H%M%S")` + `.wav`) — i.e. the
/// regex `^\d{8}T\d{6}\.wav$`, hand-rolled to avoid pulling in the
/// `regex` crate for one fixed-width pattern.
///
/// Anti (ISC-218): this is also the guard that keeps `imported-*` files
/// — owned exclusively by the `import_legacy_recordings` example binary
/// — out of the scan. They cannot match a fixed 8-digit/T/6-digit shape.
fn is_auto_generated_recording_name(name: &str) -> bool {
    // 8 digits + 'T' + 6 digits + ".wav"
    if name.len() != 8 + 1 + 6 + 4 {
        return false;
    }
    let (stem, ext) = name.split_at(15);
    if ext != ".wav" {
        return false;
    }
    let bytes = stem.as_bytes();
    bytes[8] == b'T'
        && bytes[..8].iter().all(u8::is_ascii_digit)
        && bytes[9..].iter().all(u8::is_ascii_digit)
}

/// Every `.wav` in `recordings_dir` that the app generated itself and
/// that has no matching `meetings.audio_path` row. Sorted, so recovery
/// order is deterministic (oldest recording first, since the filename is
/// itself a sortable timestamp).
fn find_orphaned_recordings(
    recordings_dir: &std::path::Path,
    known_audio_paths: &std::collections::HashSet<String>,
) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(recordings_dir) {
        Ok(e) => e,
        // No recordings dir yet (first ever launch) is not an error —
        // there is simply nothing to recover.
        Err(_) => return Vec::new(),
    };

    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_auto_generated_recording_name)
        })
        .filter(|path| !known_audio_paths.contains(&path.display().to_string()))
        .collect();
    out.sort();
    out
}

/// Real duration of a WAV, read from its own header — never guessed,
/// never hardcoded (ISC-220). `duration()` is frames-per-channel, so
/// dividing by the declared sample rate gives real wall-clock seconds
/// regardless of channel count.
fn wav_duration_secs(path: &std::path::Path) -> Result<u64, hound::Error> {
    let reader = hound::WavReader::open(path)?;
    let rate = reader.spec().sample_rate.max(1) as u64;
    Ok(reader.duration() as u64 / rate)
}

/// Creates the `meetings` row for one orphaned recording and returns its
/// id. Split out from the pipeline kickoff so the DB half is testable
/// without loading the four heavy engines.
///
/// Returns `Ok(None)` for a file whose header reports zero seconds of
/// audio — a recording killed inside its very first checkpoint window
/// (the honest bound named on `FLUSH_INTERVAL_SECS`). There is nothing
/// to transcribe, and creating a row for it would only produce a
/// permanently-failed meeting in the library.
fn recover_orphan_into_db(
    conn: &rusqlite::Connection,
    path: &std::path::Path,
) -> Result<Option<i64>, String> {
    let duration_secs = wav_duration_secs(path).map_err(|e| format!("unreadable WAV {}: {e}", path.display()))?;
    if duration_secs == 0 {
        return Ok(None);
    }
    // ISC-251: hardcoded `Recovered`, not derived from anything. Whatever
    // originally triggered this recording died with the process — saying
    // "recovered" is the only claim we can actually stand behind.
    // `known_title: None` for the same reason the trigger source is
    // hardcoded `Recovered` — whatever this recording was called died with
    // the process. The deterministic "Recovered Recording — …" fallback is
    // the only name we can stand behind.
    let meeting_id = storage::create_meeting(
        conn,
        &path.display().to_string(),
        duration_secs,
        storage::TriggerSource::Recovered,
        None,
    )
    .map_err(|e| e.to_string())?;
    Ok(Some(meeting_id))
}

/// The full startup pass: find orphans, create their meetings, and hand
/// each one to the *same* `pipeline::process_meeting` background-thread
/// call `stop_recording` uses (ISC-221) — no parallel recovery pipeline.
fn recover_orphaned_recordings(
    data_dir: &std::path::Path,
    engines_state: Arc<Mutex<Option<Arc<PipelineEngines>>>>,
) {
    let db_path = data_dir.join("kai-notetaker.sqlite3");
    let audit_path = data_dir.join("audit-log.jsonl");
    let recordings_dir = data_dir.join("recordings");

    let conn = match storage::open_connection(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("orphan recovery: failed to open db: {e}");
            return;
        }
    };
    let known = match storage::all_audio_paths(&conn) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("orphan recovery: failed to read existing audio paths: {e}");
            return;
        }
    };

    for path in find_orphaned_recordings(&recordings_dir, &known) {
        let meeting_id = match recover_orphan_into_db(&conn, &path) {
            Ok(Some(id)) => id,
            Ok(None) => {
                eprintln!(
                    "orphan recovery: skipping {} — zero-duration WAV (crashed before its first checkpoint)",
                    path.display()
                );
                continue;
            }
            Err(e) => {
                eprintln!("orphan recovery: {e}");
                continue;
            }
        };
        println!(
            "orphan recovery: recovered crashed recording {} as meeting {meeting_id}",
            path.display()
        );

        // Identical shape to stop_recording's kickoff: own OS thread,
        // wait for engines, own DB connection, same audit log, same
        // process_meeting call.
        let engines_handle = engines_state.clone();
        let db_path = db_path.clone();
        let audit_path = audit_path.clone();
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
                    eprintln!("orphan recovery: failed to open db for meeting {meeting_id}: {e}");
                    return;
                }
            };
            let audit = AuditLog::new(&audit_path);
            if let Err(e) = pipeline::process_meeting(&conn, &audit, &engines, meeting_id, &path, None) {
                eprintln!("orphan recovery: pipeline failed for meeting {meeting_id}: {e}");
            }
        });
    }
}

/// The system-wide mic-mute hotkey (ISC-274): **Cmd+Option+M** on macOS
/// (Super+Alt+M elsewhere).
///
/// Deliberately NOT Cmd+Shift+M, which is Microsoft Teams' own mute
/// shortcut. Two different mutes on one keystroke would be genuinely
/// ambiguous — the user could never tell whether they had silenced the
/// call, the recording, or both — and Phase 2 (driving the call app's mute
/// from this same hotkey) is explicitly a separate, later decision.
///
/// Built through a function rather than a `const` because `Shortcut::new`
/// is not `const fn`; both the registration in `run()`'s setup and the
/// handler's equality check call this, so they cannot drift apart.
#[cfg(desktop)]
fn mic_mute_shortcut() -> tauri_plugin_global_shortcut::Shortcut {
    use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};
    // `Modifiers::META` is normalized to `SUPER` inside `Shortcut::new`,
    // which is Command on macOS — confirmed against global-hotkey's own
    // source, not assumed.
    Shortcut::new(Some(Modifiers::META | Modifiers::ALT), Code::KeyM)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(desktop)]
    let global_shortcut_plugin = {
        use tauri_plugin_global_shortcut::ShortcutState;

        let watched = mic_mute_shortcut();
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(move |app, shortcut, event| {
                // The handler is invoked for EVERY registered shortcut, so
                // the identity check is required, not defensive noise.
                if shortcut != &watched {
                    return;
                }
                // Press only. Without this the toggle would fire twice per
                // keystroke — once down, once up — netting out to no
                // change at all, which is the classic global-shortcut bug.
                if event.state() != ShortcutState::Pressed {
                    return;
                }
                // The SAME function the `toggle_mic_mute` command calls
                // (ISC-274) — no duplicated toggle logic, so the emit can
                // never be forgotten on one path.
                match toggle_mic_mute_inner(app) {
                    Ok(muted) => println!(
                        "mic mute: global hotkey toggled capture {}",
                        if muted { "MUTED" } else { "LIVE" }
                    ),
                    // The overwhelmingly common case is "no recording is
                    // running" — an ordinary accidental keypress, not a
                    // fault worth surfacing to the user.
                    Err(e) => eprintln!("mic mute: global hotkey ignored: {e}"),
                }
            })
            .build()
    };

    let builder = tauri::Builder::default();
    #[cfg(desktop)]
    let builder = builder.plugin(global_shortcut_plugin);

    builder
        .plugin(tauri_plugin_opener::init())
        // Native OS message dialogs, for the silence-based Stop/Continue
        // prompt (ISC-202). Registered exactly as the plugin's own v2 docs
        // specify.
        .plugin(tauri_plugin_dialog::init())
        .manage(RecordingState::default())
        .manage(EnginesState::default())
        .manage(EngineLoadHandle::default())
        .manage(AutoRecordingState::default())
        .manage(SilenceTrackerState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices,
            start_recording,
            switch_recording_device,
            stop_recording,
            recording_status,
            toggle_mic_mute,
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
            disconnect_provider,
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

            // The recording indicator overlay (ISC-238/ISC-239): built ONCE,
            // here, hidden — never re-created per recording. `.show()` /
            // `.hide()` on this same handle is all any later state
            // transition does, which avoids window-creation cost and the
            // visible flicker of a fresh window on every start.
            //
            // It loads the same `index.html` bundle as the main window and
            // branches on its own label frontend-side (ISC-242), so there's
            // no second Vite entry point to keep in sync.
            //
            // Verified against the installed tauri 2.11.5 source, not
            // assumed: every method below exists on `WebviewWindowBuilder`
            // at this version, including `.visible(false)`, so the window
            // never flashes on screen before being hidden.
            //
            // A failure to build is logged, not fatal: the app must still
            // record without its status badge.
            let (badge_x, badge_y) = recording_badge_position(
                app.primary_monitor().ok().flatten().map(|m| {
                    let size = m.size().to_logical::<f64>(m.scale_factor());
                    (size.width, size.height)
                }),
            );
            match tauri::WebviewWindowBuilder::new(
                app,
                RECORDING_BADGE_LABEL,
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("Recording")
            .inner_size(RECORDING_BADGE_SIZE.0, RECORDING_BADGE_SIZE.1)
            .position(badge_x, badge_y)
            .always_on_top(true)
            .decorations(false)
            .resizable(false)
            .visible(false)
            // Keep it out of the taskbar: it's an indicator, not a window
            // anyone should try to focus. Documented as a no-op on macOS in
            // tauri 2.11.5's own source — kept anyway because it's the right
            // declaration for the Windows build, which is deliberately
            // untouched for now but will inherit this.
            .skip_taskbar(true)
            // The badge draws its own rounded border in CSS; an OS drop
            // shadow around an undecorated 176x44 window reads as a glitch.
            .shadow(false)
            .build()
            {
                Ok(_) => println!("recording badge: overlay window created (hidden)"),
                Err(e) => eprintln!("recording badge: failed to create the overlay window: {e}"),
            }

            // MicMuteToggle (ISC-274): bind the system-wide hotkey. Done
            // here rather than via the plugin builder's `with_shortcut`
            // because that returns a `Result` mid-chain, and a keyboard
            // shortcut the OS refuses to grant (another app already owns
            // the combination, or Accessibility permission is missing)
            // must never be the reason this app fails to launch — the
            // in-app mute button is unaffected either way.
            #[cfg(desktop)]
            {
                use tauri_plugin_global_shortcut::GlobalShortcutExt;
                let shortcut = mic_mute_shortcut();
                match app.global_shortcut().register(shortcut) {
                    Ok(()) => println!("mic mute: global hotkey registered (Cmd+Option+M)"),
                    Err(e) => eprintln!(
                        "mic mute: could not register the global hotkey ({e}) — \
                         the badge's mute button still works"
                    ),
                }
            }

            // Load the four heavy models in a background OS thread so the
            // window appears immediately rather than stalling on multi-
            // second model loads.
            let engines_state = app.state::<EnginesState>().0.clone();
            let load_handle = app.state::<EngineLoadHandle>().0.clone();
            let models_dir = model_provisioning::resolve_models_dir(&data_dir);
            spawn_engine_loading(models_dir, data_dir.clone(), engines_state.clone(), load_handle);

            // One-time (not interval) scan for recordings a previous
            // process died holding — a dev-mode rebuild, a crash, a
            // force-quit. Runs AFTER ensure_schema above so the
            // `meetings` table is guaranteed to exist, and before any
            // recording can start, so it can never race a live session
            // (ISC-217). Every filesystem/DB failure inside is logged
            // and skipped, never panicked — a weird leftover file must
            // not stop the app from launching.
            recover_orphaned_recordings(&data_dir, engines_state);

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

            // TeamsPresenceAdhocRecording's poller (ISC-229) — a THIRD,
            // separate loop, distinct from both the 60s calendar poller
            // above and the silence monitor below. Deliberately not folded
            // into the calendar cycle: it runs on a shorter interval (ad-hoc
            // start latency is felt directly, see
            // PRESENCE_POLL_INTERVAL_SECS), it must keep polling regardless
            // of the auto-join enabled toggle's calendar-specific gating,
            // and its cycle is a single cheap Graph GET with no SQLite at
            // all.
            //
            // Like the calendar cycle it IS genuinely blocking (real HTTP
            // plus Keychain reads), so it runs on the blocking pool rather
            // than stalling an async worker for a network round trip.
            let presence_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                    presence::PRESENCE_POLL_INTERVAL_SECS,
                ));
                loop {
                    interval.tick().await;
                    let app_handle = presence_handle.clone();
                    if let Err(e) =
                        tauri::async_runtime::spawn_blocking(move || run_presence_cycle(&app_handle)).await
                    {
                        eprintln!("presence: poll cycle task failed: {e}");
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
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // ISC (crash fix, 2026-08-07): never let the process's normal
            // exit proceed while model loading is still in flight — see
            // `EngineLoadHandle`'s doc comment for the exact race this
            // closes. The common case (loading already finished) adds a
            // single non-blocking `is_finished()` check and nothing else.
            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                let load_handle = app_handle.state::<EngineLoadHandle>().0.clone();
                if engine_load_still_in_progress(&load_handle) {
                    api.prevent_exit();
                    let app_handle = app_handle.clone();
                    std::thread::spawn(move || {
                        if let Some(handle) = load_handle.lock().unwrap().take() {
                            let _ = handle.join();
                        }
                        // By now the loading thread (and any Metal init it
                        // kicked off) has fully finished, so this second
                        // exit is safe — `still_loading` will be false and
                        // the default exit path proceeds normally.
                        app_handle.exit(0);
                    });
                }
            }
        });
}

#[cfg(test)]
mod recording_badge_tests {
    use super::*;

    /// ISC-238: the badge lands in the top-right, fully on screen, across
    /// the real display sizes Jeremiah actually uses — the built-in laptop
    /// panel and an external monitor.
    #[test]
    fn the_badge_sits_fully_on_screen_in_the_top_right_of_any_real_display() {
        for (w, h) in [(1280.0, 800.0), (1440.0, 900.0), (1512.0, 982.0), (1920.0, 1080.0), (3440.0, 1440.0)] {
            let (x, y) = recording_badge_position(Some((w, h)));
            assert!(x >= 0.0, "{w}x{h}: never off the left edge");
            assert!(
                x + RECORDING_BADGE_SIZE.0 <= w,
                "{w}x{h}: the badge's right edge ({}) must stay on screen",
                x + RECORDING_BADGE_SIZE.0
            );
            assert_eq!(y, RECORDING_BADGE_MARGIN, "{w}x{h}: inset from the top, not flush");
            // Genuinely top-RIGHT, not merely on-screen: it must sit in the
            // right half of the display.
            assert!(x > w / 2.0, "{w}x{h}: must be a right-corner badge");
        }
    }

    /// The fallback path — Tauri could not resolve a primary monitor. Must
    /// still produce a position that is on-screen for every common display,
    /// because an invisible indicator is the one failure this feature cannot
    /// tolerate.
    #[test]
    fn an_unresolvable_monitor_falls_back_to_a_position_visible_on_any_real_display() {
        let (x, y) = recording_badge_position(None);
        assert_eq!(y, RECORDING_BADGE_MARGIN);
        assert!(x > 0.0);
        // The narrowest display this could realistically land on still shows
        // the whole badge.
        assert!(x + RECORDING_BADGE_SIZE.0 <= 1280.0, "must fit a 1280-wide screen");
    }

    /// A degenerate monitor size (a real value Tauri can report on some
    /// headless/virtual displays) must not produce a negative coordinate.
    #[test]
    fn a_degenerate_monitor_size_never_produces_an_offscreen_negative_position() {
        for size in [Some((0.0, 0.0)), Some((-1.0, -1.0)), Some((100.0, 100.0))] {
            let (x, y) = recording_badge_position(size);
            assert!(x >= 0.0, "{size:?} produced a negative x");
            assert!(y >= 0.0);
        }
    }
}

#[cfg(test)]
mod engine_load_handle_tests {
    use super::*;
    use std::sync::mpsc;

    /// Steady state before any load has ever run, and after a prior
    /// load's handle was already taken/joined — both must read as "not
    /// in progress," or the exit handler would wait forever on nothing.
    #[test]
    fn nothing_ever_spawned_is_not_in_progress() {
        let handle: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);
        assert!(!engine_load_still_in_progress(&handle));
    }

    /// The exact race this fix closes: while the loading thread is still
    /// running, the check must say so — this is what makes the exit
    /// handler call `prevent_exit()` instead of letting ggml's atexit
    /// teardown run concurrently with in-flight Metal init.
    #[test]
    fn a_running_load_thread_is_in_progress() {
        let (tx, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            rx.recv().ok();
        });
        let slot = Mutex::new(Some(handle));
        assert!(engine_load_still_in_progress(&slot));

        // Release it and let the check catch up — proves this isn't
        // hardcoded true, it's actually reading real thread state.
        tx.send(()).unwrap();
        if let Some(h) = slot.lock().unwrap().take() {
            h.join().unwrap();
        }
        assert!(!engine_load_still_in_progress(&slot));
    }

    /// A handle that finished on its own (no exit ever raced it) must
    /// read as "not in progress" without needing an explicit `.take()` —
    /// the exit handler only actually joins when it found the race.
    #[test]
    fn a_finished_load_thread_is_not_in_progress() {
        let handle = std::thread::spawn(|| {});
        handle.join().unwrap();
        // is_finished() on an already-joined handle would panic on most
        // std APIs (JoinHandle is consumed by join), so this test spawns
        // a second one and waits for it to naturally finish instead —
        // the real-world case is "loading completed before quit was ever
        // requested," not "already joined by someone else."
        let (tx, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
        // Give the thread a moment to actually mark itself finished after
        // the send — is_finished() reflects OS thread completion, not the
        // channel send itself.
        for _ in 0..1000 {
            if handle.is_finished() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let slot = Mutex::new(Some(handle));
        assert!(!engine_load_still_in_progress(&slot));
    }
}

#[cfg(test)]
mod orphan_recovery_tests {
    use super::*;
    use std::collections::HashSet;

    /// A real, decodable WAV of a known duration, written the same way
    /// (16-bit mono @ CANONICAL_SAMPLE_RATE) a real recording is — so
    /// duration assertions are against genuine header math, not a stub.
    fn write_fixture_wav(path: &std::path::Path, secs: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..(48_000 * secs) {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kai-notetaker-orphan-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// ISC-217: the pattern is exactly the one `start_recording`
    /// generates — nothing wider, nothing narrower.
    #[test]
    fn only_the_apps_own_auto_generated_filename_pattern_is_recognized() {
        assert!(is_auto_generated_recording_name("20260806T165724.wav"));
        assert!(is_auto_generated_recording_name("19991231T000000.wav"));

        // ISC-218: the import tool's files, in every shape it produces.
        assert!(!is_auto_generated_recording_name("imported-20260806T165724.wav"));
        assert!(!is_auto_generated_recording_name("imported-first-meeting.wav"));

        // Near-misses that must not slip through.
        assert!(!is_auto_generated_recording_name("20260806T16572.wav"), "5-digit time");
        assert!(!is_auto_generated_recording_name("20260806T1657244.wav"), "7-digit time");
        assert!(!is_auto_generated_recording_name("20260806X165724.wav"), "wrong separator");
        assert!(!is_auto_generated_recording_name("2026080AT165724.wav"), "non-digit date");
        assert!(!is_auto_generated_recording_name("20260806T16572A.wav"), "non-digit time");
        assert!(!is_auto_generated_recording_name("20260806T165724.WAV"), "case-sensitive extension");
        assert!(!is_auto_generated_recording_name("20260806T165724.wav.bak"));
        assert!(!is_auto_generated_recording_name("test-recording.wav"));
        assert!(!is_auto_generated_recording_name(""));
    }

    /// ISC-217 + ISC-218 + ISC-219 in one realistic fixture: a mixed
    /// `recordings/` directory containing an orphan, an already-known
    /// recording, an `imported-*` file, and unrelated junk. Exactly the
    /// orphan comes back.
    #[test]
    fn scan_selects_only_the_orphaned_auto_generated_recording() {
        let dir = temp_dir("scan");
        let recordings = dir.join("recordings");
        std::fs::create_dir_all(&recordings).unwrap();

        let orphan = recordings.join("20260806T165724.wav");
        let already_known = recordings.join("20260805T090000.wav");
        let imported = recordings.join("imported-20260101T120000.wav");
        let junk = recordings.join("notes.txt");
        write_fixture_wav(&orphan, 1);
        write_fixture_wav(&already_known, 1);
        write_fixture_wav(&imported, 1);
        std::fs::write(&junk, b"not audio").unwrap();

        let known: HashSet<String> = [already_known.display().to_string()].into_iter().collect();
        let found = find_orphaned_recordings(&recordings, &known);

        assert_eq!(found, vec![orphan.clone()], "expected exactly the orphan, got {found:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// ISC-219 (anti): a file with a matching `meetings.audio_path` row
    /// is never re-recovered — proven against a real SQLite DB and
    /// re-run to simulate repeated app restarts.
    #[test]
    fn a_recording_with_a_matching_meetings_row_is_never_re_recovered() {
        let dir = temp_dir("norepeat");
        let recordings = dir.join("recordings");
        std::fs::create_dir_all(&recordings).unwrap();

        let existing = recordings.join("20260806T101010.wav");
        write_fixture_wav(&existing, 2);

        let conn = rusqlite::Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();
        storage::create_meeting(
            &conn,
            &existing.display().to_string(),
            2,
            storage::TriggerSource::Manual,
            None,
        )
        .unwrap();

        // Three "restarts" — each one must find nothing.
        for restart in 1..=3 {
            let known = storage::all_audio_paths(&conn).unwrap();
            let found = find_orphaned_recordings(&recordings, &known);
            assert!(found.is_empty(), "restart {restart} wrongly re-recovered {found:?}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// ISC-218 (anti), end to end against a real DB: an `imported-*`
    /// file with NO `meetings` row — the exact situation that would tempt
    /// a naive scan — is still left completely alone, while the orphan
    /// beside it is recovered.
    #[test]
    fn imported_prefixed_files_are_never_recovered_even_with_no_meetings_row() {
        let dir = temp_dir("imported");
        let recordings = dir.join("recordings");
        std::fs::create_dir_all(&recordings).unwrap();

        let orphan = recordings.join("20260806T235959.wav");
        let imported = recordings.join("imported-legacy-call.wav");
        write_fixture_wav(&orphan, 3);
        write_fixture_wav(&imported, 3);

        let conn = rusqlite::Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();
        // Deliberately empty: neither file has a row.
        let known = storage::all_audio_paths(&conn).unwrap();
        assert!(known.is_empty());

        let found = find_orphaned_recordings(&recordings, &known);
        assert_eq!(found, vec![orphan], "the imported-* file must never be touched");

        let recovered: Vec<i64> = found
            .iter()
            .filter_map(|p| recover_orphan_into_db(&conn, p).unwrap())
            .collect();
        assert_eq!(recovered.len(), 1);

        // And the imported file still has no row afterwards.
        let after = storage::all_audio_paths(&conn).unwrap();
        assert!(!after.contains(&imported.display().to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// ISC-220: the recovered meeting's `duration_secs` is the WAV's real
    /// header duration, not a guess and not zero.
    #[test]
    fn recovered_meeting_duration_matches_the_real_wav_header_duration() {
        let dir = temp_dir("duration");
        let recordings = dir.join("recordings");
        std::fs::create_dir_all(&recordings).unwrap();

        let orphan = recordings.join("20260806T121212.wav");
        write_fixture_wav(&orphan, 7); // known real duration

        assert_eq!(wav_duration_secs(&orphan).unwrap(), 7);

        let conn = rusqlite::Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();
        let meeting_id = recover_orphan_into_db(&conn, &orphan).unwrap().expect("a 7s WAV is recoverable");

        let detail = storage::get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.duration_secs, 7, "duration must come from the real WAV header");
        assert_eq!(detail.audio_path.as_deref(), Some(orphan.display().to_string().as_str()));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The honest bound from `FLUSH_INTERVAL_SECS`, made explicit: a
    /// recording killed before its first checkpoint has a valid but
    /// empty WAV. Recovery skips it rather than creating a meeting that
    /// could only ever fail in the pipeline.
    #[test]
    fn a_zero_duration_orphan_is_skipped_not_turned_into_a_broken_meeting() {
        let dir = temp_dir("empty");
        let recordings = dir.join("recordings");
        std::fs::create_dir_all(&recordings).unwrap();

        let empty = recordings.join("20260806T000001.wav");
        write_fixture_wav(&empty, 0);

        let conn = rusqlite::Connection::open_in_memory().unwrap();
        storage::ensure_schema(&conn).unwrap();

        // It IS identified as an orphan by the scan...
        let known = storage::all_audio_paths(&conn).unwrap();
        assert_eq!(find_orphaned_recordings(&recordings, &known), vec![empty.clone()]);
        // ...but recovery declines to create a row for it.
        assert_eq!(recover_orphan_into_db(&conn, &empty).unwrap(), None);
        assert!(storage::all_audio_paths(&conn).unwrap().is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A missing `recordings/` directory (genuine first launch) is not an
    /// error — the app must still start.
    #[test]
    fn a_missing_recordings_directory_yields_no_orphans_and_no_panic() {
        let dir = temp_dir("missing");
        let never_created = dir.join("recordings");
        assert!(find_orphaned_recordings(&never_created, &HashSet::new()).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
