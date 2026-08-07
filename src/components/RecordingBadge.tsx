import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { formatElapsed } from "./RecordingControl";
import { MicMuteIcon } from "./MicMuteIcon";
import { useMicMute } from "./useMicMute";
import "./RecordingBadge.css";

/**
 * The always-on-top recording indicator (ISC-241).
 *
 * Jeremiah's explicit condition for accepting silent auto-start: he must
 * never be unaware that a recording is running. That makes this component's
 * job narrow and non-negotiable — say "recording", and say for how long.
 *
 * Rendered into its own small Tauri window (label `recording-badge`), whose
 * visibility Rust controls entirely from `start_recording`/`stop_recording`.
 * This component never shows or hides itself; if it's on screen at all, a
 * recording is live.
 */
interface RecordingStatus {
  elapsed_secs: number;
  /** ISC-273: the backend's authoritative mute state, so the poll the
   *  badge already runs doubles as a self-heal if a `mic-mute-changed`
   *  event is ever missed. */
  mic_muted: boolean;
}

/**
 * Faster than the 1s tick the main window uses. The window is only visible
 * while recording, so the cost is bounded, and it means the timer is correct
 * the instant the badge appears rather than up to a second stale.
 */
const POLL_MS = 500;

export function RecordingBadge() {
  const [elapsed, setElapsed] = useState<number | null>(null);
  const { muted, setMuted, toggle } = useMicMute();

  // Both windows load the same CSS bundle (see main.tsx), so a bare
  // `body { background: transparent }` in RecordingBadge.css would leak
  // into the main window's own document too, overriding the real theme —
  // a real bug this exact scoping caught and fixed. Marking only THIS
  // window's own <html> confines the override to a document the main
  // window's React tree never touches (each Tauri window is a fully
  // separate document, not a shared DOM).
  useEffect(() => {
    document.documentElement.classList.add("recording-badge-window");
    return () => document.documentElement.classList.remove("recording-badge-window");
  }, []);

  useEffect(() => {
    let cancelled = false;

    // Elapsed comes from Rust every tick rather than being counted locally
    // off a single start timestamp: the backend's monotonic `Instant` is the
    // same clock the saved recording's duration is billed against, so the
    // badge can't drift from it across a sleep/wake or an NTP correction.
    const poll = () => {
      invoke<RecordingStatus | null>("recording_status")
        .then((status) => {
          if (cancelled) return;
          setElapsed(status ? status.elapsed_secs : null);
          if (status) setMuted(status.mic_muted);
        })
        .catch(() => {
          // A failed status read must not blank out the indicator — the
          // recording is almost certainly still running, and a badge that
          // silently disappears is the exact failure mode this feature
          // exists to prevent. Keep showing the last known elapsed.
        });
    };

    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [setMuted]);

  return (
    <div className={`recording-badge ${muted ? "recording-badge--muted" : ""}`}>
      <span className="recording-badge__dot" aria-hidden="true" />
      {/* Muted is a different WORD, not just a different colour — the badge's
          whole job is to be unmistakable at a glance (ISC-275). */}
      <span className="recording-badge__label">{muted ? "Muted" : "Recording"}</span>
      <span className="recording-badge__timer">{formatElapsed(elapsed ?? 0)}</span>
      <button
        type="button"
        className="recording-badge__mute"
        onClick={toggle}
        aria-pressed={muted}
        aria-label={muted ? "Unmute microphone capture" : "Mute microphone capture"}
        title={`${muted ? "Unmute" : "Mute"} this recording (⌘⌥M)`}
      >
        <MicMuteIcon muted={muted} />
      </button>
    </div>
  );
}
