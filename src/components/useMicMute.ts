import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/**
 * Shared mic-mute state for MicMuteToggle (ISC-273/ISC-275).
 *
 * Both windows that can show or change mute — the always-on-top badge
 * overlay and the main window's recording screen — use this one hook, so
 * they cannot disagree about what "muted" currently means.
 *
 * The `mic-mute-changed` event is what makes that work without polling.
 * There are three independent ways mute can change (a click in the main
 * window, a click on the badge, and the system-wide Cmd+Option+M hotkey
 * that belongs to neither window), so no window can treat its own click as
 * the source of truth. Rust emits app-wide on every change; both windows
 * just listen.
 */
export interface MicMuteChangedPayload {
  muted: boolean;
}

export function useMicMute() {
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // `listen` resolves to an unlisten function asynchronously, so the
    // cleanup has to await it rather than assume it already exists —
    // otherwise a fast unmount leaks a live listener.
    const pending = listen<MicMuteChangedPayload>("mic-mute-changed", (event) => {
      if (!cancelled) setMuted(event.payload.muted);
    });
    return () => {
      cancelled = true;
      pending.then((unlisten) => unlisten()).catch(() => {});
    };
  }, []);

  const toggle = useCallback(async () => {
    setError(null);
    try {
      // The command returns the authoritative new state, but the emitted
      // event will deliver it too. Setting it here as well makes the click
      // feel instant rather than waiting on an IPC round trip back.
      const next = await invoke<boolean>("toggle_mic_mute");
      setMuted(next);
    } catch (e) {
      // The only real error is "no recording in progress" — the toggle is
      // only ever reachable while recording, so this means state drifted.
      // Surface it rather than leaving a dead-feeling button.
      setError(String(e));
    }
  }, []);

  return { muted, setMuted, toggle, error };
}
