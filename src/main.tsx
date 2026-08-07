import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import App from "./App";
import { RecordingBadge } from "./components/RecordingBadge";

/**
 * Both windows load this same bundle (ISC-242) — the overlay is not a second
 * Vite entry point to keep in sync. The window's own label is what decides
 * which component mounts.
 *
 * The branch lives here rather than inside `App` deliberately: `App` gates
 * everything behind `FirstRunSetup` until models are provisioned, and the
 * recording indicator must never be blocked by, or briefly flash, unrelated
 * setup UI.
 */
const RECORDING_BADGE_LABEL = "recording-badge";

const isRecordingBadge = getCurrentWindow().label === RECORDING_BADGE_LABEL;

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{isRecordingBadge ? <RecordingBadge /> : <App />}</React.StrictMode>,
);
