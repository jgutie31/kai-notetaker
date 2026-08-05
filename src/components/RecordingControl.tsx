import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./RecordingControl.css";

interface InputDeviceInfo {
  name: string;
  is_default: boolean;
}

interface StopRecordingResult {
  path: string;
  duration_secs: number;
}

function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

export function RecordingControl() {
  const [devices, setDevices] = useState<InputDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<StopRecordingResult | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    invoke<InputDeviceInfo[]>("list_audio_devices")
      .then((list) => {
        setDevices(list);
        const def = list.find((d) => d.is_default) ?? list[0];
        if (def) setSelectedDevice(def.name);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (isRecording) {
      intervalRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRecording]);

  const handleToggle = useCallback(async () => {
    setError(null);
    setIsBusy(true);
    try {
      if (!isRecording) {
        await invoke("start_recording");
        setElapsed(0);
        setLastSaved(null);
        setIsRecording(true);
      } else {
        const result = await invoke<StopRecordingResult>("stop_recording");
        setIsRecording(false);
        setLastSaved(result);
      }
    } catch (e) {
      setError(String(e));
      setIsRecording(false);
    } finally {
      setIsBusy(false);
    }
  }, [isRecording]);

  const noDevices = devices.length === 0;

  return (
    <div className="recording-screen">
      <div className="recording-screen__device">
        <label htmlFor="device-select">Microphone</label>
        <div className="recording-screen__select-wrap">
          <select
            id="device-select"
            value={selectedDevice}
            onChange={(e) => setSelectedDevice(e.target.value)}
            disabled={isRecording || noDevices}
          >
            {noDevices && <option>No input device found</option>}
            {devices.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
                {d.is_default ? " (default)" : ""}
              </option>
            ))}
          </select>
          <svg
            className="recording-screen__chevron"
            width="10"
            height="6"
            viewBox="0 0 10 6"
            fill="none"
            aria-hidden="true"
          >
            <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      <button
        type="button"
        className={`record-button ${isRecording ? "record-button--recording" : "record-button--idle"}`}
        onClick={handleToggle}
        disabled={isBusy || noDevices}
        aria-pressed={isRecording}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        <span className="record-button__icon" />
      </button>

      <div className="recording-screen__status">
        <span className="recording-screen__label">
          {noDevices ? "No microphone available" : isRecording ? "Recording" : "Ready to record"}
        </span>

        <div className={`recording-screen__timer ${isRecording ? "recording-screen__timer--visible" : ""}`}>
          <span className="recording-screen__dot" />
          <span>{formatElapsed(elapsed)}</span>
        </div>

        {error && <div className="recording-screen__error">{error}</div>}

        {!isRecording && lastSaved && !error && (
          <div className="recording-screen__last-saved">
            Saved · {formatElapsed(lastSaved.duration_secs)}
          </div>
        )}
      </div>
    </div>
  );
}
