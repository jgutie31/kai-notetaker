import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./FirstRunSetup.css";

interface DownloadProgress {
  model: string;
  downloaded: number;
  total: number;
}

interface DownloadError {
  model: string;
  error: string;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  return `${(bytes / 1_000).toFixed(0)} KB`;
}

interface FirstRunSetupProps {
  onReady: () => void;
}

// Checks whether any local AI models are missing on first launch and, if
// so, downloads them with a real progress screen instead of requiring a
// manual curl per model. Models are downloaded once, ever — most launches
// find nothing missing and this component renders nothing.
export function FirstRunSetup({ onReady }: FirstRunSetupProps) {
  const [checking, setChecking] = useState(true);
  const [missing, setMissing] = useState<string[]>([]);
  const [progressByModel, setProgressByModel] = useState<Record<string, DownloadProgress>>({});
  const [error, setError] = useState<string | null>(null);
  const downloadStarted = useRef(false);

  useEffect(() => {
    invoke<string[]>("check_missing_models")
      .then((result) => {
        setMissing(result);
        setChecking(false);
        if (result.length === 0) onReady();
      })
      .catch((e) => {
        setError(String(e));
        setChecking(false);
      });
  }, [onReady]);

  useEffect(() => {
    if (checking || missing.length === 0 || downloadStarted.current) return;
    downloadStarted.current = true;

    const unlistenPromises = [
      listen<DownloadProgress>("model-download-progress", (event) => {
        setProgressByModel((prev) => ({ ...prev, [event.payload.model]: event.payload }));
      }),
      listen("model-download-complete", () => {
        onReady();
      }),
      listen<DownloadError>("model-download-error", (event) => {
        setError(`${event.payload.model}: ${event.payload.error}`);
      }),
    ];

    invoke("download_missing_models").catch((e) => setError(String(e)));

    return () => {
      unlistenPromises.forEach((p) => p.then((unlisten) => unlisten()));
    };
  }, [checking, missing, onReady]);

  if (checking || missing.length === 0) return null;

  return (
    <div className="first-run-setup">
      <h1 className="first-run-setup__title">Setting up kai-notetaker</h1>
      <p className="first-run-setup__subtitle">
        Downloading the local AI models this app runs on — this only happens once.
      </p>

      {error && <div className="first-run-setup__error">{error}</div>}

      <div className="first-run-setup__models">
        {missing.map((name) => {
          const progress = progressByModel[name];
          const percent = progress && progress.total > 0 ? Math.round((progress.downloaded / progress.total) * 100) : 0;
          const done = percent >= 100;
          return (
            <div key={name} className="first-run-setup__model">
              <div className="first-run-setup__model-row">
                <span className="first-run-setup__model-name">{name}</span>
                <span className="first-run-setup__model-status">
                  {progress ? `${formatBytes(progress.downloaded)} / ${formatBytes(progress.total)}` : "Waiting…"}
                </span>
              </div>
              <div className="first-run-setup__bar">
                <div
                  className={`first-run-setup__bar-fill ${done ? "first-run-setup__bar-fill--done" : ""}`}
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
