import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./MeetingLibrary.css";

interface MeetingListItem {
  id: number;
  created_at: string;
  title: string | null;
  duration_secs: number;
  status: "processing" | "ready" | "failed" | string;
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatDate(isoLike: string): string {
  // SQLite's datetime('now') returns "YYYY-MM-DD HH:MM:SS" (UTC, no 'Z') —
  // append it explicitly so Date parses it as UTC instead of local time.
  const date = new Date(isoLike.replace(" ", "T") + "Z");
  if (Number.isNaN(date.getTime())) return isoLike;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

interface MeetingLibraryProps {
  onSelectMeeting: (id: number) => void;
  onDelete: (meetingId: number, title: string) => void;
  /** Bumping this forces a re-fetch — used after an undo restores a meeting. */
  refreshToken: number;
}

export function MeetingLibrary({ onSelectMeeting, onDelete, refreshToken }: MeetingLibraryProps) {
  const [meetings, setMeetings] = useState<MeetingListItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    invoke<MeetingListItem[]>("list_meetings")
      .then(setMeetings)
      .catch((e) => setError(String(e)));
  };

  useEffect(() => {
    refresh();
    // Meetings finish processing asynchronously in the background — poll
    // so a "processing" row flips to "ready" without requiring a manual
    // refresh or app restart.
    const interval = setInterval(refresh, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    refresh();
  }, [refreshToken]);

  const handleDelete = (e: React.MouseEvent, m: MeetingListItem) => {
    e.stopPropagation();
    invoke("delete_meeting", { meetingId: m.id })
      .then(() => {
        setMeetings((prev) => prev.filter((x) => x.id !== m.id));
        onDelete(m.id, m.title ?? "Untitled meeting");
      })
      .catch((err) => setError(String(err)));
  };

  return (
    <div className="library-screen">
      <h1 className="library-screen__title">Meetings</h1>

      {error && <div className="recording-screen__error">{error}</div>}

      {meetings.length === 0 && !error && (
        <div className="library-screen__empty">No meetings recorded yet.</div>
      )}

      <div className="library-list">
        {meetings.map((m) => (
          <div key={m.id} className="library-row" onClick={() => onSelectMeeting(m.id)}>
            <div className="library-row__main">
              <span className="library-row__title">{m.title ?? "Processing…"}</span>
              <span className="library-row__meta">
                {formatDate(m.created_at)} · {formatDuration(m.duration_secs)}
              </span>
            </div>
            <span className={`library-row__status library-row__status--${m.status}`}>{m.status}</span>
            <button
              type="button"
              className="library-row__delete"
              aria-label="Delete meeting"
              onClick={(e) => handleDelete(e, m)}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
