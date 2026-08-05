import { useEffect, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import "./MeetingDetail.css";

interface TranscriptSegmentRow {
  speaker: number | null;
  start_ms: number;
  end_ms: number;
  text: string;
}

interface ActionItemRow {
  description: string;
  owner: string | null;
  due_date: string | null;
}

interface MeetingDetailData {
  id: number;
  created_at: string;
  title: string | null;
  duration_secs: number;
  status: "processing" | "ready" | "failed" | string;
  error_message: string | null;
  summary: string | null;
  transcript: TranscriptSegmentRow[];
  action_items: ActionItemRow[];
  audio_path: string | null;
}

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

interface MeetingDetailProps {
  meetingId: number;
  onBack: () => void;
}

export function MeetingDetail({ meetingId, onBack }: MeetingDetailProps) {
  const [detail, setDetail] = useState<MeetingDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const load = () => {
      invoke<MeetingDetailData>("get_meeting_detail", { meetingId })
        .then((d) => {
          if (cancelled) return;
          setDetail(d);
          // Stop polling once processing has finished one way or another.
          if (d.status !== "processing" && interval) {
            clearInterval(interval);
          }
        })
        .catch((e) => !cancelled && setError(String(e)));
    };

    load();
    interval = setInterval(load, 3000);
    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [meetingId]);

  return (
    <div className="detail-screen">
      <button type="button" className="detail-screen__back" onClick={onBack}>
        ← Back to meetings
      </button>

      {error && <div className="recording-screen__error">{error}</div>}

      {detail && (
        <>
          <h1 className="detail-screen__title">{detail.title ?? "Processing…"}</h1>
          <div className="detail-screen__meta">
            {Math.floor(detail.duration_secs / 60)}:{(detail.duration_secs % 60).toString().padStart(2, "0")}
          </div>

          {detail.audio_path && (
            <audio className="detail-audio-player" controls src={convertFileSrc(detail.audio_path)} />
          )}

          {detail.status === "processing" && (
            <div className="detail-screen__processing">
              Transcribing, diarizing, and summarizing this meeting — this can take a few minutes for longer
              recordings. This page updates automatically.
            </div>
          )}

          {detail.status === "failed" && (
            <div className="detail-screen__failed">
              Processing failed: {detail.error_message ?? "unknown error"}
            </div>
          )}

          {detail.status === "ready" && (
            <>
              {detail.summary && (
                <div className="detail-section">
                  <h2 className="detail-section__heading">Summary</h2>
                  <div className="detail-summary">{detail.summary}</div>
                </div>
              )}

              {detail.action_items.length > 0 && (
                <div className="detail-section">
                  <h2 className="detail-section__heading">Action Items</h2>
                  {detail.action_items.map((item, i) => (
                    <div key={i} className="action-item">
                      <span className="action-item__description">{item.description}</span>
                      <span className="action-item__meta">
                        {[item.owner, item.due_date].filter(Boolean).join(" · ")}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <div className="detail-section">
                <h2 className="detail-section__heading">Transcript</h2>
                {detail.transcript.map((seg, i) => (
                  <div key={i} className="transcript-line">
                    <span className="transcript-line__speaker">
                      {seg.speaker !== null ? `Speaker ${seg.speaker}` : "—"}
                      <br />
                      {formatTimestamp(seg.start_ms)}
                    </span>
                    <span className="transcript-line__text">{seg.text}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
