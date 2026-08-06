import { useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import "./MeetingDetail.css";

interface TranscriptSegmentRow {
  id: number;
  speaker: number | null;
  speaker_label: string | null;
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

/// Finds the transcript segment whose [start_ms, end_ms) window contains
/// currentTimeMs. Falls back to the last segment whose start has already
/// passed, so the highlight doesn't just disappear during small gaps
/// between segments (pauses, cross-talk) — it should feel continuous.
function findActiveSegmentIndex(transcript: TranscriptSegmentRow[], currentTimeMs: number): number {
  const exact = transcript.findIndex((s) => currentTimeMs >= s.start_ms && currentTimeMs < s.end_ms);
  if (exact !== -1) return exact;

  let lastPassed = -1;
  for (let i = 0; i < transcript.length; i++) {
    if (transcript[i].start_ms <= currentTimeMs) lastPassed = i;
    else break;
  }
  return lastPassed;
}

interface MeetingDetailProps {
  meetingId: number;
  onBack: () => void;
  onDelete: (meetingId: number, title: string) => void;
}

const MIN_TOP_HEIGHT = 140;
const MIN_TRANSCRIPT_HEIGHT = 100;
const DEFAULT_TOP_FRACTION = 0.55;

export function MeetingDetail({ meetingId, onBack, onDelete }: MeetingDetailProps) {
  const [detail, setDetail] = useState<MeetingDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [topHeight, setTopHeight] = useState<number | null>(null);
  const [isResizing, setIsResizing] = useState(false);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [knownSpeakers, setKnownSpeakers] = useState<string[]>([]);
  // Transcript array indices (inclusive), not raw diarization speaker
  // indices — diarization can merge two different real people into the
  // same raw index on a real call, so editing is scoped to the exact lines
  // selected, not "every line diarization ever called this index."
  const [editingRange, setEditingRange] = useState<{ start: number; end: number } | null>(null);
  const [speakerNameDraft, setSpeakerNameDraft] = useState("");
  const [rememberSpeaker, setRememberSpeaker] = useState(true);
  const [applyToWholeSpeaker, setApplyToWholeSpeaker] = useState(false);
  const [labelingInFlight, setLabelingInFlight] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const segmentRefs = useRef<(HTMLDivElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const topRef = useRef<HTMLDivElement | null>(null);
  const resizerRef = useRef<HTMLDivElement | null>(null);

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

  useEffect(() => {
    invoke<string[]>("list_known_speakers").then(setKnownSpeakers).catch(() => {});
  }, []);

  const activeIndex = useMemo(
    () => (detail ? findActiveSegmentIndex(detail.transcript, currentTimeMs) : -1),
    [detail, currentTimeMs],
  );

  // Auto-scroll to whichever segment just became active — only fires when
  // the active index actually changes, not on every timeupdate tick, so it
  // doesn't fight a user who's manually scrolled elsewhere to read ahead.
  useEffect(() => {
    if (activeIndex < 0) return;
    segmentRefs.current[activeIndex]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeIndex]);

  const seekTo = (startMs: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = startMs / 1000;
  };

  const startEditingTitle = () => {
    if (!detail || detail.status !== "ready") return;
    setTitleDraft(detail.title ?? "");
    setIsEditingTitle(true);
  };

  const commitTitleEdit = () => {
    if (!detail) return;
    setIsEditingTitle(false);
    const trimmed = titleDraft.trim();
    if (!trimmed || trimmed === detail.title) return;
    invoke("rename_meeting", { meetingId: detail.id, title: trimmed })
      .then(() => setDetail((prev) => (prev ? { ...prev, title: trimmed } : prev)))
      .catch((e) => setError(String(e)));
  };

  const handleDeleteClick = () => {
    if (!detail) return;
    invoke("delete_meeting", { meetingId: detail.id })
      .then(() => {
        onDelete(detail.id, detail.title ?? "Untitled meeting");
        onBack();
      })
      .catch((e) => setError(String(e)));
  };

  // Plain click selects just this one line. Shift+click while a selection
  // is already open extends it to a range — lets a long mislabeled stretch
  // (diarization gave no index change at the real turn boundary) be fixed
  // in one action without dragging in unrelated lines that happen to share
  // the same raw speaker index elsewhere in the call.
  const startEditingSpeaker = (e: React.MouseEvent, seg: TranscriptSegmentRow, index: number) => {
    e.stopPropagation();
    if (seg.speaker === null) return;
    if (e.shiftKey && editingRange) {
      setEditingRange({ start: Math.min(editingRange.start, index), end: Math.max(editingRange.end, index) });
      return;
    }
    setEditingRange({ start: index, end: index });
    setSpeakerNameDraft(seg.speaker_label ?? "");
    setRememberSpeaker(true);
    setApplyToWholeSpeaker(false);
  };

  const commitSpeakerLabel = () => {
    if (!detail || !editingRange) return;
    const name = speakerNameDraft.trim();
    if (!name) {
      setEditingRange(null);
      return;
    }
    const selected = detail.transcript.slice(editingRange.start, editingRange.end + 1);
    const segmentIds = selected.map((s) => s.id);
    const wholeSpeakerIndex = applyToWholeSpeaker ? selected[0].speaker : null;
    setLabelingInFlight(true);
    invoke("label_transcript_segments", {
      meetingId: detail.id,
      segmentIds,
      name,
      remember: rememberSpeaker,
      applyToWholeSpeaker,
    })
      .then(() => {
        setDetail((prev) => {
          if (!prev) return prev;
          const idSet = new Set(segmentIds);
          return {
            ...prev,
            transcript: prev.transcript.map((s) =>
              wholeSpeakerIndex !== null ? (s.speaker === wholeSpeakerIndex ? { ...s, speaker_label: name } : s) : idSet.has(s.id) ? { ...s, speaker_label: name } : s,
            ),
          };
        });
        if (rememberSpeaker && !knownSpeakers.includes(name)) {
          setKnownSpeakers((prev) => [...prev, name].sort());
        }
        setEditingRange(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLabelingInFlight(false));
  };

  // Free-drag resize between the fixed top section and the transcript pane.
  // Reads real layout numbers at drag-start (rather than assuming a fixed
  // window size) so it clamps correctly regardless of window resize.
  const handleResizerPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    const topEl = topRef.current;
    if (!container || !topEl) return;

    e.preventDefault();
    const startY = e.clientY;
    const startHeight = topEl.getBoundingClientRect().height;
    const containerHeight = container.getBoundingClientRect().height;
    const resizerHeight = e.currentTarget.getBoundingClientRect().height;
    const maxTopHeight = containerHeight - MIN_TRANSCRIPT_HEIGHT - resizerHeight;

    setIsResizing(true);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const next = Math.min(
        Math.max(startHeight + (moveEvent.clientY - startY), MIN_TOP_HEIGHT),
        Math.max(maxTopHeight, MIN_TOP_HEIGHT),
      );
      setTopHeight(next);
    };

    const handlePointerUp = () => {
      setIsResizing(false);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
  };

  // Real bug fixed here: previously `topHeight === null` meant "let CSS
  // decide naturally" (flex-shrink: 0, no cap) — at a small window size
  // that let the top section's content (summary + action items) grow
  // taller than the actual window, pushing/clipping content off-screen
  // instead of scrolling, since flex-shrink:0 never gave the element a
  // constrained height for its own overflow-y:auto to act on. Jeremiah
  // only noticed Action Items existed after going fullscreen. Fix: always
  // maintain a real, clamped pixel height — recalculated whenever the
  // window/container is resized, not just set once at drag-time — so a
  // smaller window properly scrolls the top section instead of hiding
  // content, and a manually-dragged split stays valid instead of going
  // stale if the window shrinks afterward.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const recalculate = () => {
      const containerHeight = container.getBoundingClientRect().height;
      if (containerHeight === 0) return;
      const resizerHeight = resizerRef.current?.getBoundingClientRect().height ?? 14;
      const maxTop = Math.max(containerHeight - MIN_TRANSCRIPT_HEIGHT - resizerHeight, MIN_TOP_HEIGHT);

      setTopHeight((prev) => {
        const base = prev ?? containerHeight * DEFAULT_TOP_FRACTION;
        return Math.min(Math.max(base, MIN_TOP_HEIGHT), maxTop);
      });
    };

    recalculate();
    const observer = new ResizeObserver(recalculate);
    observer.observe(container);
    return () => observer.disconnect();
  }, [detail?.status]);

  return (
    <div ref={containerRef} className={`detail-screen ${isResizing ? "detail-screen--resizing" : ""}`}>
      <div
        ref={topRef}
        className="detail-screen__top"
        style={topHeight !== null ? { height: topHeight, flex: "none" } : undefined}
      >
        <div className="detail-screen__toolbar">
          <button type="button" className="detail-screen__back" onClick={onBack}>
            ← Back to meetings
          </button>
          {detail && (
            <button type="button" className="detail-screen__delete" onClick={handleDeleteClick}>
              Delete
            </button>
          )}
        </div>

        {error && <div className="recording-screen__error">{error}</div>}

        {detail && (
          <>
            {isEditingTitle ? (
              <input
                type="text"
                autoFocus
                className="detail-screen__title-input"
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                onBlur={commitTitleEdit}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitTitleEdit();
                  if (e.key === "Escape") setIsEditingTitle(false);
                }}
              />
            ) : (
              <h1
                className={`detail-screen__title ${detail.status === "ready" ? "detail-screen__title--editable" : ""}`}
                onClick={startEditingTitle}
                title={detail.status === "ready" ? "Click to rename" : undefined}
              >
                {detail.title ?? "Processing…"}
              </h1>
            )}
            <div className="detail-screen__meta">
              {Math.floor(detail.duration_secs / 60)}:{(detail.duration_secs % 60).toString().padStart(2, "0")}
            </div>

            {detail.audio_path && (
              <audio
                ref={audioRef}
                className="detail-audio-player"
                controls
                src={convertFileSrc(detail.audio_path)}
                onTimeUpdate={(e) => setCurrentTimeMs(e.currentTarget.currentTime * 1000)}
              />
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
              </>
            )}
          </>
        )}
      </div>

      {detail && detail.status === "ready" && (
        <>
          <div
            ref={resizerRef}
            className="detail-screen__resizer"
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize transcript section"
            onPointerDown={handleResizerPointerDown}
          />
          <div className="detail-screen__transcript">
            <h2 className="detail-section__heading">Transcript</h2>
            {detail.transcript.map((seg, i) => {
              const inRange = editingRange !== null && i >= editingRange.start && i <= editingRange.end;
              const isEditFormRow = editingRange !== null && i === editingRange.end;
              const rangeSize = editingRange ? editingRange.end - editingRange.start + 1 : 0;
              return (
                <div
                  key={seg.id}
                  ref={(el) => {
                    segmentRefs.current[i] = el;
                  }}
                  className={`transcript-line ${i === activeIndex ? "transcript-line--active" : ""} ${
                    inRange ? "transcript-line--selected" : ""
                  } ${detail.audio_path ? "transcript-line--clickable" : ""}`}
                  onClick={() => detail.audio_path && seekTo(seg.start_ms)}
                >
                  {isEditFormRow ? (
                    <span className="transcript-line__speaker-edit" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        autoFocus
                        list="known-speakers-list"
                        className="transcript-line__speaker-input"
                        placeholder="Name"
                        value={speakerNameDraft}
                        onChange={(e) => setSpeakerNameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitSpeakerLabel();
                          if (e.key === "Escape") setEditingRange(null);
                        }}
                      />
                      {rangeSize > 1 && <div className="transcript-line__speaker-hint">Labeling {rangeSize} lines</div>}
                      <label className="transcript-line__speaker-remember">
                        <input type="checkbox" checked={rememberSpeaker} onChange={(e) => setRememberSpeaker(e.target.checked)} />
                        Remember
                      </label>
                      <label className="transcript-line__speaker-remember">
                        <input type="checkbox" checked={applyToWholeSpeaker} onChange={(e) => setApplyToWholeSpeaker(e.target.checked)} />
                        Apply everywhere diarization called this Speaker {seg.speaker}
                      </label>
                      <button type="button" disabled={labelingInFlight} onClick={commitSpeakerLabel}>
                        Save
                      </button>
                    </span>
                  ) : (
                    <span
                      className={`transcript-line__speaker transcript-line__speaker--editable ${inRange ? "transcript-line__speaker--selected" : ""}`}
                      onClick={(e) => startEditingSpeaker(e, seg, i)}
                      title="Click to name this speaker. Shift+click another line to select a range."
                    >
                      {seg.speaker_label ?? (seg.speaker !== null ? `Speaker ${seg.speaker}` : "—")}
                      <br />
                      {formatTimestamp(seg.start_ms)}
                    </span>
                  )}
                  <span className="transcript-line__text">{seg.text}</span>
                </div>
              );
            })}
          </div>
        </>
      )}
      <datalist id="known-speakers-list">
        {knownSpeakers.map((name) => (
          <option key={name} value={name} />
        ))}
      </datalist>
    </div>
  );
}
