import { useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RecordingControl } from "./components/RecordingControl";
import { MeetingLibrary } from "./components/MeetingLibrary";
import { MeetingDetail } from "./components/MeetingDetail";
import { FirstRunSetup } from "./components/FirstRunSetup";
import { CalendarSettings } from "./components/CalendarSettings";
import "./styles/tokens.css";
import "./App.css";

type View = "recording" | "library" | "calendar";

const UNDO_WINDOW_MS = 8000;

function App() {
  const [view, setView] = useState<View>("recording");
  const [selectedMeetingId, setSelectedMeetingId] = useState<number | null>(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [pendingUndo, setPendingUndo] = useState<{ id: number; title: string } | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const undoTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Shared by both the meeting list and the meeting detail screen — a
  // delete triggered from either place gets the same undo grace window,
  // since accidental deletes are equally likely from both.
  const handleDelete = (meetingId: number, title: string) => {
    if (undoTimer.current) clearTimeout(undoTimer.current);
    setPendingUndo({ id: meetingId, title });
    undoTimer.current = setTimeout(() => setPendingUndo(null), UNDO_WINDOW_MS);
  };

  const handleUndo = () => {
    if (!pendingUndo) return;
    if (undoTimer.current) clearTimeout(undoTimer.current);
    invoke("undelete_meeting", { meetingId: pendingUndo.id }).then(() => {
      setPendingUndo(null);
      setRefreshToken((n) => n + 1);
    });
  };

  const undoToast = pendingUndo && (
    <div className="app-undo-toast">
      <span>Deleted "{pendingUndo.title}"</span>
      <button type="button" className="app-undo-toast__button" onClick={handleUndo}>
        Undo
      </button>
    </div>
  );

  if (!modelsReady) {
    return (
      <main className="app-shell">
        <div className="app-content">
          <FirstRunSetup onReady={() => setModelsReady(true)} />
        </div>
      </main>
    );
  }

  if (selectedMeetingId !== null) {
    return (
      <main className="app-shell">
        <div className="app-content">
          <MeetingDetail
            meetingId={selectedMeetingId}
            onBack={() => setSelectedMeetingId(null)}
            onDelete={handleDelete}
          />
          {undoToast}
        </div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <nav className="app-nav">
        <button
          type="button"
          className={`app-nav__tab ${view === "recording" ? "app-nav__tab--active" : ""}`}
          onClick={() => setView("recording")}
        >
          Record
        </button>
        <button
          type="button"
          className={`app-nav__tab ${view === "library" ? "app-nav__tab--active" : ""}`}
          onClick={() => setView("library")}
        >
          Meetings
        </button>
        <button
          type="button"
          className={`app-nav__tab ${view === "calendar" ? "app-nav__tab--active" : ""}`}
          onClick={() => setView("calendar")}
        >
          Calendar
        </button>
      </nav>
      <div className="app-content">
        {view === "recording" && <RecordingControl />}
        {view === "library" && (
          <MeetingLibrary
            onSelectMeeting={(id) => setSelectedMeetingId(id)}
            onDelete={handleDelete}
            refreshToken={refreshToken}
          />
        )}
        {view === "calendar" && <CalendarSettings />}
        {undoToast}
      </div>
    </main>
  );
}

export default App;
