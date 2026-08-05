import { useState } from "react";
import { RecordingControl } from "./components/RecordingControl";
import { MeetingLibrary } from "./components/MeetingLibrary";
import { MeetingDetail } from "./components/MeetingDetail";
import "./styles/tokens.css";
import "./App.css";

type View = "recording" | "library";

function App() {
  const [view, setView] = useState<View>("recording");
  const [selectedMeetingId, setSelectedMeetingId] = useState<number | null>(null);

  if (selectedMeetingId !== null) {
    return (
      <main className="app-shell">
        <div className="app-content">
          <MeetingDetail meetingId={selectedMeetingId} onBack={() => setSelectedMeetingId(null)} />
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
      </nav>
      <div className="app-content">
        {view === "recording" ? (
          <RecordingControl />
        ) : (
          <MeetingLibrary onSelectMeeting={(id) => setSelectedMeetingId(id)} />
        )}
      </div>
    </main>
  );
}

export default App;
