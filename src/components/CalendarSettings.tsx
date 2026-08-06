import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./CalendarSettings.css";

interface UpcomingMeeting {
  subject: string;
  start: string;
  end: string;
  attendees: string[];
  join_url: string | null;
}

export function CalendarSettings() {
  const [connected, setConnected] = useState<boolean | null>(null);
  const [clientIdDraft, setClientIdDraft] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meetings, setMeetings] = useState<UpcomingMeeting[] | null>(null);
  const [loadingMeetings, setLoadingMeetings] = useState(false);

  useEffect(() => {
    invoke<boolean>("is_microsoft_calendar_connected")
      .then(setConnected)
      .catch((e) => setError(String(e)));
  }, []);

  const handleConnect = () => {
    const clientId = clientIdDraft.trim();
    if (!clientId) {
      setError("Paste the Application (client) ID from your Azure App Registration first.");
      return;
    }
    setError(null);
    setConnecting(true);
    // Opens the real Microsoft sign-in page in the default browser and
    // blocks (up to 3 minutes) until that browser redirects back — the
    // command doesn't return until the whole consent flow finishes.
    invoke("connect_microsoft_calendar", { clientId })
      .then(() => setConnected(true))
      .catch((e) => setError(String(e)))
      .finally(() => setConnecting(false));
  };

  const handleListMeetings = () => {
    setLoadingMeetings(true);
    setError(null);
    invoke<UpcomingMeeting[]>("list_upcoming_meetings", { hoursAhead: 48 })
      .then(setMeetings)
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingMeetings(false));
  };

  if (connected === null) {
    return <div className="calendar-settings">Checking calendar connection…</div>;
  }

  return (
    <div className="calendar-settings">
      <h2 className="calendar-settings__heading">Microsoft 365 / Outlook Calendar</h2>

      {error && <div className="calendar-settings__error">{error}</div>}

      {!connected ? (
        <div className="calendar-settings__connect">
          <p className="calendar-settings__hint">
            Paste the Application (client) ID from your Azure App Registration below, then click Connect. Your
            browser will open Microsoft's real sign-in page — nothing is sent anywhere except directly to Microsoft.
          </p>
          <input
            type="text"
            className="calendar-settings__client-id-input"
            placeholder="Application (client) ID"
            value={clientIdDraft}
            onChange={(e) => setClientIdDraft(e.target.value)}
          />
          <button type="button" disabled={connecting} onClick={handleConnect}>
            {connecting ? "Waiting for sign-in in your browser…" : "Connect Microsoft Calendar"}
          </button>
        </div>
      ) : (
        <div className="calendar-settings__connected">
          <p className="calendar-settings__status">✓ Connected</p>
          <button type="button" disabled={loadingMeetings} onClick={handleListMeetings}>
            {loadingMeetings ? "Loading…" : "Show upcoming meetings (next 48h)"}
          </button>
          {meetings && (
            <ul className="calendar-settings__meetings">
              {meetings.length === 0 && <li className="calendar-settings__meeting">Nothing on the calendar in the next 48 hours.</li>}
              {meetings.map((m, i) => (
                <li key={i} className="calendar-settings__meeting">
                  <div className="calendar-settings__meeting-subject">{m.subject}</div>
                  <div className="calendar-settings__meeting-time">
                    {m.start} – {m.end}
                  </div>
                  {m.attendees.length > 0 && <div className="calendar-settings__meeting-attendees">With: {m.attendees.join(", ")}</div>}
                  {m.join_url && (
                    <a href={m.join_url} target="_blank" rel="noreferrer" className="calendar-settings__meeting-join">
                      Join link
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
