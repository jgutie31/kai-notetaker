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

interface AutoJoinLogEntry {
  event_id: string;
  subject: string;
  /** Unix seconds. */
  triggered_at: number;
}

/** One connectable calendar/meetings provider, driving one section of this tab. */
interface Provider {
  key: "microsoft" | "google" | "zoom";
  heading: string;
  /** Tauri command that runs the real browser consent flow. */
  connectCommand: string;
  /** Tauri command reporting whether tokens are already stored. */
  isConnectedCommand: string;
  connectLabel: string;
  clientIdPlaceholder: string;
  /** What the user has to have set up before pasting a client ID. */
  hint: string;
  /** Shown when the Connect button is pressed with an empty input. */
  missingClientIdError: string;
}

/**
 * All three providers are structurally identical — same OAuth engine
 * Rust-side, so same connect UI here. Declaring them as data rather than
 * three near-copies of the same JSX means a fourth provider is one entry,
 * and it's impossible for the three sections to drift apart visually.
 */
const PROVIDERS: Provider[] = [
  {
    key: "microsoft",
    heading: "Microsoft 365 / Outlook Calendar",
    connectCommand: "connect_microsoft_calendar",
    isConnectedCommand: "is_microsoft_calendar_connected",
    connectLabel: "Connect Microsoft Calendar",
    clientIdPlaceholder: "Application (client) ID",
    hint:
      "Paste the Application (client) ID from your Azure App Registration below, then click Connect. Your " +
      "browser will open Microsoft's real sign-in page — nothing is sent anywhere except directly to Microsoft.",
    missingClientIdError: "Paste the Application (client) ID from your Azure App Registration first.",
  },
  {
    key: "google",
    heading: "Google Calendar",
    connectCommand: "connect_google_calendar",
    isConnectedCommand: "is_google_calendar_connected",
    connectLabel: "Connect Google Calendar",
    clientIdPlaceholder: "Client ID (…apps.googleusercontent.com)",
    hint:
      "In the Google Cloud console, create an OAuth client ID of type Desktop app and paste its Client ID below. " +
      "No client secret is needed — a desktop app is a public client and this uses PKCE. Read-only calendar " +
      "access only; your browser opens Google's real sign-in page.",
    missingClientIdError: "Paste the Client ID from your Google Cloud OAuth client (Desktop app) first.",
  },
  {
    key: "zoom",
    heading: "Zoom",
    connectCommand: "connect_zoom",
    isConnectedCommand: "is_zoom_connected",
    connectLabel: "Connect Zoom",
    clientIdPlaceholder: "Public Client ID",
    hint:
      "One-time setup: in the Zoom Marketplace, open your app's Basic Information → App Credentials and turn on " +
      "“Use Public Client OAuth”, then paste the public Client ID below. That app type is what lets this connect " +
      "with PKCE and no client secret — a standard Zoom app's client ID will not work here.",
    missingClientIdError:
      "Paste the public Client ID from your Zoom app (with “Use Public Client OAuth” enabled) first.",
  },
];

export function CalendarSettings() {
  /** Per-provider connection state; `null` until the first check resolves. */
  const [connectedByProvider, setConnectedByProvider] = useState<Record<string, boolean | null>>({
    microsoft: null,
    google: null,
    zoom: null,
  });
  const [clientIdDrafts, setClientIdDrafts] = useState<Record<string, string>>({});
  const [connectingProvider, setConnectingProvider] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meetings, setMeetings] = useState<UpcomingMeeting[] | null>(null);
  const [loadingMeetings, setLoadingMeetings] = useState(false);
  const [windowDays, setWindowDays] = useState(7);
  const [autoJoin, setAutoJoin] = useState(false);
  const [autoJoined, setAutoJoined] = useState<AutoJoinLogEntry[]>([]);

  const microsoftConnected = connectedByProvider.microsoft === true;
  // Auto-join polls every connected provider, so its controls belong to the
  // tab as a whole — not to the Microsoft section. Someone who connects only
  // Google or only Zoom must still be able to reach the toggle.
  const anyConnected = PROVIDERS.some((p) => connectedByProvider[p.key] === true);
  const stillChecking = PROVIDERS.some((p) => connectedByProvider[p.key] === null);

  useEffect(() => {
    for (const provider of PROVIDERS) {
      invoke<boolean>(provider.isConnectedCommand)
        .then((isConnected) =>
          setConnectedByProvider((prev) => ({ ...prev, [provider.key]: isConnected })),
        )
        .catch((e) => {
          // A provider whose check fails is shown as disconnected rather
          // than leaving the whole tab stuck on "Checking…" forever.
          setConnectedByProvider((prev) => ({ ...prev, [provider.key]: false }));
          setError(String(e));
        });
    }
    invoke<boolean>("get_auto_join_enabled")
      .then(setAutoJoin)
      .catch((e) => setError(String(e)));
  }, []);

  // Refresh the auto-joined log on mount and every 30s, so a meeting the
  // poller picks up while this tab is open shows up on its own rather
  // than only after a manual navigation.
  useEffect(() => {
    const load = () => {
      invoke<AutoJoinLogEntry[]>("list_auto_joined_meetings")
        .then(setAutoJoined)
        .catch(() => {
          /* a failed log refresh is not worth interrupting the user over */
        });
    };
    load();
    const timer = setInterval(load, 30_000);
    return () => clearInterval(timer);
  }, []);

  const handleToggleAutoJoin = (enabled: boolean) => {
    // Optimistic: the checkbox reflects intent immediately, and reverts
    // if the write actually failed.
    setAutoJoin(enabled);
    setError(null);
    invoke("set_auto_join_enabled", { enabled }).catch((e) => {
      setError(String(e));
      setAutoJoin(!enabled);
    });
  };

  const handleConnect = (provider: Provider) => {
    const clientId = (clientIdDrafts[provider.key] ?? "").trim();
    if (!clientId) {
      setError(provider.missingClientIdError);
      return;
    }
    setError(null);
    setConnectingProvider(provider.key);
    // Opens the provider's real sign-in page in the default browser and
    // blocks (up to 3 minutes) until that browser redirects back — the
    // command doesn't return until the whole consent flow finishes.
    invoke(provider.connectCommand, { clientId })
      .then(() => setConnectedByProvider((prev) => ({ ...prev, [provider.key]: true })))
      .catch((e) => setError(String(e)))
      .finally(() => setConnectingProvider(null));
  };

  const handleListMeetings = () => {
    setLoadingMeetings(true);
    setError(null);
    invoke<UpcomingMeeting[]>("list_upcoming_meetings", { hoursAhead: windowDays * 24 })
      .then(setMeetings)
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingMeetings(false));
  };

  if (stillChecking) {
    return <div className="calendar-settings">Checking calendar connections…</div>;
  }

  return (
    <div className="calendar-settings">
      {error && <div className="calendar-settings__error">{error}</div>}

      {PROVIDERS.map((provider) => {
        const isConnected = connectedByProvider[provider.key] === true;
        const isConnecting = connectingProvider === provider.key;
        return (
          <section key={provider.key} className="calendar-settings__provider">
            <h2 className="calendar-settings__heading">{provider.heading}</h2>
            {!isConnected ? (
              <div className="calendar-settings__connect">
                <p className="calendar-settings__hint">{provider.hint}</p>
                <input
                  type="text"
                  className="calendar-settings__client-id-input"
                  placeholder={provider.clientIdPlaceholder}
                  value={clientIdDrafts[provider.key] ?? ""}
                  onChange={(e) =>
                    setClientIdDrafts((prev) => ({ ...prev, [provider.key]: e.target.value }))
                  }
                />
                <button
                  type="button"
                  disabled={connectingProvider !== null}
                  onClick={() => handleConnect(provider)}
                >
                  {isConnecting ? "Waiting for sign-in in your browser…" : provider.connectLabel}
                </button>
              </div>
            ) : (
              <p className="calendar-settings__status">✓ Connected</p>
            )}
          </section>
        );
      })}

      {anyConnected && (
        <div className="calendar-settings__connected">
          <div className="calendar-settings__auto-join">
            <label className="calendar-settings__auto-join-toggle">
              <input
                type="checkbox"
                checked={autoJoin}
                onChange={(e) => handleToggleAutoJoin(e.target.checked)}
              />
              <span>Auto-join &amp; record calendar meetings</span>
            </label>
            <p className="calendar-settings__hint">
              When a meeting with a Teams, Google Meet, or Zoom join link is about to start, its link opens and
              recording begins automatically, across every provider connected above. Recording stops on its own at
              the meeting's scheduled end time, and you'll be asked whether to stop if the call goes quiet for a
              minute. Off by default; turning it off takes effect within a minute.
            </p>

            {autoJoined.length > 0 && (
              <>
                <div className="calendar-settings__auto-join-log-heading">Auto-joined so far</div>
                <ul className="calendar-settings__auto-join-log">
                  {autoJoined.map((entry) => (
                    <li key={entry.event_id} className="calendar-settings__auto-join-log-item">
                      <span className="calendar-settings__auto-join-log-subject">{entry.subject}</span>
                      <span className="calendar-settings__auto-join-log-time">
                        {new Date(entry.triggered_at * 1000).toLocaleString()}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>

          {/*
            The meeting browser below reads Microsoft Graph specifically
            (`list_upcoming_meetings`), so it only appears when Microsoft is
            connected — showing it for a Google-only or Zoom-only setup
            would just produce an "isn't connected yet" error on click.
            Auto-join itself is unaffected: it polls every connected
            provider regardless of this panel.
          */}
          {microsoftConnected && (
            <div className="calendar-settings__window">
              <label htmlFor="calendar-window-select">Show Outlook meetings within:</label>
              <select
                id="calendar-window-select"
                value={windowDays}
                onChange={(e) => setWindowDays(Number(e.target.value))}
              >
                <option value={2}>2 days</option>
                <option value={7}>7 days</option>
                <option value={14}>14 days</option>
                <option value={30}>30 days</option>
              </select>
            </div>
          )}
          {microsoftConnected && (
            <button type="button" disabled={loadingMeetings} onClick={handleListMeetings}>
              {loadingMeetings ? "Loading…" : `Show upcoming meetings (next ${windowDays}d)`}
            </button>
          )}
          {meetings && (
            <ul className="calendar-settings__meetings">
              {meetings.length === 0 && <li className="calendar-settings__meeting">Nothing on the calendar in the next {windowDays} days.</li>}
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
