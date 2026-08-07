//! Google Calendar provider — the second calendar provider, built on the
//! exact same generic OAuth engine (`oauth.rs`) the Microsoft provider in
//! `calendar.rs` uses. There is deliberately no second PKCE/state-machine
//! implementation here (ISC-182): this module only knows Google's URLs,
//! scope, and JSON shape.
//!
//! Every endpoint and field name below was verified against Google's own
//! current documentation before writing this file, not assumed from
//! memory — the same discipline `calendar.rs` applied to Microsoft Graph:
//! - Native/installed-app OAuth (loopback redirect, PKCE, `client_secret`
//!   not applicable for an installed app):
//!   developers.google.com/identity/protocols/oauth2/native-app
//! - Authorize endpoint `https://accounts.google.com/o/oauth2/v2/auth`,
//!   token endpoint `https://oauth2.googleapis.com/token`: same page.
//! - `https://www.googleapis.com/auth/calendar.events.readonly` scope
//!   (narrower than `calendar.readonly`, which also grants calendar
//!   metadata/settings access this feature never needs — least privilege,
//!   refined 2026-08-06 per an Advisor VERIFY review):
//!   developers.google.com/calendar/api/guides/auth
//! - `events.list` request URL, `timeMin`/`timeMax` (RFC3339, exclusive
//!   bounds on end/start respectively), `singleEvents`, and the `items`
//!   response array:
//!   developers.google.com/workspace/calendar/api/v3/reference/events/list
//! - Event resource fields (`id`, `summary`, `start.dateTime`/`start.date`,
//!   `end.dateTime`/`end.date`, `attendees[].email`/`.displayName`,
//!   `hangoutLink`, `conferenceData.entryPoints[].entryPointType`/`.uri`):
//!   developers.google.com/workspace/calendar/api/v3/reference/events

use crate::calendar::{CalendarError, UpcomingMeeting};
use crate::oauth::{self, OAuthProviderConfig};
use serde::Deserialize;

pub const GOOGLE_PROVIDER_ID: &str = "google";

/// `entryPointType` value that identifies a joinable video conference
/// entry point (the others are `phone`, `sip`, `more`) — a phone dial-in
/// URI is not something to hand to `open::that` as a "join link".
const VIDEO_ENTRY_POINT: &str = "video";

/// No `client_secret`: per Google's own native-app documentation, an
/// installed application is a public client and the secret "is obviously
/// not treated as a secret" — this app is PKCE-only, matching the
/// Microsoft provider's existing public-client posture (ISC-183).
///
/// `access_type=offline` + `prompt=consent` are the two provider-specific
/// extras `oauth.rs`'s `extra_authorize_params` was built for: without
/// them Google returns no `refresh_token`, and the connection would
/// silently die roughly an hour after consent.
fn google_config(client_id: &str) -> OAuthProviderConfig {
    OAuthProviderConfig {
        authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
        token_url: "https://oauth2.googleapis.com/token".to_string(),
        client_id: client_id.to_string(),
        scope: "https://www.googleapis.com/auth/calendar.events.readonly".to_string(),
        extra_authorize_params: vec![
            ("access_type".to_string(), "offline".to_string()),
            ("prompt".to_string(), "consent".to_string()),
        ],
    }
}

/// Google's `EventDateTime`: exactly one of `dateTime` (a timed event) or
/// `date` (an all-day event, `yyyy-mm-dd`) is present, never both.
#[derive(Debug, Deserialize)]
struct GoogleEventDateTime {
    #[serde(rename = "dateTime")]
    date_time: Option<String>,
    date: Option<String>,
}

impl GoogleEventDateTime {
    /// The timed value when there is one; otherwise the bare all-day date.
    ///
    /// An all-day `date` ("2026-08-10") deliberately flows through as-is
    /// rather than being invented into a timestamp: `auto_join::eligibility`
    /// can't parse it, so it fails closed with `UnparseableStart` and the
    /// event is never auto-joined (ISC-187). That's the honest outcome —
    /// an all-day event has no real start moment to record from, and
    /// synthesizing midnight-local would produce a nonsensical trigger
    /// window instead of no window at all.
    fn as_raw(&self) -> String {
        self.date_time.clone().or_else(|| self.date.clone()).unwrap_or_default()
    }
}

#[derive(Debug, Deserialize)]
struct GoogleAttendee {
    email: Option<String>,
    #[serde(rename = "displayName")]
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoogleEntryPoint {
    #[serde(rename = "entryPointType")]
    entry_point_type: Option<String>,
    uri: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoogleConferenceData {
    #[serde(default, rename = "entryPoints")]
    entry_points: Vec<GoogleEntryPoint>,
}

#[derive(Debug, Deserialize)]
struct GoogleEvent {
    id: String,
    /// Optional in Google's own schema — an event with no title is legal
    /// and must not fail the whole response's deserialization.
    summary: Option<String>,
    start: GoogleEventDateTime,
    end: GoogleEventDateTime,
    #[serde(default)]
    attendees: Vec<GoogleAttendee>,
    #[serde(default, rename = "hangoutLink")]
    hangout_link: Option<String>,
    #[serde(default, rename = "conferenceData")]
    conference_data: Option<GoogleConferenceData>,
}

#[derive(Debug, Deserialize)]
struct GoogleEventList {
    #[serde(default)]
    items: Vec<GoogleEvent>,
}

impl GoogleEvent {
    /// ISC-186: prefer the modern `conferenceData` video entry point,
    /// fall back to the legacy top-level `hangoutLink`. Both are real,
    /// documented fields; `hangoutLink` still appears on events created by
    /// older clients, so dropping it would silently make those events
    /// look join-link-less.
    fn resolve_join_url(&self) -> Option<String> {
        let from_conference_data = self.conference_data.as_ref().and_then(|c| {
            c.entry_points
                .iter()
                .find(|e| e.entry_point_type.as_deref() == Some(VIDEO_ENTRY_POINT))
                .and_then(|e| e.uri.clone())
        });
        from_conference_data.or_else(|| self.hangout_link.clone())
    }
}

impl From<GoogleEvent> for UpcomingMeeting {
    fn from(e: GoogleEvent) -> Self {
        let join_url = e.resolve_join_url();
        UpcomingMeeting {
            id: e.id,
            subject: e.summary.unwrap_or_else(|| "(no title)".to_string()),
            start: e.start.as_raw(),
            end: e.end.as_raw(),
            // Mirrors the Microsoft provider's null-name-falls-back-to-
            // address behavior: a real attendee with no display name shows
            // as their address rather than vanishing from the list.
            attendees: e
                .attendees
                .into_iter()
                .filter_map(|a| a.display_name.or(a.email))
                .collect(),
            join_url,
        }
    }
}

/// Runs the full interactive consent flow against Google's real sign-in
/// page. Structurally identical to `calendar::connect_microsoft` — same
/// PKCE pair, same loopback listener, same anti-CSRF state comparison,
/// same token storage — differing only in which `OAuthProviderConfig` and
/// provider id it hands to the shared engine.
pub fn connect_google(client_id: &str, port: u16) -> Result<(), CalendarError> {
    let config = google_config(client_id);
    let pkce = oauth::PkcePair::generate()?;
    let expected_state = oauth::generate_state()?;
    // Google's native-app docs explicitly support a loopback redirect on
    // an arbitrary port ("http://127.0.0.1:port" / "http://localhost:port"),
    // which is why this is a runtime-chosen port rather than one that has
    // to be pre-registered per-port in the Google Cloud console.
    let redirect_uri = format!("http://localhost:{port}/callback");
    let authorize_url = oauth::build_authorize_url(&config, &pkce, &redirect_uri, &expected_state);

    let _ = open::that(&authorize_url);

    let (code, returned_state) = oauth::await_authorization_code(port, std::time::Duration::from_secs(180))?;
    if returned_state.as_deref() != Some(expected_state.as_str()) {
        return Err(CalendarError::OAuth(oauth::OAuthError::StateMismatch));
    }

    let tokens = oauth::exchange_code_for_tokens(&config, &code, &pkce.verifier, &redirect_uri)?;
    oauth::store_tokens(GOOGLE_PROVIDER_ID, &tokens)?;
    Ok(())
}

pub fn is_google_connected() -> Result<bool, CalendarError> {
    Ok(oauth::load_tokens(GOOGLE_PROVIDER_ID)?.is_some())
}

/// Real events for the next `hours_ahead` hours from the user's primary
/// calendar, as `UpcomingMeeting` — the exact same shared struct the
/// Microsoft provider returns, which is what lets `auto_join` stay
/// provider-agnostic (ISC-172.1).
///
/// `singleEvents=true` is not cosmetic: without it Google returns the
/// underlying recurring-event object carrying the series' *original*
/// start date rather than the occurrence happening this week, which would
/// make every recurring meeting either permanently ineligible or trigger
/// against the wrong time. `orderBy=startTime` is only valid together
/// with `singleEvents=true` and makes the returned order deterministic.
pub fn list_upcoming_meetings(client_id: &str, hours_ahead: i64) -> Result<Vec<UpcomingMeeting>, CalendarError> {
    let config = google_config(client_id);
    let access_token = oauth::get_valid_access_token(GOOGLE_PROVIDER_ID, &config)?;

    let now = chrono::Utc::now();
    let until = now + chrono::Duration::hours(hours_ahead);

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get("https://www.googleapis.com/calendar/v3/calendars/primary/events")
        .query(&[
            ("timeMin", now.to_rfc3339()),
            ("timeMax", until.to_rfc3339()),
            ("singleEvents", "true".to_string()),
            ("orderBy", "startTime".to_string()),
        ])
        .bearer_auth(&access_token)
        .send()?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(CalendarError::ApiError(format!("{status}: {body}")));
    }

    let list: GoogleEventList = resp.json()?;
    Ok(list.items.into_iter().map(UpcomingMeeting::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_join::{self, Decision, Eligibility};
    use chrono::{DateTime, Utc};

    /// A real, Google-Events-shaped response. Every field name and level of
    /// nesting here was taken from Google's own Event resource reference
    /// (cited in this module's header), not invented to match the structs
    /// above — that's the whole point: if Google's field names were guessed
    /// wrong, this fixture fails to produce the expected values rather than
    /// quietly compiling.
    const REAL_SHAPED_GOOGLE_RESPONSE: &str = r#"{
        "kind": "calendar#events",
        "items": [
            {
                "kind": "calendar#event",
                "id": "6f1n2g3h4i5j6k7l8m9n0o1p2q",
                "status": "confirmed",
                "summary": "KCG HIPAA Readiness Sync",
                "start": { "dateTime": "2026-08-10T14:00:00-05:00", "timeZone": "America/Chicago" },
                "end": { "dateTime": "2026-08-10T15:00:00-05:00", "timeZone": "America/Chicago" },
                "attendees": [
                    { "email": "paula@example.com", "displayName": "Paula", "responseStatus": "accepted" },
                    { "email": "dave@example.com", "responseStatus": "needsAction" }
                ],
                "hangoutLink": "https://meet.google.com/legacy-link-abc",
                "conferenceData": {
                    "entryPoints": [
                        { "entryPointType": "video", "uri": "https://meet.google.com/abc-defg-hij", "label": "meet.google.com/abc-defg-hij" },
                        { "entryPointType": "phone", "uri": "tel:+1-555-0100", "label": "+1 555-0100" }
                    ],
                    "conferenceSolution": { "key": { "type": "hangoutsMeet" }, "name": "Google Meet" }
                }
            },
            {
                "kind": "calendar#event",
                "id": "legacy-hangout-link-only",
                "summary": "Older event, no conferenceData",
                "start": { "dateTime": "2026-08-11T09:00:00Z" },
                "end": { "dateTime": "2026-08-11T09:30:00Z" },
                "hangoutLink": "https://meet.google.com/legacy-only-xyz"
            },
            {
                "kind": "calendar#event",
                "id": "all-day-offsite",
                "summary": "Team offsite (all day)",
                "start": { "date": "2026-08-12" },
                "end": { "date": "2026-08-13" },
                "hangoutLink": "https://meet.google.com/allday-should-never-fire"
            },
            {
                "kind": "calendar#event",
                "id": "no-conference-at-all",
                "start": { "dateTime": "2026-08-13T09:00:00Z" },
                "end": { "dateTime": "2026-08-13T09:30:00Z" }
            }
        ]
    }"#;

    fn parse_fixture() -> Vec<UpcomingMeeting> {
        let list: GoogleEventList = serde_json::from_str(REAL_SHAPED_GOOGLE_RESPONSE).unwrap();
        list.items.into_iter().map(UpcomingMeeting::from).collect()
    }

    /// ISC-185: the real Google `items[]` field names deserialize into the
    /// shared `UpcomingMeeting`, with the same null-display-name-falls-back-
    /// to-email behavior the Microsoft provider already has.
    #[test]
    fn parses_a_real_google_shaped_response_into_upcoming_meetings() {
        let meetings = parse_fixture();
        assert_eq!(meetings.len(), 4);

        assert_eq!(meetings[0].id, "6f1n2g3h4i5j6k7l8m9n0o1p2q");
        assert_eq!(meetings[0].subject, "KCG HIPAA Readiness Sync");
        assert_eq!(meetings[0].start, "2026-08-10T14:00:00-05:00");
        assert_eq!(meetings[0].end, "2026-08-10T15:00:00-05:00");
        assert_eq!(
            meetings[0].attendees,
            vec!["Paula".to_string(), "dave@example.com".to_string()],
            "displayName wins; an attendee with only an email falls back to it, matching the Microsoft provider"
        );

        // Distinct ids, load-bearing for idempotent auto-join tracking —
        // the same property calendar.rs asserts for Graph event ids.
        let ids: Vec<&str> = meetings.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["6f1n2g3h4i5j6k7l8m9n0o1p2q", "legacy-hangout-link-only", "all-day-offsite", "no-conference-at-all"]);

        // An event with no `summary` at all must not fail deserialization.
        assert_eq!(meetings[3].subject, "(no title)");
        assert_eq!(meetings[3].join_url, None, "no conferenceData and no hangoutLink resolves to None, not an error");
    }

    /// ISC-186: the video entry point wins over the legacy field, and the
    /// legacy field is still honored when `conferenceData` is absent.
    /// Fixture 0 carries BOTH, so this also proves the phone entry point
    /// is never mistaken for a join link.
    #[test]
    fn resolves_join_url_from_conference_data_then_falls_back_to_hangout_link() {
        let meetings = parse_fixture();

        assert_eq!(
            meetings[0].join_url,
            Some("https://meet.google.com/abc-defg-hij".to_string()),
            "the conferenceData video entryPoint must win over the legacy hangoutLink"
        );
        assert_ne!(
            meetings[0].join_url,
            Some("tel:+1-555-0100".to_string()),
            "a phone entryPoint must never be handed out as a join link"
        );
        assert_eq!(
            meetings[1].join_url,
            Some("https://meet.google.com/legacy-only-xyz".to_string()),
            "with no conferenceData at all, the legacy hangoutLink is the join link"
        );
    }

    /// ISC-187: an all-day event (`start.date`, no `start.dateTime`) parses
    /// without panicking and can never reach `Decision::Trigger` — it fails
    /// closed through the shared eligibility path rather than being
    /// special-cased in `auto_join`.
    #[test]
    fn an_all_day_event_never_auto_joins_and_never_panics() {
        let meetings = parse_fixture();
        let all_day = &meetings[2];

        assert_eq!(all_day.start, "2026-08-12", "the bare all-day date flows through as-is");
        assert!(all_day.join_url.is_some(), "it genuinely has a join link — so exclusion must come from the time, not a missing URL");

        // Swept across a whole day of candidate "now" values: there is no
        // moment at which this event becomes eligible.
        let midnight = "2026-08-12T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        for hour in 0..24 {
            let now = midnight + chrono::Duration::hours(hour);
            assert_eq!(
                auto_join::eligibility(all_day, now),
                Eligibility::UnparseableStart,
                "an all-day event must fail closed at every hour of its own day"
            );
            assert_ne!(auto_join::decide(all_day, now, false, false), Decision::Trigger);
        }
    }

    /// ISC-183: the Google config is PKCE-only — no client secret is ever
    /// built into the request — and carries the two extras Google requires
    /// for a refresh token to come back at all.
    #[test]
    fn google_config_is_public_client_pkce_only_with_offline_access() {
        let config = google_config("real-google-client-id.apps.googleusercontent.com");
        assert_eq!(config.authorize_url, "https://accounts.google.com/o/oauth2/v2/auth");
        assert_eq!(config.token_url, "https://oauth2.googleapis.com/token");
        assert_eq!(config.scope, "https://www.googleapis.com/auth/calendar.events.readonly");
        assert!(config.scope.ends_with(".readonly"), "least privilege: read-only, never a write scope");
        assert_ne!(config.scope, "https://www.googleapis.com/auth/calendar.readonly", "the broader scope also grants calendar metadata/settings access this feature never needs");

        let extras: std::collections::HashMap<&str, &str> =
            config.extra_authorize_params.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(extras.get("access_type"), Some(&"offline"), "without access_type=offline Google returns no refresh_token");
        assert_eq!(extras.get("prompt"), Some(&"consent"));

        // The authorize URL the user's browser actually opens carries PKCE
        // and no secret of any kind.
        let pkce = oauth::PkcePair::generate().unwrap();
        let url = oauth::build_authorize_url(&config, &pkce, "http://localhost:53683/callback", "state-abc");
        assert!(url.contains("code_challenge_method=S256"));
        assert!(!url.contains("client_secret"), "a public client must never put a secret in the authorize URL");
    }

    /// ISC-188 / ISC-190: Google's stored tokens live under their own
    /// provider id and are fully independent of Microsoft's and Zoom's.
    /// Deliberately uses test-only provider ids rather than the real
    /// constants — writing fixture tokens into the real `google` slot would
    /// clobber a genuinely connected account, the same hazard
    /// `calendar.rs`'s own tests call out.
    #[test]
    fn google_tokens_are_stored_independently_of_the_other_providers() {
        let (g, m, z) = ("google-independence-test", "microsoft-independence-test", "zoom-independence-test");
        for p in [g, m, z] {
            oauth::delete_tokens_for_test(p).unwrap();
            assert!(oauth::load_tokens(p).unwrap().is_none(), "precondition: {p} starts disconnected");
        }

        oauth::store_tokens(
            g,
            &oauth::TokenResponse { access_token: "google-only".to_string(), refresh_token: Some("g-refresh".to_string()), expires_in: 3600 },
        )
        .unwrap();

        assert_eq!(oauth::load_tokens(g).unwrap().unwrap().access_token, "google-only");
        assert!(oauth::load_tokens(m).unwrap().is_none(), "connecting Google must not connect Microsoft");
        assert!(oauth::load_tokens(z).unwrap().is_none(), "connecting Google must not connect Zoom");

        for p in [g, m, z] {
            oauth::delete_tokens_for_test(p).unwrap();
        }
    }

    /// The real provider ids are distinct strings — a copy-paste slip that
    /// gave two providers the same id would silently make them share one
    /// Keychain slot, which is exactly the failure ISC-188/196 guard.
    #[test]
    fn provider_ids_are_distinct() {
        use crate::calendar::MICROSOFT_PROVIDER_ID;
        use crate::zoom::ZOOM_PROVIDER_ID;
        let ids = [MICROSOFT_PROVIDER_ID, GOOGLE_PROVIDER_ID, ZOOM_PROVIDER_ID];
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 3, "every provider needs its own keychain namespace, got {ids:?}");
    }
}
