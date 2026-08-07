//! Zoom provider — the third meetings provider, built on the exact same
//! generic OAuth engine (`oauth.rs`) as Microsoft (`calendar.rs`) and
//! Google (`google.rs`). No second PKCE implementation (ISC-191).
//!
//! Endpoints, scope, and app type verified against Zoom's own current
//! documentation before writing this file:
//! - Authorize `https://zoom.us/oauth/authorize`, token
//!   `https://zoom.us/oauth/token`, and PKCE-without-a-secret:
//!   developers.zoom.us/docs/integrations/oauth/ — "Use PKCE when you
//!   don't have a backend server for user authorization... Zoom offers a
//!   separate public client ID in the authorization request that doesn't
//!   require an associated client secret." Enabled by toggling **Use
//!   Public Client OAuth** on in the app's Basic Information → App
//!   Credentials. This CORRECTS an earlier assumption recorded in this
//!   project's ISA that Zoom always requires a client secret (ISC-192).
//! - `meeting:read:list_meetings` — the non-admin granular scope (the
//!   `:admin` variant exists and is deliberately NOT used, since this app
//!   only ever reads the signed-in user's own meetings):
//!   developers.zoom.us/docs/integrations/oauth-scopes-granular/
//! - `GET /users/me/meetings` response shape (`meetings[]` with `id`,
//!   `topic`, `start_time`, `duration` in minutes, `timezone`,
//!   `join_url`): Zoom's Meetings API reference. Note that Zoom's API
//!   reference pages render client-side and were not re-fetchable as text
//!   this session — this shape comes from the ISA's already-verified
//!   record (ISC-194), and the deserializer below is written defensively
//!   (every non-identity field optional) so a shape drift degrades into
//!   "this meeting isn't eligible" rather than a failed parse of the
//!   whole response.

use crate::calendar::{CalendarError, UpcomingMeeting};
use crate::oauth::{self, OAuthProviderConfig};
use serde::Deserialize;

pub const ZOOM_PROVIDER_ID: &str = "zoom";

/// No `client_secret` — this app registers as a Zoom **public client**
/// (ISC-192), matching the Microsoft and Google providers' PKCE-only
/// posture. `extra_authorize_params` is empty: Zoom needs none of the
/// Google-style offline-access extras.
fn zoom_config(client_id: &str) -> OAuthProviderConfig {
    OAuthProviderConfig {
        authorize_url: "https://zoom.us/oauth/authorize".to_string(),
        token_url: "https://zoom.us/oauth/token".to_string(),
        client_id: client_id.to_string(),
        scope: "meeting:read:list_meetings".to_string(),
        extra_authorize_params: vec![],
    }
}

/// Zoom returns a meeting `id` as a JSON number for ordinary meetings, but
/// a string in some responses (and for Personal Meeting IDs). Accepting
/// both and normalizing to `String` means a numeric id can't fail the
/// whole response's deserialization — and `UpcomingMeeting.id` is a
/// `String` for every provider anyway, since it's the idempotency key the
/// `auto_join_log` stores.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ZoomMeetingId {
    Number(i64),
    Text(String),
}

impl std::fmt::Display for ZoomMeetingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZoomMeetingId::Number(n) => write!(f, "{n}"),
            ZoomMeetingId::Text(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ZoomMeeting {
    id: ZoomMeetingId,
    #[serde(default)]
    topic: Option<String>,
    /// Absent for recurring meetings with no fixed time (Zoom meeting
    /// `type` 3) and for some instant meetings — a real case, not a
    /// hypothetical, which is why this is `Option` rather than `String`.
    #[serde(default)]
    start_time: Option<String>,
    /// Scheduled length in **minutes**. Zoom's list response carries no
    /// `end` field at all; this is the only thing an end time can be
    /// derived from (ISC-195).
    #[serde(default)]
    duration: Option<i64>,
    #[serde(default)]
    join_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZoomMeetingList {
    #[serde(default)]
    meetings: Vec<ZoomMeeting>,
}

/// Zoom documents `start_time` as UTC RFC3339 (`yyyy-MM-ddTHH:mm:ssZ`).
/// Parsed here rather than via `auto_join`'s Graph-shaped helper so this
/// module owns its own provider's format contract.
fn parse_zoom_start(raw: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Utc))
}

impl From<ZoomMeeting> for UpcomingMeeting {
    fn from(m: ZoomMeeting) -> Self {
        let start_raw = m.start_time.clone().unwrap_or_default();

        // ISC-195: Zoom has no `end` field — compute it as
        // `start_time + duration minutes` here, at the edge, so
        // `auto_join::eligibility()` (which reads `meeting.end`) works
        // identically for Zoom with zero Zoom-specific branching inside
        // `auto_join` itself.
        //
        // When either input is missing (a no-fixed-time recurring meeting,
        // or a meeting with no duration), `end` is left empty on purpose:
        // `eligibility()` already falls back to its fixed post-start grace
        // for an unparseable `end`, and a start we can't parse fails
        // closed. Inventing a default duration here would be guessing at
        // how long a real call runs.
        let end_raw = match (parse_zoom_start(&start_raw), m.duration) {
            (Some(start), Some(minutes)) => (start + chrono::Duration::minutes(minutes)).to_rfc3339(),
            _ => String::new(),
        };

        UpcomingMeeting {
            id: m.id.to_string(),
            subject: m.topic.unwrap_or_else(|| "(no topic)".to_string()),
            start: start_raw,
            end: end_raw,
            // Zoom's meeting-list response carries no attendee/registrant
            // data at all — an empty Vec is the honest representation, not
            // a gap to be filled by a second API call this scope can't make.
            attendees: Vec::new(),
            join_url: m.join_url,
        }
    }
}

/// Runs the full interactive consent flow against Zoom's real sign-in
/// page. Structurally identical to `calendar::connect_microsoft` and
/// `google::connect_google`.
pub fn connect_zoom(client_id: &str, port: u16) -> Result<(), CalendarError> {
    let config = zoom_config(client_id);
    let pkce = oauth::PkcePair::generate()?;
    let expected_state = oauth::generate_state()?;
    let redirect_uri = format!("http://localhost:{port}/callback");
    let authorize_url = oauth::build_authorize_url(&config, &pkce, &redirect_uri, &expected_state);

    let _ = open::that(&authorize_url);

    let (code, returned_state) = oauth::await_authorization_code(port, std::time::Duration::from_secs(180))?;
    if returned_state.as_deref() != Some(expected_state.as_str()) {
        return Err(CalendarError::OAuth(oauth::OAuthError::StateMismatch));
    }

    let tokens = oauth::exchange_code_for_tokens(&config, &code, &pkce.verifier, &redirect_uri)?;
    oauth::store_tokens(ZOOM_PROVIDER_ID, &tokens)?;
    Ok(())
}

pub fn is_zoom_connected() -> Result<bool, CalendarError> {
    Ok(oauth::load_tokens(ZOOM_PROVIDER_ID)?.is_some())
}

/// The signed-in user's own meetings, as the same shared `UpcomingMeeting`
/// every other provider returns.
///
/// Deliberately takes no `hours_ahead`: unlike Graph's `calendarView` and
/// Google's `events.list`, Zoom's list endpoint has no time-window
/// parameters at all. Its default `type` (`scheduled`) already returns
/// unexpired, live, and upcoming scheduled meetings, and
/// `auto_join::eligibility()` applies the real trigger window afterwards —
/// so accepting an `hours_ahead` argument here and ignoring it would be a
/// lie in the signature.
pub fn list_upcoming_meetings(client_id: &str) -> Result<Vec<UpcomingMeeting>, CalendarError> {
    let config = zoom_config(client_id);
    let access_token = oauth::get_valid_access_token(ZOOM_PROVIDER_ID, &config)?;

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get("https://api.zoom.us/v2/users/me/meetings")
        .bearer_auth(&access_token)
        .send()?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(CalendarError::ApiError(format!("{status}: {body}")));
    }

    let list: ZoomMeetingList = resp.json()?;
    Ok(list.meetings.into_iter().map(UpcomingMeeting::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_join::{self, Decision, Eligibility};
    use chrono::{DateTime, Utc};

    /// A real, Zoom-List-Meetings-shaped response using the documented
    /// field names. Includes the numeric-id case (Zoom's normal shape), a
    /// string-id case, and a no-fixed-time recurring meeting with no
    /// `start_time` — all three are real shapes this endpoint returns.
    const REAL_SHAPED_ZOOM_RESPONSE: &str = r#"{
        "page_size": 30,
        "total_records": 4,
        "next_page_token": "",
        "meetings": [
            {
                "uuid": "aDYlohsHRtCd4ii1uC2+hA==",
                "id": 92345678901,
                "host_id": "z8dsdsssssgSSgs",
                "topic": "Smithville PCI-DSS Evidence Review",
                "type": 2,
                "start_time": "2026-08-10T14:00:00Z",
                "duration": 30,
                "timezone": "America/Chicago",
                "created_at": "2026-08-01T12:00:00Z",
                "join_url": "https://zoom.us/j/92345678901"
            },
            {
                "id": "84512345678",
                "topic": "Nesta strategy call",
                "type": 2,
                "start_time": "2026-08-11T16:30:00Z",
                "duration": 90,
                "timezone": "America/Chicago",
                "join_url": "https://zoom.us/j/84512345678"
            },
            {
                "id": 71234567890,
                "topic": "Recurring, no fixed time",
                "type": 3,
                "timezone": "America/Chicago",
                "join_url": "https://zoom.us/j/71234567890"
            },
            {
                "id": 60000000001,
                "type": 2,
                "start_time": "2026-08-12T09:00:00Z",
                "duration": 45
            }
        ]
    }"#;

    fn parse_fixture() -> Vec<UpcomingMeeting> {
        let list: ZoomMeetingList = serde_json::from_str(REAL_SHAPED_ZOOM_RESPONSE).unwrap();
        list.meetings.into_iter().map(UpcomingMeeting::from).collect()
    }

    /// ISC-194: the documented Zoom field names deserialize into the shared
    /// `UpcomingMeeting`, including a numeric `id` normalized to a string
    /// (the idempotency key `auto_join_log` stores).
    #[test]
    fn parses_a_real_zoom_shaped_response_into_upcoming_meetings() {
        let meetings = parse_fixture();
        assert_eq!(meetings.len(), 4);

        assert_eq!(meetings[0].id, "92345678901", "a numeric Zoom id must normalize to its exact decimal string");
        assert_eq!(meetings[1].id, "84512345678", "a string Zoom id passes through unchanged");
        assert_eq!(meetings[0].subject, "Smithville PCI-DSS Evidence Review");
        assert_eq!(meetings[0].start, "2026-08-10T14:00:00Z");
        assert_eq!(meetings[0].join_url, Some("https://zoom.us/j/92345678901".to_string()));

        // Zoom's list response genuinely carries no attendee data — an
        // empty Vec is the honest answer, asserted so a future change that
        // silently invents attendees fails here.
        assert_eq!(meetings[0].attendees, Vec::<String>::new());

        // A meeting with no `topic` must not fail the whole parse.
        assert_eq!(meetings[3].subject, "(no topic)");
        assert_eq!(meetings[3].join_url, None);

        let ids: std::collections::HashSet<&str> = meetings.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids.len(), 4, "distinct meetings must get distinct idempotency keys");
    }

    /// ISC-195: `end` is exactly `start_time + duration` minutes — the
    /// single most load-bearing Zoom-specific transformation, since
    /// `auto_join::eligibility()` closes its trigger window at `end` and
    /// ISC-181's auto-stop fires off it.
    #[test]
    fn end_is_computed_as_start_plus_duration_minutes() {
        let meetings = parse_fixture();

        for (index, expected_minutes) in [(0usize, 30i64), (1, 90), (3, 45)] {
            let m = &meetings[index];
            let start = DateTime::parse_from_rfc3339(&m.start).unwrap().with_timezone(&Utc);
            let end = DateTime::parse_from_rfc3339(&m.end)
                .unwrap_or_else(|e| panic!("meeting {index} produced an unparseable end '{}': {e}", m.end))
                .with_timezone(&Utc);
            assert_eq!(
                (end - start).num_minutes(),
                expected_minutes,
                "meeting {index}: end must be exactly {expected_minutes} minutes after start"
            );
        }

        // Asserted as a concrete instant, not just an offset: a unit slip
        // (treating `duration` as seconds or hours instead of minutes)
        // fails loudly right here.
        let first_end = DateTime::parse_from_rfc3339(&meetings[0].end).unwrap().with_timezone(&Utc);
        assert_eq!(first_end, "2026-08-10T14:30:00Z".parse::<DateTime<Utc>>().unwrap());
    }

    /// The shared, provider-agnostic `auto_join` logic works on a
    /// Zoom-derived meeting with no Zoom-specific branch: it triggers
    /// inside the window and closes at the computed end.
    #[test]
    fn a_zoom_meeting_flows_through_the_shared_eligibility_logic() {
        let meetings = parse_fixture();
        let m = &meetings[0]; // 14:00Z, 30 minutes

        let just_before = "2026-08-10T13:59:30Z".parse::<DateTime<Utc>>().unwrap();
        assert_eq!(auto_join::eligibility(m, just_before), Eligibility::Eligible);
        assert_eq!(auto_join::decide(m, just_before, false, false), Decision::Trigger);

        let mid_meeting = "2026-08-10T14:20:00Z".parse::<DateTime<Utc>>().unwrap();
        assert_eq!(auto_join::eligibility(m, mid_meeting), Eligibility::Eligible);

        // The window closes at the computed end, driven purely by
        // `duration` — this is what proves ISC-195's computation is
        // actually load-bearing downstream, not just a parsed field.
        let one_second_past_end = "2026-08-10T14:30:01Z".parse::<DateTime<Utc>>().unwrap();
        assert_eq!(auto_join::eligibility(m, one_second_past_end), Eligibility::WindowPassed);

        // And ISC-181's auto-stop fires off that same computed end.
        let end = "2026-08-10T14:30:00Z".parse::<DateTime<Utc>>().unwrap();
        assert!(!auto_join::should_auto_stop(end, mid_meeting));
        assert!(auto_join::should_auto_stop(end, one_second_past_end));
    }

    /// A recurring-no-fixed-time meeting (Zoom type 3, no `start_time`)
    /// parses without panicking and can never trigger — fail closed, the
    /// same way Google's all-day events do.
    #[test]
    fn a_recurring_meeting_with_no_fixed_time_never_auto_joins() {
        let meetings = parse_fixture();
        let no_fixed_time = &meetings[2];

        assert_eq!(no_fixed_time.start, "");
        assert_eq!(no_fixed_time.end, "");
        assert!(no_fixed_time.join_url.is_some(), "it has a real join link — exclusion must come from the missing time");

        let now = "2026-08-10T14:00:00Z".parse::<DateTime<Utc>>().unwrap();
        for hours in 0..48 {
            let t = now + chrono::Duration::hours(hours);
            assert_eq!(auto_join::eligibility(no_fixed_time, t), Eligibility::UnparseableStart);
            assert_ne!(auto_join::decide(no_fixed_time, t, false, false), Decision::Trigger);
        }
    }

    /// ISC-192 / ISC-193: public-client PKCE only (no secret anywhere) and
    /// the non-admin granular scope.
    #[test]
    fn zoom_config_is_public_client_pkce_only_with_the_non_admin_scope() {
        let config = zoom_config("real-zoom-public-client-id");
        assert_eq!(config.authorize_url, "https://zoom.us/oauth/authorize");
        assert_eq!(config.token_url, "https://zoom.us/oauth/token");
        assert_eq!(config.scope, "meeting:read:list_meetings");
        assert!(!config.scope.ends_with(":admin"), "least privilege: the admin scope variant must never be requested");
        assert!(config.extra_authorize_params.is_empty(), "Zoom needs none of Google's offline-access extras");

        let pkce = oauth::PkcePair::generate().unwrap();
        let url = oauth::build_authorize_url(&config, &pkce, "http://localhost:53684/callback", "state-xyz");
        assert!(url.starts_with("https://zoom.us/oauth/authorize?"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(!url.contains("client_secret"), "a Zoom public-client app must never send a secret");
    }

    /// ISC-196: Zoom's tokens live under their own provider id, independent
    /// of Microsoft's and Google's. Test-only ids, never the real slots.
    #[test]
    fn zoom_tokens_are_stored_independently_of_the_other_providers() {
        let (z, m, g) = ("zoom-indep-test", "microsoft-indep-test", "google-indep-test");
        for p in [z, m, g] {
            oauth::delete_tokens_for_test(p).unwrap();
        }

        oauth::store_tokens(
            z,
            &oauth::TokenResponse { access_token: "zoom-only".to_string(), refresh_token: None, expires_in: 3600 },
        )
        .unwrap();

        assert_eq!(oauth::load_tokens(z).unwrap().unwrap().access_token, "zoom-only");
        assert!(oauth::load_tokens(m).unwrap().is_none(), "connecting Zoom must not connect Microsoft");
        assert!(oauth::load_tokens(g).unwrap().is_none(), "connecting Zoom must not connect Google");

        for p in [z, m, g] {
            oauth::delete_tokens_for_test(p).unwrap();
        }
    }
}
