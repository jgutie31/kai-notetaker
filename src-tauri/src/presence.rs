//! Microsoft Graph presence polling — the calendar-independent signal that
//! detects an *unscheduled* Teams call (Jeremiah clicking "Meet Now"), which
//! AutoJoinRecording structurally cannot see: that feature is driven entirely
//! by calendar data, and an ad-hoc call has no calendar entry to key off of.
//!
//! Every field name and endpoint shape here was verified against Microsoft's
//! own current documentation before writing this file, not assumed from
//! memory — same discipline as `calendar.rs`:
//! - Endpoint and response shape: learn.microsoft.com/graph/api/presence-get
//! - Possible `activity` values: learn.microsoft.com/graph/api/resources/presence
//! - `Presence.Read` (delegated, least privilege — NOT `Presence.Read.All`,
//!   NOT an application permission): learn.microsoft.com/graph/permissions-reference
//!
//! Real doc-conflict resolved before coding (ISC-224, see ISA Decisions
//! 2026-08-07): the `/graph/api/resources/presence` property table has the
//! possible-value lists for `activity` and `availability` SWAPPED relative to
//! the actual example JSON on the sibling `/graph/api/presence-get` page,
//! which pairs `"availability": "DoNotDisturb"` with `"activity":
//! "Presenting"`. The example payloads are ground truth: the granular states
//! (`inACall`, `inAMeeting`, `Presenting`) live in `activity`. This module
//! keys on `activity` accordingly.
//!
//! Documented limitation, not a bug if a future user hits it: `/me/presence`
//! is unsupported for personal Microsoft accounts — it requires a work or
//! school account (ISC-225).
//!
//! Like `auto_join.rs`, everything decision-shaped in here is deliberately
//! free of Tauri, HTTP, and secure storage: the classifier and both trigger
//! decisions take plain values, which is what makes the real behavior —
//! fail-closed classification, start idempotency, and marker-variant
//! isolation — unit-testable without a live Microsoft account.

use crate::auto_join::AutoStopTrigger;
use serde::Deserialize;
use thiserror::Error;

/// How often presence is polled, deliberately shorter than
/// `auto_join::POLL_INTERVAL_SECS` (60s).
///
/// The latency tradeoff is genuinely different from the calendar poller's
/// (ISC-229): a scheduled meeting's trigger window is known a full minute in
/// advance, so a 60s cadence never actually costs recorded audio. An ad-hoc
/// call has no advance signal at all — the poll interval *is* the amount of
/// the call's opening that goes uncaptured, and Jeremiah is already mid-action
/// (he just clicked Meet Now) when it should fire. 15s bounds that loss to
/// roughly the greeting, at the cost of 4 cheap Graph GETs per minute against
/// an already-authenticated connection.
pub const PRESENCE_POLL_INTERVAL_SECS: u64 = 15;

/// Whether the user is in a call right now, as far as presence can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallState {
    InCall,
    NotInCall,
}

/// Classifies a raw Graph `activity` string (ISC-228).
///
/// Two properties are load-bearing:
///
/// 1. **`inACall` and `inAMeeting` are treated as equivalent.** Which one a
///    real "Meet Now" click actually produces is a [DEFERRED-VERIFY] item
///    needing Jeremiah's own Teams client (ISC-227) — Microsoft's admin docs
///    distinguish them ("In a call" is Teams-app-state driven, "In a meeting"
///    is calendar driven), but the join lifecycle may pass through either.
///    Accepting both means that empirical uncertainty cannot make this code
///    wrong, so it doesn't block shipping.
///
/// 2. **It fails closed.** Anything unrecognized — including `activity`
///    values Microsoft adds after this was written — is `NotInCall`. The
///    failure mode of guessing wrong in that direction is a missed ad-hoc
///    recording (recoverable: Jeremiah clicks Start). Guessing the other way
///    silently hot-mics him because a future string looked call-ish.
///
/// The compare is case-insensitive. Graph returns lower-camel (`inACall`),
/// but the same docs' example payloads capitalize sibling values
/// (`"DoNotDisturb"`, `"Presenting"`), so casing is demonstrably not a stable
/// contract across this resource.
pub fn classify_activity(activity: &str) -> CallState {
    let normalized = activity.trim().to_ascii_lowercase();
    if normalized == "inacall" || normalized == "inameeting" {
        CallState::InCall
    } else {
        CallState::NotInCall
    }
}

/// Whether this poll cycle should auto-start a recording (ISC-231/ISC-232).
///
/// `already_recording` is read from `RecordingState` and is deliberately
/// trigger-agnostic: a manual recording, a calendar-triggered one, or one an
/// earlier presence cycle started all suppress a second start identically.
/// That's what makes presence-triggered start idempotent against a 15s poll
/// that will keep reporting `InCall` for the whole call.
pub fn should_start(state: CallState, already_recording: bool) -> bool {
    state == CallState::InCall && !already_recording
}

/// Whether this poll cycle should auto-stop the active recording (ISC-234).
///
/// The bidirectional half of the same signal: the poll that detects entering
/// a call also detects leaving one, so an ad-hoc recording needs no calendar
/// end-time and no new mechanism (see the FirstPrinciples pass in the ISA's
/// 2026-08-07 Decisions).
///
/// Marker-variant isolation is the point of the `AutoStopTrigger` argument. A
/// `CalendarEnd`-marked recording is *never* stopped here, even on a real
/// `NotInCall` transition — Teams presence dropping to `available` mid-call
/// (app backgrounded, network blip, or simply a Zoom meeting Teams knows
/// nothing about) must not kill a scheduled recording that the calendar-end
/// check owns. The two stop mechanisms stay independent and never
/// cross-trigger. `None` — a manual recording, unmarked — is likewise never
/// touched.
pub fn should_stop(state: CallState, active_marker: Option<&AutoStopTrigger>) -> bool {
    state == CallState::NotInCall && matches!(active_marker, Some(AutoStopTrigger::PresenceBased))
}

/// The subject recorded for a presence-triggered recording. Unlike the
/// calendar path there is no real meeting subject available — presence
/// reports a state, not an event — so this is a fixed, honest label rather
/// than a fabricated title.
pub const ADHOC_SUBJECT: &str = "Ad-hoc Teams call";

#[derive(Debug, Error)]
pub enum PresenceError {
    #[error("oauth error: {0}")]
    OAuth(#[from] crate::oauth::OAuthError),
    #[error("presence API request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("presence API returned an error response: {0}")]
    ApiError(String),
}

/// The subset of Graph's presence resource this feature reads.
///
/// `availability` is deserialized but unused for the decision — kept because
/// it costs nothing, is in every real response, and makes the eventual
/// [DEFERRED-VERIFY] "what does Meet Now actually produce?" check (ISC-227) a
/// log line rather than a code change.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PresenceResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub availability: String,
    pub activity: String,
}

impl PresenceResponse {
    pub fn call_state(&self) -> CallState {
        classify_activity(&self.activity)
    }
}

/// The real `GET /me/presence` call. Same request/error shape as
/// `calendar::list_upcoming_meetings` — blocking `reqwest`, bearer auth,
/// non-2xx bodies surfaced verbatim rather than collapsed into "it failed",
/// since a revoked-consent 403 and a transient 503 need different responses
/// from whoever reads the log.
pub fn get_presence(access_token: &str) -> Result<PresenceResponse, PresenceError> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get("https://graph.microsoft.com/v1.0/me/presence")
        .bearer_auth(access_token)
        .send()?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(PresenceError::ApiError(format!("{status}: {body}")));
    }

    Ok(resp.json()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISC-228. Both real in-call values, a real non-call value, and an
    /// unrecognized future value — the fail-closed property is the one that
    /// actually protects Jeremiah, so it's asserted explicitly rather than
    /// left implicit in an `else` branch.
    #[test]
    fn classifies_both_real_in_call_activity_values_as_in_call() {
        assert_eq!(classify_activity("inACall"), CallState::InCall);
        assert_eq!(classify_activity("inAMeeting"), CallState::InCall);
    }

    #[test]
    fn classifies_real_non_call_activity_values_as_not_in_call() {
        for activity in [
            "available",
            "busy",
            "away",
            "beRightBack",
            "doNotDisturb",
            "focusing",
            "offline",
            "presenceUnknown",
            "presenting",
        ] {
            assert_eq!(
                classify_activity(activity),
                CallState::NotInCall,
                "'{activity}' is not a call"
            );
        }
    }

    #[test]
    fn an_unrecognized_future_activity_value_fails_closed() {
        assert_eq!(classify_activity("someFutureValue"), CallState::NotInCall);
        assert_eq!(classify_activity(""), CallState::NotInCall);
        // Deliberately call-ish but not one of the two real values: proves
        // the match is exact, not a substring/prefix heuristic that a new
        // Microsoft value could trip into hot-miking him.
        assert_eq!(classify_activity("inACallLobby"), CallState::NotInCall);
        assert_eq!(classify_activity("call"), CallState::NotInCall);
    }

    /// Casing is not a stable contract on this resource (see
    /// `classify_activity`'s doc comment) — a capitalized `InACall` must not
    /// silently become "not in a call".
    #[test]
    fn classification_is_case_and_whitespace_insensitive() {
        assert_eq!(classify_activity("InACall"), CallState::InCall);
        assert_eq!(classify_activity("INAMEETING"), CallState::InCall);
        assert_eq!(classify_activity("  inACall  "), CallState::InCall);
    }

    /// ISC-224: a real, Graph-shaped payload through the real deserializer —
    /// proves the field names are right, not just that the struct compiles.
    /// This is the exact pairing from Microsoft's own example response that
    /// resolved the swapped-table doc conflict.
    #[test]
    fn parses_a_real_shaped_presence_response_and_reads_activity_not_availability() {
        let raw = r#"{
            "id": "fa8bf3dc-eca7-46b7-bad1-db199b62afc3",
            "availability": "DoNotDisturb",
            "activity": "Presenting"
        }"#;
        let presence: PresenceResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(presence.availability, "DoNotDisturb");
        assert_eq!(presence.activity, "Presenting");
        assert_eq!(presence.call_state(), CallState::NotInCall);

        let in_a_call = r#"{
            "id": "fa8bf3dc-eca7-46b7-bad1-db199b62afc3",
            "availability": "Busy",
            "activity": "inACall"
        }"#;
        let presence: PresenceResponse = serde_json::from_str(in_a_call).unwrap();
        // The whole point of ISC-224: `availability` says only "Busy", which
        // is indistinguishable from a focus block. `activity` carries the
        // signal this feature exists to read.
        assert_eq!(presence.availability, "Busy");
        assert_eq!(presence.call_state(), CallState::InCall);
    }

    /// ISC-231.
    #[test]
    fn in_call_with_nothing_recording_starts_a_recording() {
        assert!(should_start(CallState::InCall, false));
    }

    /// ISC-232 — mirrors `auto_join`'s `already_recording` idempotency
    /// pattern. Trigger-agnostic on purpose: this same `true` covers a
    /// manual recording, a calendar-triggered one, and one an earlier
    /// presence cycle started 15 seconds ago.
    #[test]
    fn in_call_with_a_recording_already_running_starts_nothing() {
        assert!(!should_start(CallState::InCall, true));
    }

    #[test]
    fn not_in_call_never_starts_a_recording() {
        assert!(!should_start(CallState::NotInCall, false));
        assert!(!should_start(CallState::NotInCall, true));
    }

    /// ISC-234, the start of the bidirectional pair.
    #[test]
    fn leaving_a_call_stops_a_presence_triggered_recording() {
        assert!(should_stop(CallState::NotInCall, Some(&AutoStopTrigger::PresenceBased)));
    }

    /// ISC-233/ISC-234's isolation requirement — the test that would catch
    /// the real bug this enum exists to prevent: a Teams presence blip
    /// killing a scheduled, calendar-triggered recording that the
    /// calendar-end check owns.
    #[test]
    fn leaving_a_call_never_stops_a_calendar_triggered_recording() {
        let calendar_marker = AutoStopTrigger::CalendarEnd(
            chrono::DateTime::parse_from_rfc3339("2026-08-10T15:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        );
        assert!(!should_stop(CallState::NotInCall, Some(&calendar_marker)));
    }

    /// A manually-started recording carries no marker at all and must be
    /// equally untouchable from this path.
    #[test]
    fn leaving_a_call_never_stops_an_unmarked_manual_recording() {
        assert!(!should_stop(CallState::NotInCall, None));
    }

    /// Still in the call — nothing stops, whatever the marker says.
    #[test]
    fn staying_in_a_call_stops_nothing() {
        assert!(!should_stop(CallState::InCall, Some(&AutoStopTrigger::PresenceBased)));
        assert!(!should_stop(CallState::InCall, None));
    }

    /// The full ad-hoc lifecycle as the poll loop actually sees it, cycle by
    /// cycle — proves start and stop compose correctly across the repeated
    /// polls of one real call, which the individual decision tests above
    /// can't show in isolation.
    #[test]
    fn a_full_adhoc_call_lifecycle_starts_once_and_stops_once() {
        let mut recording = false;
        let mut marker: Option<AutoStopTrigger> = None;
        let mut starts = 0;
        let mut stops = 0;

        // available → inACall → inACall → inACall → available
        for activity in ["available", "inACall", "inACall", "inACall", "available"] {
            let state = classify_activity(activity);
            if should_start(state, recording) {
                recording = true;
                marker = Some(AutoStopTrigger::PresenceBased);
                starts += 1;
            } else if should_stop(state, marker.as_ref()) {
                recording = false;
                marker = None;
                stops += 1;
            }
        }

        assert_eq!(starts, 1, "exactly one start across three in-call polls");
        assert_eq!(stops, 1, "exactly one stop when the call ends");
        assert!(!recording);
        assert!(marker.is_none(), "the marker is cleared so a later poll can't re-stop");
    }

    /// ISC-230's decision half: the presence poll must stand down entirely
    /// unless Microsoft is genuinely connected — both a client id and stored
    /// tokens. Reuses the exact same predicate the calendar poller uses
    /// rather than a second, drift-prone copy of the rule, mirroring
    /// `auto_join`'s own `a_client_id_without_tokens_is_not_an_active_provider`.
    #[test]
    fn presence_polling_requires_a_fully_connected_microsoft_provider() {
        use crate::auto_join::ProviderConnection;
        let connected = ProviderConnection {
            provider_id: "microsoft",
            client_id: Some("real-client-id".to_string()),
            has_tokens: true,
        };
        assert!(connected.is_active());

        let client_id_but_abandoned_consent = ProviderConnection {
            provider_id: "microsoft",
            client_id: Some("real-client-id".to_string()),
            has_tokens: false,
        };
        assert!(!client_id_but_abandoned_consent.is_active());

        let tokens_but_no_client_id = ProviderConnection {
            provider_id: "microsoft",
            client_id: None,
            has_tokens: true,
        };
        assert!(!tokens_but_no_client_id.is_active());

        let nothing = ProviderConnection { provider_id: "microsoft", client_id: None, has_tokens: false };
        assert!(!nothing.is_active());
    }
}
