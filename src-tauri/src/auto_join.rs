//! The poll/trigger-decision core of AutoJoinRecording — the background
//! feature that replaces the old, now-disabled `meeting-watcher` cron.
//!
//! Honest naming note (from this feature's FirstPrinciples pass): nothing
//! here makes a bot "join" a call. The real mechanism is auto-*record*,
//! with opening the meeting's join link as a convenience side effect. No
//! Zoom/Teams bot API is involved, and no lobby/waiting-room prompt is
//! bypassed.
//!
//! Everything in this module is deliberately free of Tauri, SQLite, and
//! HTTP: the decision logic takes plain values and the fetch step takes a
//! list of closures. That's what makes the real behavior — idempotency,
//! the eligibility window, the toggle, error resilience — unit-testable
//! without a live Microsoft account, a database, or a running app.

use crate::calendar::UpcomingMeeting;
use chrono::{DateTime, NaiveDateTime, Utc};

/// How long before a meeting's start time it becomes eligible to trigger.
/// Deliberately equal to the poll interval (see `POLL_INTERVAL_SECS`) so
/// no meeting can open and close its window between two consecutive polls
/// — the antecedent ISC-178 depends on.
pub const PRE_START_WINDOW_SECS: i64 = 60;

/// Fallback grace period after `start` when a meeting's `end` can't be
/// parsed. Real end-time-aware eligibility (below) is preferred — this
/// only covers the degenerate case of a malformed `end` field.
pub const POST_START_GRACE_SECS: i64 = 5 * 60;

/// The poller's interval. Must stay <= `PRE_START_WINDOW_SECS` — see
/// `no_meeting_can_slip_between_two_consecutive_polls`, which fails if
/// this invariant is ever broken by a "harmless" tuning change.
pub const POLL_INTERVAL_SECS: u64 = 60;

/// How far ahead each poll asks providers for meetings. Wider than the
/// trigger window on purpose: a meeting must already be in hand before
/// its 60-second window opens, and re-fetching a day's worth is cheap.
pub const FETCH_WINDOW_HOURS: i64 = 24;

/// Why a meeting is (or isn't) inside its trigger window right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eligibility {
    /// Real, upcoming, but its window hasn't opened yet.
    NotYet,
    /// Inside the window — `[start - 60s, start + 5m]`.
    Eligible,
    /// Started long enough ago that auto-starting now would be wrong.
    WindowPassed,
    /// The provider sent a start time this module can't parse. Treated as
    /// ineligible (fail closed) rather than guessed at.
    UnparseableStart,
}

/// What the poller should do about one meeting on one cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Open the join link and start a recording.
    Trigger,
    /// Already handled on an earlier cycle (or an earlier app run) — do
    /// nothing at all. This is the idempotency guard (ISC-161).
    SkipAlreadyTriggered,
    /// A recording is already running. Still opens the join link (the
    /// user clearly wants to be in this meeting) but never starts a
    /// second, competing capture (ISC-163).
    SkipAlreadyRecording,
    /// No Graph-native `onlineMeeting.joinUrl`. v1 deliberately does not
    /// text-scrape the event body or location for a bare meeting URL
    /// (ISC-162/ISC-175).
    SkipNoJoinUrl,
    /// Outside the trigger window, for the carried reason.
    SkipNotEligible(Eligibility),
}

impl Decision {
    /// True when the join link should be handed to the OS opener.
    pub fn opens_join_url(&self) -> bool {
        matches!(self, Decision::Trigger | Decision::SkipAlreadyRecording)
    }

    /// True only for `Trigger` — a second recording is never started on
    /// top of a live one.
    pub fn starts_recording(&self) -> bool {
        matches!(self, Decision::Trigger)
    }

    /// True whenever the poller acted at all. `SkipAlreadyRecording` logs
    /// too: it opened a browser tab, so without a log row it would open
    /// another one every 60 seconds for the whole window.
    pub fn should_log(&self) -> bool {
        self.opens_join_url()
    }
}

/// Microsoft Graph returns `dateTime` values in UTC unless the caller
/// sends a `Prefer: outlook.timezone` header — this app never sends one,
/// so the naive, offset-less strings Graph returns (e.g.
/// `2026-08-10T14:00:00.0000000`) are UTC. Offset-bearing strings are
/// also accepted so a future provider that does send an offset parses
/// correctly rather than silently landing in the wrong hour.
pub fn parse_graph_utc(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Some(dt.with_timezone(&Utc));
    }
    raw.parse::<NaiveDateTime>().ok().map(|naive| naive.and_utc())
}

/// Where `meeting` sits relative to its trigger window at `now`.
///
/// Refined after an Advisor commitment-boundary review (2026-08-06, see
/// ISA Decisions): the window closes at the meeting's real `end` time, not
/// a fixed 5-minute grace — the manual Start Recording path has no such
/// cutoff at all, and a narrower auto-trigger window than the manual one
/// would only produce "why didn't it record?" reports for something that
/// looks, to Jeremiah, like the exact same feature. The fixed grace period
/// still applies as a fallback for the rare case `end` doesn't parse.
/// Whether an auto-started recording, ending at `marker_end`, should be
/// auto-stopped at `now` (ISC-181). A one-line comparison, but pulled out
/// as its own named, unit-tested function per this module's own
/// convention — the same reasoning `eligibility()`'s boundary tests
/// exist for: a `>` vs `>=` slip here is exactly the kind of bug that
/// leaves a recording running past a meeting Jeremiah has already left.
pub fn should_auto_stop(marker_end: DateTime<Utc>, now: DateTime<Utc>) -> bool {
    now > marker_end
}

pub fn eligibility(meeting: &UpcomingMeeting, now: DateTime<Utc>) -> Eligibility {
    let Some(start) = parse_graph_utc(&meeting.start) else {
        return Eligibility::UnparseableStart;
    };
    let seconds_until_start = (start - now).num_seconds();

    if seconds_until_start > PRE_START_WINDOW_SECS {
        return Eligibility::NotYet;
    }

    let window_closes_at = match parse_graph_utc(&meeting.end) {
        Some(end) if end > start => end,
        _ => start + chrono::Duration::seconds(POST_START_GRACE_SECS),
    };

    if now > window_closes_at {
        Eligibility::WindowPassed
    } else {
        Eligibility::Eligible
    }
}

/// The whole trigger decision for one meeting, as a pure function of
/// plain values. No DB, no network, no Tauri — every anti-criterion this
/// feature has (ISC-161/162/163/168) is decided right here and tested
/// directly.
///
/// Check order is load-bearing:
/// 1. No join URL — nothing to open, nothing to be in.
/// 2. Already triggered — must beat every other branch, or a meeting
///    handled on cycle N gets re-opened on cycle N+1.
/// 3. Outside the window.
/// 4. Already recording — open the link, don't double-capture.
pub fn decide(
    meeting: &UpcomingMeeting,
    now: DateTime<Utc>,
    already_recording: bool,
    already_triggered: bool,
) -> Decision {
    if meeting.join_url.as_deref().unwrap_or("").trim().is_empty() {
        return Decision::SkipNoJoinUrl;
    }
    if already_triggered {
        return Decision::SkipAlreadyTriggered;
    }
    match eligibility(meeting, now) {
        Eligibility::Eligible => {}
        other => return Decision::SkipNotEligible(other),
    }
    if already_recording {
        return Decision::SkipAlreadyRecording;
    }
    Decision::Trigger
}

/// One registered calendar provider's "give me the upcoming meetings"
/// call, already bound to whatever credentials it needs.
///
/// This exists because of a real design correction (ISC-172.1): the poll
/// cycle must not name `calendar::list_upcoming_meetings` inline. Today
/// there is exactly one fetcher — the Microsoft one, built in `lib.rs`
/// from the stored client ID — but adding Google or Zoom later has to be
/// "register a second closure," not "rewrite the poll cycle."
pub type MeetingFetcher = Box<dyn Fn() -> Result<Vec<UpcomingMeeting>, String> + Send + Sync>;

/// One provider's stored connection state, as read out of secure storage.
///
/// Exists so the "which providers are actually connected?" decision is a
/// pure function of plain values (testable against a 0/1/2/3 fixture
/// matrix) rather than something only observable by writing fake tokens
/// into the real `microsoft`/`google`/`zoom` Keychain slots — which would
/// clobber Jeremiah's genuinely connected accounts, the exact hazard
/// `calendar.rs`'s own tests already call out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConnection {
    pub provider_id: &'static str,
    pub client_id: Option<String>,
    pub has_tokens: bool,
}

impl ProviderConnection {
    /// ISC-199: a stored client id alone is NOT "connected" — that's a
    /// user who opened the connect flow and never finished consent. Both
    /// halves must be present, and a blank/whitespace client id counts as
    /// absent (it would only produce a guaranteed-failing API call every
    /// cycle).
    pub fn is_active(&self) -> bool {
        self.has_tokens && self.client_id.as_deref().map(|id| !id.trim().is_empty()).unwrap_or(false)
    }
}

/// ISC-198: exactly the genuinely-connected providers, each paired with
/// its client id — zero, one, two, or three of them, in the given order.
/// The caller turns each entry into that provider's own fetcher closure;
/// this function never names a provider, which is what keeps "Jeremiah
/// connected only Outlook and Zoom" from being a code change.
pub fn active_provider_client_ids(states: &[ProviderConnection]) -> Vec<(&'static str, String)> {
    states
        .iter()
        .filter(|s| s.is_active())
        .filter_map(|s| s.client_id.clone().map(|id| (s.provider_id, id)))
        .collect()
}

/// Runs every registered fetcher and concatenates the results. A fetcher
/// that fails (expired token, no network, provider outage) is logged and
/// skipped — one broken provider must never blank out the others, and it
/// must never panic the poller thread (ISC-169).
pub fn collect_meetings(fetchers: &[MeetingFetcher]) -> Vec<UpcomingMeeting> {
    let mut all = Vec::new();
    for (index, fetch) in fetchers.iter().enumerate() {
        match fetch() {
            Ok(meetings) => all.extend(meetings),
            Err(e) => eprintln!("auto-join: calendar fetcher {index} failed this cycle (will retry next cycle): {e}"),
        }
    }
    all
}

/// One full poll cycle's decisions, provider-agnostic end to end.
///
/// Returns every meeting it looked at paired with what should happen to
/// it, so the caller (`lib.rs`) owns all the side effects — opening URLs,
/// starting recordings, writing log rows — and this function stays pure
/// enough to test.
///
/// `enabled` is passed in (rather than read here) because it must be
/// re-read from storage on every single cycle, never cached at startup:
/// that's what makes flipping the toggle off take effect within 60
/// seconds (ISC-166).
///
/// `already_triggered` is a closure over the persisted `auto_join_log`.
/// If it errors, the cycle ends early with no decisions at all — without
/// a trustworthy idempotency answer, doing nothing is the only safe
/// behavior, since the alternative is re-opening and re-recording a
/// meeting that was already handled.
pub fn poll_cycle(
    enabled: bool,
    fetchers: &[MeetingFetcher],
    now: DateTime<Utc>,
    already_recording: bool,
    already_triggered: &dyn Fn(&str) -> Result<bool, String>,
) -> Vec<(UpcomingMeeting, Decision)> {
    if !enabled {
        return Vec::new();
    }

    let mut decisions = Vec::new();
    for meeting in collect_meetings(fetchers) {
        let triggered = match already_triggered(&meeting.id) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("auto-join: could not read the auto-join log, ending this cycle early: {e}");
                return Vec::new();
            }
        };
        let decision = decide(&meeting, now, already_recording, triggered);
        decisions.push((meeting, decision));
    }
    decisions
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISC-181: auto-stop fires strictly after the meeting's real end,
    /// never at or before it — a meeting still in its final second must
    /// not get cut off.
    #[test]
    fn should_auto_stop_fires_strictly_after_the_meetings_end() {
        let end = "2026-08-10T15:00:00Z".parse::<DateTime<Utc>>().unwrap();
        assert!(!should_auto_stop(end, end - chrono::Duration::seconds(1)), "one second before end: not yet");
        assert!(!should_auto_stop(end, end), "exactly at end: not yet — still their last second");
        assert!(should_auto_stop(end, end + chrono::Duration::seconds(1)), "one second past end: stop it");
    }

    /// Graph's real, offset-less UTC shape: seven fractional-second
    /// digits, exactly what the live API returned in this project's own
    /// verified `calendar.rs` fixture. Built by hand rather than with a
    /// chrono format string because chrono only offers fixed-width
    /// `%.3f`/`%.6f`/`%.9f` — there is no `%.7f`, and asking for one
    /// panics at runtime (found by this test suite doing exactly that).
    fn graph_timestamp(dt: DateTime<Utc>) -> String {
        format!("{}.0000000", dt.format("%Y-%m-%dT%H:%M:%S"))
    }

    /// A fixed "now" with no sub-second component, rather than
    /// `Utc::now()`. Not just determinism hygiene — a real failure this
    /// suite caught: `graph_timestamp` truncates to whole seconds (as
    /// Graph's own values effectively are), so with a wall-clock `now`
    /// carrying ~0.7s of fraction, a meeting built as "61 seconds out"
    /// actually lands 60.3s out and reads as Eligible. The boundary
    /// assertions are only meaningful against an exact second.
    fn fixed_now() -> DateTime<Utc> {
        "2026-08-10T13:59:00Z".parse::<DateTime<Utc>>().unwrap()
    }

    fn meeting(id: &str, start: DateTime<Utc>, join_url: Option<&str>) -> UpcomingMeeting {
        UpcomingMeeting {
            id: id.to_string(),
            subject: "Smithville PCI-DSS Scoping Call".to_string(),
            start: graph_timestamp(start),
            end: graph_timestamp(start + chrono::Duration::hours(1)),
            attendees: vec!["Nesta".to_string(), "dave@example.com".to_string()],
            join_url: join_url.map(str::to_string),
        }
    }

    fn never_triggered(_: &str) -> Result<bool, String> {
        Ok(false)
    }

    #[test]
    fn parses_the_offsetless_utc_shape_graph_actually_returns() {
        let parsed = parse_graph_utc("2026-08-10T14:00:00.0000000").unwrap();
        assert_eq!(parsed.to_rfc3339(), "2026-08-10T14:00:00+00:00");
        // Offset-bearing input still works, for a future provider.
        assert_eq!(
            parse_graph_utc("2026-08-10T10:00:00-04:00").unwrap().to_rfc3339(),
            "2026-08-10T14:00:00+00:00"
        );
        assert!(parse_graph_utc("not a timestamp").is_none());
    }

    /// ISC-168: the window is `[start - 60s, end]` (falling back to a
    /// fixed 5-minute grace only when `end` can't be parsed), not the
    /// entire multi-day range `list_upcoming_meetings` returns — a meeting
    /// three days out is fetched for the Calendar tab without auto-joining
    /// it.
    #[test]
    fn eligibility_window_boundaries() {
        let now = fixed_now();
        let url = Some("https://teams.microsoft.com/l/meetup-join/real");

        let two_hours_out = meeting("e1", now + chrono::Duration::hours(2), url);
        assert_eq!(eligibility(&two_hours_out, now), Eligibility::NotYet);
        assert_eq!(
            decide(&two_hours_out, now, false, false),
            Decision::SkipNotEligible(Eligibility::NotYet)
        );

        let thirty_seconds_out = meeting("e2", now + chrono::Duration::seconds(30), url);
        assert_eq!(eligibility(&thirty_seconds_out, now), Eligibility::Eligible);
        assert_eq!(decide(&thirty_seconds_out, now, false, false), Decision::Trigger);

        // 10 minutes into a real 1-hour meeting is still Eligible now —
        // the window closes at `end`, not a fixed grace (see the widened
        // 50-minutes-in / past-end cases below for the real boundary).
        let ten_minutes_past = meeting("e3", now - chrono::Duration::minutes(10), url);
        assert_eq!(eligibility(&ten_minutes_past, now), Eligibility::Eligible);
        assert_eq!(decide(&ten_minutes_past, now, false, false), Decision::Trigger);

        let over_an_hour_past = meeting("e3b", now - chrono::Duration::minutes(70), url);
        assert_eq!(eligibility(&over_an_hour_past, now), Eligibility::WindowPassed);
        assert_eq!(
            decide(&over_an_hour_past, now, false, false),
            Decision::SkipNotEligible(Eligibility::WindowPassed)
        );

        // Exact pre-start boundaries, both inclusive.
        assert_eq!(eligibility(&meeting("e4", now + chrono::Duration::seconds(60), url), now), Eligibility::Eligible);
        assert_eq!(eligibility(&meeting("e5", now + chrono::Duration::seconds(61), url), now), Eligibility::NotYet);

        // Window closes at the meeting's real `end`, not a fixed grace —
        // `meeting()` builds a 1-hour meeting, so 50 minutes in is still
        // Eligible (the manual Start Recording path has no such cutoff
        // either) and 1 minute past `end` is WindowPassed.
        let fifty_minutes_in = meeting("e6", now - chrono::Duration::minutes(50), url);
        assert_eq!(eligibility(&fifty_minutes_in, now), Eligibility::Eligible);
        let one_minute_past_end = meeting("e7", now - chrono::Duration::hours(1) - chrono::Duration::minutes(1), url);
        assert_eq!(eligibility(&one_minute_past_end, now), Eligibility::WindowPassed);

        // Fallback: an unparseable `end` still falls back to the fixed
        // 5-minute grace period rather than treating the meeting as
        // open-ended.
        let mut short_grace_only = meeting("e6b", now - chrono::Duration::seconds(300), url);
        short_grace_only.end = "not a timestamp".to_string();
        assert_eq!(eligibility(&short_grace_only, now), Eligibility::Eligible);
        let mut short_grace_expired = meeting("e7b", now - chrono::Duration::seconds(301), url);
        short_grace_expired.end = "not a timestamp".to_string();
        assert_eq!(eligibility(&short_grace_expired, now), Eligibility::WindowPassed);

        // Fails closed on a start time we can't understand.
        let mut garbled = meeting("e8", now, url);
        garbled.start = "whenever, honestly".to_string();
        assert_eq!(eligibility(&garbled, now), Eligibility::UnparseableStart);
        assert_eq!(decide(&garbled, now, false, false), Decision::SkipNotEligible(Eligibility::UnparseableStart));
    }

    /// ISC-178: the antecedent for the whole euphoric-surprise moment —
    /// a meeting 61 seconds out is NOT yet eligible at this poll, but the
    /// very next poll (60s later) catches it with a second to spare. The
    /// loop then proves the general property across a full two-minute
    /// sweep: there is no start offset where a meeting falls between two
    /// consecutive polls unseen.
    #[test]
    fn no_meeting_can_slip_between_two_consecutive_polls() {
        let poll_at = fixed_now();
        let url = Some("https://teams.microsoft.com/l/meetup-join/real");

        let sixty_one_seconds_out = meeting("boundary", poll_at + chrono::Duration::seconds(61), url);
        assert_eq!(
            eligibility(&sixty_one_seconds_out, poll_at),
            Eligibility::NotYet,
            "61s out is one second outside the window at this poll"
        );
        let next_poll = poll_at + chrono::Duration::seconds(POLL_INTERVAL_SECS as i64);
        assert_eq!(
            decide(&sixty_one_seconds_out, next_poll, false, false),
            Decision::Trigger,
            "…and must be caught on the very next cycle, before it starts"
        );

        for offset_secs in 0..=120 {
            let m = meeting("sweep", poll_at + chrono::Duration::seconds(offset_secs), url);
            let caught_now = eligibility(&m, poll_at) == Eligibility::Eligible;
            let caught_next = eligibility(&m, next_poll) == Eligibility::Eligible;
            assert!(
                caught_now || caught_next,
                "a meeting starting {offset_secs}s from now fell between two polls — the poll interval \
                 ({POLL_INTERVAL_SECS}s) must stay <= the pre-start window ({PRE_START_WINDOW_SECS}s)"
            );
        }
    }

    /// ISC-161: the same meeting is never opened/recorded twice across
    /// repeated polls within its trigger window.
    #[test]
    fn does_not_double_trigger_same_event() {
        let now = fixed_now();
        let m = meeting(
            "AAMkAGI2TG93AAA=_ScopingCall",
            now + chrono::Duration::seconds(30),
            Some("https://teams.microsoft.com/l/meetup-join/real"),
        );

        // First poll: nothing logged yet.
        assert_eq!(decide(&m, now, false, false), Decision::Trigger);

        // Second poll ~60s later, this time with the log row present.
        let next_poll = now + chrono::Duration::seconds(60);
        let second = decide(&m, next_poll, false, true);
        assert_eq!(second, Decision::SkipAlreadyTriggered);
        assert!(!second.opens_join_url(), "must not re-open the join link");
        assert!(!second.starts_recording(), "must not start a second recording");

        // And the same through the whole poll cycle, driven by a real
        // (in-memory) idempotency lookup rather than a bare bool.
        let seen = std::sync::Mutex::new(std::collections::HashSet::<String>::new());
        let fetchers: Vec<MeetingFetcher> = vec![Box::new({
            let m = m.clone();
            move || Ok(vec![m.clone()])
        })];
        let lookup = |id: &str| -> Result<bool, String> { Ok(seen.lock().unwrap().contains(id)) };

        let first_cycle = poll_cycle(true, &fetchers, now, false, &lookup);
        assert_eq!(first_cycle[0].1, Decision::Trigger);
        seen.lock().unwrap().insert(m.id.clone()); // the caller logs it

        let second_cycle = poll_cycle(true, &fetchers, next_poll, false, &lookup);
        assert_eq!(second_cycle[0].1, Decision::SkipAlreadyTriggered);
    }

    /// ISC-162 / ISC-175: a meeting with no Graph-native
    /// `onlineMeeting.joinUrl` is never triggered, and nothing in this
    /// module goes looking for a URL in the event's free text.
    #[test]
    fn never_triggers_a_meeting_without_a_join_url() {
        let now = fixed_now();

        let no_url = meeting("no-url", now + chrono::Duration::seconds(30), None);
        let decision = decide(&no_url, now, false, false);
        assert_eq!(decision, Decision::SkipNoJoinUrl);
        assert!(!decision.opens_join_url(), "there is no URL to open");
        assert!(!decision.starts_recording());

        // An empty/whitespace string is the same thing as absent — this
        // must not reach `open::that("")`.
        let blank_url = meeting("blank-url", now + chrono::Duration::seconds(30), Some("   "));
        assert_eq!(decide(&blank_url, now, false, false), Decision::SkipNoJoinUrl);

        // A Zoom link sitting in the event body/location is invisible to
        // v1 by design: the struct carries no body/location at all, so
        // this meeting looks exactly like any other joinUrl-less one.
        let mut body_only = meeting("body-only", now + chrono::Duration::seconds(30), None);
        body_only.subject = "Sync (https://zoom.us/j/123456789 in the invite body)".to_string();
        assert_eq!(decide(&body_only, now, false, false), Decision::SkipNoJoinUrl);
    }

    /// ISC-163: never start a second recording on top of a live one —
    /// but still open the join link, since the user clearly wants to be
    /// in this meeting.
    #[test]
    fn skips_auto_start_when_already_recording() {
        let now = fixed_now();
        let m = meeting("live", now + chrono::Duration::seconds(30), Some("https://teams.microsoft.com/l/meetup-join/real"));

        let decision = decide(&m, now, true, false);
        assert_eq!(decision, Decision::SkipAlreadyRecording);
        assert!(!decision.starts_recording(), "a second capture must never be started");
        assert!(decision.opens_join_url(), "the join link is still opened");
        assert!(decision.should_log(), "and it's logged, or it re-opens every 60s for the whole window");

        // With nothing recording, the identical meeting triggers.
        assert_eq!(decide(&m, now, false, false), Decision::Trigger);
    }

    /// ISC-166: the toggle is re-read every cycle, so turning it off
    /// stops the very next cycle (≤60s) rather than at the next app
    /// restart.
    #[test]
    fn toggle_takes_effect_next_cycle() {
        let now = fixed_now();
        let m = meeting("toggle", now + chrono::Duration::seconds(30), Some("https://teams.microsoft.com/l/meetup-join/real"));
        let fetchers: Vec<MeetingFetcher> = vec![Box::new(move || Ok(vec![m.clone()]))];

        let enabled_cycle = poll_cycle(true, &fetchers, now, false, &never_triggered);
        assert_eq!(enabled_cycle.len(), 1);
        assert_eq!(enabled_cycle[0].1, Decision::Trigger);

        // Jeremiah unticks the box; same meeting, same window, next tick.
        let disabled_cycle = poll_cycle(false, &fetchers, now + chrono::Duration::seconds(60), false, &never_triggered);
        assert!(disabled_cycle.is_empty(), "a disabled poller must take no action at all");
    }

    /// ISC-169: a failing Graph call (network down, token revoked) is
    /// caught and logged — the poller thread survives and retries next
    /// cycle instead of panicking and taking the app with it.
    #[test]
    fn survives_graph_api_error() {
        let now = fixed_now();
        let failing: Vec<MeetingFetcher> =
            vec![Box::new(|| Err("401 Unauthorized: token revoked".to_string()))];

        let decisions = poll_cycle(true, &failing, now, false, &never_triggered);
        assert!(decisions.is_empty(), "a failed fetch yields no decisions, not a panic");

        // A second, healthy provider still works when the first one is
        // broken — one bad provider must not blank out the others.
        let good = meeting("still-fine", now + chrono::Duration::seconds(30), Some("https://teams.microsoft.com/l/meetup-join/real"));
        let mixed: Vec<MeetingFetcher> = vec![
            Box::new(|| Err("network unreachable".to_string())),
            Box::new(move || Ok(vec![good.clone()])),
        ];
        let decisions = poll_cycle(true, &mixed, now, false, &never_triggered);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].1, Decision::Trigger);

        // A failing idempotency lookup ends the cycle with zero actions
        // rather than risking a double-trigger.
        let working: Vec<MeetingFetcher> = vec![Box::new(move || {
            Ok(vec![meeting("db-down", fixed_now() + chrono::Duration::seconds(30), Some("https://teams.microsoft.com/l/meetup-join/real"))])
        })];
        let broken_db = |_: &str| -> Result<bool, String> { Err("database is locked".to_string()) };
        assert!(poll_cycle(true, &working, now, false, &broken_db).is_empty());
    }

    /// ISC-172 / ISC-172.1: the poll cycle is provider-agnostic — it
    /// concatenates results from an arbitrary list of fetcher closures
    /// over the shared `UpcomingMeeting` type. Adding a second provider
    /// is registering a closure, not editing the poll loop.
    #[test]
    fn merges_meetings_from_multiple_registered_providers() {
        let now = fixed_now();
        let microsoft = meeting("ms-1", now + chrono::Duration::seconds(30), Some("https://teams.microsoft.com/l/meetup-join/real"));
        let hypothetical_google = meeting("g-1", now + chrono::Duration::seconds(45), Some("https://meet.google.com/abc-defg-hij"));

        let fetchers: Vec<MeetingFetcher> = vec![
            Box::new(move || Ok(vec![microsoft.clone()])),
            Box::new(move || Ok(vec![hypothetical_google.clone()])),
        ];

        let merged = collect_meetings(&fetchers);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["ms-1", "g-1"]);

        let decisions = poll_cycle(true, &fetchers, now, false, &never_triggered);
        assert_eq!(decisions.len(), 2);
        assert!(decisions.iter().all(|(_, d)| *d == Decision::Trigger));

        // Zero registered providers is a quiet no-op, not an error.
        assert!(poll_cycle(true, &[], now, false, &never_triggered).is_empty());
    }

    fn connection(provider_id: &'static str, client_id: Option<&str>, has_tokens: bool) -> ProviderConnection {
        ProviderConnection {
            provider_id,
            client_id: client_id.map(str::to_string),
            has_tokens,
        }
    }

    /// ISC-198: the active-fetcher set is exactly the connected set, for
    /// every one of the 8 possible on/off combinations of three providers
    /// — not a fixed one-Microsoft list. "Only Outlook and Zoom", "just
    /// Google Meet", and "all three" are all just different rows of this
    /// same table, with zero code change per combination.
    #[test]
    fn active_providers_match_exactly_the_connected_set_for_every_combination() {
        let all = ["microsoft", "google", "zoom"];

        for bits in 0u8..8 {
            let states: Vec<ProviderConnection> = all
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let connected = bits & (1 << i) != 0;
                    connection(p, Some(&format!("{p}-client-id")), connected)
                })
                .collect();

            let expected: Vec<&str> = all
                .iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, p)| *p)
                .collect();

            let active = active_provider_client_ids(&states);
            let active_ids: Vec<&str> = active.iter().map(|(p, _)| *p).collect();
            assert_eq!(active_ids, expected, "combination bits={bits:03b} selected the wrong providers");
            assert_eq!(active.len(), expected.len());

            // Each selected provider carries its OWN client id — a
            // cross-wired id would send Google's credentials to Zoom.
            for (provider_id, client_id) in &active {
                assert_eq!(client_id, &format!("{provider_id}-client-id"));
            }
        }

        // The two endpoints of that sweep, stated explicitly.
        let none: Vec<ProviderConnection> = all.iter().map(|p| connection(p, Some("id"), false)).collect();
        assert!(active_provider_client_ids(&none).is_empty(), "nothing connected means zero fetchers, not a Microsoft default");
        let every: Vec<ProviderConnection> = all.iter().map(|p| connection(p, Some("id"), true)).collect();
        assert_eq!(active_provider_client_ids(&every).len(), 3);
    }

    /// ISC-199: started-but-never-finished connect flows (client id stored,
    /// consent abandoned, so no tokens) are NOT active fetchers. Without
    /// this, every poll cycle would fire a guaranteed-401 request for a
    /// provider the user never actually connected.
    #[test]
    fn a_client_id_without_tokens_is_not_an_active_provider() {
        let states = vec![
            // Finished the whole flow.
            connection("microsoft", Some("ms-client-id"), true),
            // Pasted a client id, closed the browser tab, never consented.
            connection("google", Some("google-client-id"), false),
            // Never touched at all.
            connection("zoom", None, false),
        ];

        let active = active_provider_client_ids(&states);
        assert_eq!(active, vec![("microsoft", "ms-client-id".to_string())]);

        assert!(states[0].is_active());
        assert!(!states[1].is_active(), "a client id with no tokens is an abandoned connect flow, not a connection");
        assert!(!states[2].is_active());

        // The mirror-image degenerate case: tokens somehow present but no
        // client id means token refresh can't work, so it isn't active
        // either. Blank/whitespace is treated the same as absent.
        assert!(!connection("google", None, true).is_active());
        assert!(!connection("google", Some(""), true).is_active());
        assert!(!connection("google", Some("   "), true).is_active());
    }

    /// The selected set really does drive `collect_meetings` — proving the
    /// selection is wired to the poll cycle, not just a list computed and
    /// discarded. Each "connected" provider contributes its own meeting;
    /// disconnected ones contribute nothing.
    #[test]
    fn only_connected_providers_contribute_meetings_to_the_poll_cycle() {
        let now = fixed_now();
        let url = Some("https://example.com/join/real");

        let states = vec![
            connection("microsoft", Some("ms-id"), true),
            connection("google", Some("google-id"), false), // abandoned consent
            connection("zoom", Some("zoom-id"), true),
        ];

        let fetchers: Vec<MeetingFetcher> = active_provider_client_ids(&states)
            .into_iter()
            .map(|(provider_id, _client_id)| -> MeetingFetcher {
                Box::new(move || Ok(vec![meeting(provider_id, now + chrono::Duration::seconds(30), url)]))
            })
            .collect();

        assert_eq!(fetchers.len(), 2, "two connected providers, two fetchers");

        let merged = collect_meetings(&fetchers);
        let ids: Vec<&str> = merged.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["microsoft", "zoom"], "Google contributed nothing because it was never actually connected");

        let decisions = poll_cycle(true, &fetchers, now, false, &never_triggered);
        assert_eq!(decisions.len(), 2);
        assert!(decisions.iter().all(|(_, d)| *d == Decision::Trigger));
    }
}
