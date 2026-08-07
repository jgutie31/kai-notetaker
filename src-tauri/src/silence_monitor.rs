//! The decision core of SilenceBasedStopPrompt — the second, lower-
//! confidence safety net layered *on top of* ISC-181's calendar-end
//! auto-stop (which is untouched and still stops silently and directly
//! when a meeting's scheduled end genuinely passes).
//!
//! Why a prompt and not a silent stop: a real call can run past or end
//! before its scheduled calendar time, so prolonged silence is genuine
//! evidence the meeting is over — but it is not proof. People mute
//! themselves, share a screen in silence, or step away mid-call. A wrong
//! silent stop destroys the rest of a real recording; a wrong prompt costs
//! one click. That asymmetry is the whole design (Jeremiah's own
//! requirement: trigger off "60 seconds of actual silence and no talking",
//! and ask rather than assume).
//!
//! Following this module's sibling `auto_join.rs`: everything here is a
//! pure function of plain values — no Tauri, no audio device, no dialog.
//! The side effects (reading the live buffer, showing the OS dialog,
//! calling `stop_recording`) all live in `lib.rs`.

use chrono::{DateTime, Utc};

/// RMS below this counts as silence.
///
/// Calibration: 0.005 is roughly -46 dBFS. Real room tone and mic self-
/// noise on an open input typically sit below that, while even quiet
/// speech sits well above it — so this catches "nobody is talking" without
/// requiring the digital-zero that only a muted or disconnected device
/// produces. Deliberately not 0.0: a genuinely-ended call still captures
/// ambient noise, and a zero threshold would mean the prompt never fires
/// in any real room.
pub const SILENCE_RMS_THRESHOLD: f32 = 0.005;

/// The audio window each RMS reading is computed over, and the length of
/// continuous sub-threshold time required before prompting.
pub const SILENCE_WINDOW_SECS: f32 = 60.0;

/// How often the monitor samples RMS.
///
/// 5 seconds, not 60: the calendar poller's 60s cadence is far too coarse
/// to measure a *continuous* 60-second silence window — at 60s spacing a
/// single reading is the entire window, so one badly-timed sample decides
/// everything. At 5s the window is covered by 12 independent readings, so
/// any real speech reliably lands in one of them and resets the timer
/// within one interval. It's also cheap: one buffer read per 5s against a
/// pass that costs about a millisecond.
pub const CHECK_INTERVAL_SECS: u64 = 5;

/// What the monitor should do on one check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilenceAction {
    /// Nothing to do — either there's sound, there isn't enough audio to
    /// judge yet, or the silence hasn't run long enough.
    Idle,
    /// Silence has run continuously for a full `SILENCE_WINDOW_SECS` of
    /// real time. Ask whether to stop.
    Prompt,
}

/// Tracks how long the current run of silence has lasted.
///
/// Effective latency, stated plainly because it isn't obvious from the
/// constants: each reading is itself RMS over the trailing 60 seconds of
/// audio, and a `Prompt` additionally requires 60 seconds of *consecutive*
/// sub-threshold readings. So a call that goes quiet at T prompts at about
/// T+120s, not T+60s. That is deliberate — this is the lower-confidence
/// layer, and over-waiting costs nothing while under-waiting interrupts a
/// live meeting with a popup.
#[derive(Debug, Default)]
pub struct SilenceTracker {
    /// When the current unbroken run of silence began. `None` means
    /// there's no run in progress.
    silent_since: Option<DateTime<Utc>>,
}

impl SilenceTracker {
    /// Feed one RMS reading. `rms` is `None` when the recording doesn't yet
    /// hold a full audio window — treated as "can't judge", which clears
    /// any run in progress rather than counting as silence.
    ///
    /// Returns `Prompt` exactly once per silence run: the run is cleared on
    /// firing, so a caller checking every 5 seconds gets one prompt, not
    /// twelve per minute.
    pub fn observe(&mut self, rms: Option<f32>, now: DateTime<Utc>) -> SilenceAction {
        let Some(rms) = rms else {
            self.silent_since = None;
            return SilenceAction::Idle;
        };

        if !(rms < SILENCE_RMS_THRESHOLD) {
            // Written as `!(<)` rather than `>=` so a NaN reading (which
            // compares false against everything) is treated as sound —
            // fail safe, i.e. never prompt to stop a recording based on a
            // measurement that isn't a real number.
            self.silent_since = None;
            return SilenceAction::Idle;
        }

        let started = *self.silent_since.get_or_insert(now);
        if (now - started).num_seconds() >= SILENCE_WINDOW_SECS as i64 {
            self.silent_since = None;
            SilenceAction::Prompt
        } else {
            SilenceAction::Idle
        }
    }

    /// ISC-204: the user chose "Continue". Discard the current run so
    /// re-prompting requires a fresh, full `SILENCE_WINDOW_SECS` of
    /// continuous silence — never an immediate re-fire on the next check.
    ///
    /// Also called whenever monitoring stops applying at all (recording
    /// ended, or it wasn't an auto-started recording), so a later
    /// auto-recording never inherits a stale run from an earlier one.
    pub fn reset(&mut self) {
        self.silent_since = None;
    }

    #[cfg(test)]
    fn is_tracking(&self) -> bool {
        self.silent_since.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> DateTime<Utc> {
        "2026-08-10T14:00:00Z".parse::<DateTime<Utc>>().unwrap()
    }

    fn silent() -> Option<f32> {
        Some(SILENCE_RMS_THRESHOLD / 10.0)
    }

    fn talking() -> Option<f32> {
        Some(0.12)
    }

    /// ISC-202: continuous sub-threshold silence for a full 60 seconds
    /// fires exactly one `Prompt` — and it fires at the 60-second mark,
    /// not before.
    #[test]
    fn prompts_once_after_a_full_continuous_silence_window() {
        let mut tracker = SilenceTracker::default();
        let start = t0();

        // 55 seconds of silence sampled every 5s: not yet.
        for elapsed in (0..60).step_by(CHECK_INTERVAL_SECS as usize) {
            let now = start + chrono::Duration::seconds(elapsed);
            assert_eq!(
                tracker.observe(silent(), now),
                SilenceAction::Idle,
                "at {elapsed}s of silence the window isn't full yet"
            );
        }

        // The check that lands exactly on 60 seconds fires.
        let at_sixty = start + chrono::Duration::seconds(60);
        assert_eq!(tracker.observe(silent(), at_sixty), SilenceAction::Prompt);

        // And it does NOT fire again on the checks immediately after —
        // one prompt per silence run, not one every 5 seconds.
        for elapsed in [65, 70, 75, 80, 90, 100, 110] {
            let now = start + chrono::Duration::seconds(elapsed);
            assert_eq!(
                tracker.observe(silent(), now),
                SilenceAction::Idle,
                "must not re-fire at {elapsed}s — the run was consumed by the prompt"
            );
        }
    }

    /// The core anti-false-positive property: any sound at all resets the
    /// clock, so a real conversation with natural pauses never prompts.
    #[test]
    fn any_sound_resets_the_silence_clock() {
        let mut tracker = SilenceTracker::default();
        let start = t0();

        // 55 seconds of silence...
        for elapsed in (0..=55).step_by(5) {
            assert_eq!(tracker.observe(silent(), start + chrono::Duration::seconds(elapsed)), SilenceAction::Idle);
        }
        assert!(tracker.is_tracking(), "a run is in progress just before the deadline");

        // ...then somebody speaks, 5 seconds before the prompt would fire.
        assert_eq!(tracker.observe(talking(), start + chrono::Duration::seconds(60)), SilenceAction::Idle);
        assert!(!tracker.is_tracking(), "speech must clear the run entirely, not pause it");

        // The clock now restarts from scratch: 55 more seconds of silence
        // still isn't enough.
        for elapsed in (65..=115).step_by(5) {
            assert_eq!(
                tracker.observe(silent(), start + chrono::Duration::seconds(elapsed)),
                SilenceAction::Idle,
                "at {elapsed}s the post-speech run is only {}s old", elapsed - 65
            );
        }
        // 60 full seconds after the speech, it fires.
        assert_eq!(tracker.observe(silent(), start + chrono::Duration::seconds(125)), SilenceAction::Prompt);

        // A whole simulated hour of ordinary conversation — silence broken
        // by speech every 30 seconds — never prompts once.
        let mut chatty = SilenceTracker::default();
        for tick in 0..720 {
            let now = start + chrono::Duration::seconds(tick * 5);
            let rms = if tick % 6 == 0 { talking() } else { silent() };
            assert_eq!(chatty.observe(rms, now), SilenceAction::Idle, "a real conversation must never prompt (tick {tick})");
        }
    }

    /// Exactly at the threshold counts as sound, not silence — the
    /// boundary is stated in one place and pinned here so a `<` vs `<=`
    /// slip can't quietly widen what counts as a dead call.
    #[test]
    fn the_silence_threshold_boundary_is_exact() {
        let start = t0();

        let mut at_threshold = SilenceTracker::default();
        assert_eq!(at_threshold.observe(Some(SILENCE_RMS_THRESHOLD), start), SilenceAction::Idle);
        assert!(!at_threshold.is_tracking(), "exactly at the threshold is sound, so no run starts");

        let mut just_below = SilenceTracker::default();
        assert_eq!(just_below.observe(Some(SILENCE_RMS_THRESHOLD - f32::EPSILON), start), SilenceAction::Idle);
        assert!(just_below.is_tracking(), "just below the threshold is silence, so a run starts");

        // A NaN reading must never be treated as silence — fail safe.
        let mut nan = SilenceTracker::default();
        assert_eq!(nan.observe(Some(f32::NAN), start), SilenceAction::Idle);
        assert!(!nan.is_tracking(), "a NaN measurement must not start a silence run");
    }

    /// `None` (not enough audio buffered yet) is "can't judge", never
    /// "silent" — a recording in its first minute must not be prompted to
    /// stop just because there's no full window to measure yet.
    #[test]
    fn insufficient_audio_is_not_treated_as_silence() {
        let mut tracker = SilenceTracker::default();
        let start = t0();

        // Ten minutes of `None` readings can never accumulate a prompt.
        for elapsed in (0..600).step_by(5) {
            assert_eq!(tracker.observe(None, start + chrono::Duration::seconds(elapsed)), SilenceAction::Idle);
            assert!(!tracker.is_tracking());
        }

        // And a `None` in the middle of a real silence run clears it,
        // rather than letting the run straddle the gap.
        tracker.observe(silent(), start + chrono::Duration::seconds(600));
        assert!(tracker.is_tracking());
        tracker.observe(None, start + chrono::Duration::seconds(605));
        assert!(!tracker.is_tracking(), "an unmeasurable reading must break the run, not bridge it");
    }

    /// ISC-204: after "Continue", the next several checks do NOT re-prompt
    /// — a fresh, full 60 seconds of silence is required.
    #[test]
    fn continue_requires_a_fresh_full_window_before_prompting_again() {
        let mut tracker = SilenceTracker::default();
        let start = t0();

        // First prompt, the normal way.
        for elapsed in (0..60).step_by(5) {
            tracker.observe(silent(), start + chrono::Duration::seconds(elapsed));
        }
        assert_eq!(tracker.observe(silent(), start + chrono::Duration::seconds(60)), SilenceAction::Prompt);

        // Jeremiah clicks "Continue" — say 20 seconds later, after reading
        // the dialog. The meeting is still silent the whole time.
        tracker.reset();

        // The fresh window is measured from the first reading the tracker
        // actually SEES after the reset, not from the moment of the click:
        // a tracker can only count silence it has observed. With checks on
        // a 5-second cadence that's the tick after the click.
        let first_reading_after_continue = start + chrono::Duration::seconds(85);

        // Every check from there up to one tick short of a full window must
        // stay quiet, even though the audio never stopped being silent.
        for elapsed in (0..60).step_by(CHECK_INTERVAL_SECS as usize) {
            let now = first_reading_after_continue + chrono::Duration::seconds(elapsed);
            assert_eq!(
                tracker.observe(silent(), now),
                SilenceAction::Idle,
                "only {elapsed}s into the post-Continue window — must not re-prompt yet"
            );
        }

        // A genuinely fresh, full 60 seconds later, re-prompting is correct.
        assert_eq!(
            tracker.observe(silent(), first_reading_after_continue + chrono::Duration::seconds(60)),
            SilenceAction::Prompt,
            "after a genuinely fresh 60s window, re-prompting is correct"
        );

        // The point of the whole test, stated as the property it protects:
        // the gap between the first prompt and the second is a full fresh
        // window plus the un-observed time spent with the dialog open —
        // never the single 5-second tick that a missing reset would give.
        let first_prompt_at = start + chrono::Duration::seconds(60);
        let second_prompt_at = first_reading_after_continue + chrono::Duration::seconds(60);
        assert!(
            (second_prompt_at - first_prompt_at).num_seconds() >= SILENCE_WINDOW_SECS as i64,
            "re-prompting must be at least a full window apart, not an immediate re-fire"
        );
    }

    /// `reset()` is also the "not applicable right now" path (recording
    /// stopped, or it was a manual recording) — a later auto-recording must
    /// never inherit a stale silence run from an earlier one.
    #[test]
    fn reset_prevents_a_later_recording_inheriting_a_stale_run() {
        let mut tracker = SilenceTracker::default();
        let start = t0();

        for elapsed in (0..=55).step_by(5) {
            tracker.observe(silent(), start + chrono::Duration::seconds(elapsed));
        }
        assert!(tracker.is_tracking(), "a run is 55s deep");

        // The recording ends / stops being auto-started — lib.rs resets.
        tracker.reset();
        assert!(!tracker.is_tracking());

        // A brand-new auto-recording, an hour later, must start from zero.
        let much_later = start + chrono::Duration::hours(1);
        assert_eq!(
            tracker.observe(silent(), much_later),
            SilenceAction::Idle,
            "the first check of a new recording must never inherit the old run and fire instantly"
        );
    }

    /// The interval must divide the window evenly and be materially finer
    /// than it — otherwise a "harmless" tuning change could make the
    /// continuous-silence measurement meaningless (the same invariant-
    /// guarding style as `auto_join`'s poll-interval sweep test).
    #[test]
    fn the_check_interval_meaningfully_samples_the_silence_window() {
        let window = SILENCE_WINDOW_SECS as u64;
        assert!(CHECK_INTERVAL_SECS > 0);
        assert!(
            window % CHECK_INTERVAL_SECS == 0,
            "the window ({window}s) must be a whole number of check intervals ({CHECK_INTERVAL_SECS}s), or a prompt can land late by up to one interval"
        );
        let samples_per_window = window / CHECK_INTERVAL_SECS;
        assert!(
            samples_per_window >= 6,
            "at least 6 readings per window, or one badly-timed sample decides the whole thing; got {samples_per_window}"
        );
    }
}
