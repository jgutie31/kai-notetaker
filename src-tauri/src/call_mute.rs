//! TeamsCallMuteSync (ISC-279 .. ISC-283) — the mechanism that lets ONE
//! kai-notetaker hotkey press mute both this app's own capture *and* the
//! real Microsoft Teams call.
//!
//! Phase 1 (MicMuteToggle) only ever gated what kai-notetaker writes to
//! disk. Jeremiah's live call with Paula proved the obvious consequence:
//! kai-notetaker went silent in the recording while Paula kept hearing
//! him, because Teams' own mute is a completely separate thing this app
//! had no hand in. This module closes that gap.
//!
//! The approach is deliberately *write-only*: kai-notetaker SENDS Teams'
//! own mute shortcut into Teams' process rather than trying to READ
//! Teams' internal mute state. Reading was already proven impossible
//! earlier in this project — no OS-level API exposes a conferencing app's
//! in-call mute flag, and a muted Teams never releases the OS mic stream,
//! so there is no observable signal to key off. Sending is the only path
//! that actually works.
//!
//! `CGEvent::post_to_pid` is what makes it usable in practice: it
//! delivers the keystroke to a *specific process*, not to whatever window
//! currently has focus. Jeremiah presses the hotkey while looking at
//! kai-notetaker, or a browser, or nothing at all — Teams still gets it.
//! A focus-dependent approach (posting to the session event tap) would
//! have required stealing focus to Teams and handing it back, which is
//! visible, racy, and would fight the user's actual window.
//!
//! Scope note (ISC-283): Teams only. Not a stub-and-extend design — the
//! same mechanism generalizes to other conferencing apps by executable
//! name and key combo, but nothing here is written for an app that has
//! not been verified against a real call.

use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use std::process::Command;

/// Look up the PID of a running process by its **executable name**.
///
/// Executable name, not bundle identifier: `pgrep` matches the process
/// name the kernel knows, which for Teams is `MSTeams` (confirmed from
/// `CFBundleExecutable` in the installed `/Applications/Microsoft
/// Teams.app/Contents/Info.plist`). Passing the bundle id
/// `com.microsoft.teams2` here would silently never match.
///
/// `-x` is load-bearing: without it `pgrep` substring-matches, so a name
/// like `Teams` could match an unrelated helper process and we would post
/// keystrokes at the wrong target.
///
/// Returns `None` — never `Err` — for every failure mode, because they
/// all collapse to the same ordinary fact: *that app is not running right
/// now*. Recording with Teams closed is a completely normal case (testing
/// the mute button standalone, recording an in-person meeting), not a
/// fault worth surfacing to the user or the caller.
pub fn find_running_pid(exec_name: &str) -> Option<i32> {
    let output = Command::new("pgrep").arg("-x").arg(exec_name).output().ok()?;

    // pgrep exits non-zero when nothing matched. Checking stdout parsing
    // alone would be enough, but this makes the ordinary "not running"
    // path explicit rather than incidental.
    if !output.status.success() {
        return None;
    }

    // Multiple matches are possible (several processes sharing a name).
    // The first is taken rather than trying to be clever about which is
    // "the real one": Teams' main process is what `-x MSTeams` matches,
    // and guessing among hypothetical duplicates would be inventing a
    // problem that has not been observed.
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse::<i32>()
        .ok()
}

/// Post a modifier + key combination directly to one process by PID.
///
/// Sends a full keyDown/keyUp **pair**. Both halves are required, not
/// belt-and-braces: a bare keyDown with no matching keyUp leaves the
/// target app believing the key is still physically held down, which for
/// a modifier-bearing combo can wedge that app's own keyboard handling
/// until something else releases it.
///
/// Both events carry the modifier flags. Modifiers on macOS keyboard
/// events are a property *of each event*, not separate keystrokes — so
/// `.set_flags()` on both is how Cmd+Shift+M differs from a plain M,
/// rather than posting synthetic modifier key events around it.
///
/// Returns `Err` only when event construction fails. `post_to_pid` itself
/// returns `()` with no success signal — CoreGraphics gives the caller no
/// way to learn whether the target actually received or acted on the
/// event. That is an inherent limit of the API, deliberately not papered
/// over with a fake `Ok` that implies more than it knows: a successful
/// return here means "the events were constructed and handed to the OS",
/// nothing stronger.
pub fn post_key_combo_to_pid(
    pid: i32,
    keycode: CGKeyCode,
    flags: CGEventFlags,
) -> Result<(), String> {
    // CombinedSessionState: the event source behaves as though it came
    // from the user's own session, which is what a synthesized keystroke
    // standing in for a real hotkey press should look like to Teams.
    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .map_err(|_| "failed to create CGEventSource".to_string())?;

    // The source is cloned rather than rebuilt for the second event:
    // `CGEventSource` is a retain/release handle (foreign_type with a
    // CFRetain-based clone), so this is a refcount bump, and both events
    // genuinely originating from one source is the accurate model of a
    // single physical key press.
    let key_down = CGEvent::new_keyboard_event(source.clone(), keycode, true)
        .map_err(|_| "failed to create keyDown event".to_string())?;
    let key_up = CGEvent::new_keyboard_event(source, keycode, false)
        .map_err(|_| "failed to create keyUp event".to_string())?;

    key_down.set_flags(flags);
    key_up.set_flags(flags);

    // Ordering matters: down before up, same as a real press. Posted to
    // the target pid specifically — this is the whole reason the feature
    // works without kai-notetaker stealing focus from whatever the user
    // is actually looking at.
    key_down.post_to_pid(pid);
    key_up.post_to_pid(pid);

    Ok(())
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

/// Whether this process holds macOS Accessibility permission.
///
/// Synthesizing keyboard events into *another* process is gated behind
/// Accessibility. Without it `post_to_pid` still returns normally and
/// still reports no error — the events are simply dropped by the OS.
/// That silent-swallow is exactly why this check exists: it turns "the
/// hotkey mysteriously does nothing to the call" into a diagnosable,
/// actionable message.
///
/// Declared as raw FFI because nothing already in this dependency tree
/// wraps it — same precedent as `global-hotkey`'s own `ffi.rs`, which
/// hand-declares the Carbon/CoreGraphics symbols it needs. Pulling in a
/// whole accessibility crate for one boolean would be the heavier choice.
pub fn accessibility_permission_granted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISC-279 negative case: a process that cannot exist resolves to
    /// `None`, not a panic and not an error.
    #[test]
    fn find_running_pid_returns_none_for_a_process_that_does_not_exist() {
        assert_eq!(
            find_running_pid("this-process-definitely-does-not-exist-12345"),
            None
        );
    }

    /// ISC-279 positive case: proves the match path works without
    /// depending on Teams — or on anything else about the machine.
    ///
    /// **The ISA specified `launchd` as the fixture; that is measurably
    /// wrong and this test deliberately does not use it.** Two facts,
    /// both measured on this machine rather than assumed:
    ///
    /// 1. Unprivileged `pgrep` cannot see PID 1 at all. `pgrep -x launchd`
    ///    and even bare `pgrep launchd` both exit 1 while `ps -p 1` shows
    ///    it running — pgrep enumerates 394 of this box's 401 processes,
    ///    and PID 1 is among the 7 it is not permitted to see. A test
    ///    asserting `Some` for launchd fails against a *correct*
    ///    implementation, making it a broken probe rather than a strict
    ///    one.
    /// 2. A process also cannot find *itself* this way — macOS `pgrep`
    ///    excludes its own ancestry from results (verified: a child
    ///    `pgrep -x zsh` omits the very zsh that spawned it, while
    ///    sibling processes are listed normally). So "pgrep for the test
    ///    binary's own name" is equally invalid as a fixture.
    ///
    /// Neither fact affects production correctness: Teams is neither PID
    /// 1 nor an ancestor of kai-notetaker, so it is exactly the ordinary
    /// sibling case that pgrep reports reliably.
    ///
    /// The fixture used instead is hermetic — a process this test starts
    /// itself, under a name unique to this run, via a **symlink** to
    /// `/bin/sleep`. A symlink rather than a copy because copying a
    /// system binary invalidates its code signature and macOS kills it on
    /// sight; exec'ing through a symlink keeps the signed original while
    /// giving the process our chosen name. That uniqueness is what lets
    /// this assert exact PID equality — proving `find_running_pid`
    /// identified the *correct* process, not merely that it returned some
    /// number.
    ///
    /// Asserted as a full round trip — absent, then present, then absent
    /// again — which additionally proves the function tracks real process
    /// lifetime rather than returning a stale or hardcoded hit. That is
    /// strictly stronger than the static existence check the ISA asked
    /// for, and it depends on no installed app, no GUI session, and no
    /// system process.
    #[test]
    fn find_running_pid_tracks_a_uniquely_named_process_we_start_ourselves() {
        let dir = tempfile::tempdir().expect("temp dir");
        let name = format!("kai-notetaker-pidprobe-{}", std::process::id());
        let link = dir.path().join(&name);
        std::os::unix::fs::symlink("/bin/sleep", &link).expect("symlink to /bin/sleep");

        assert_eq!(
            find_running_pid(&name),
            None,
            "nothing by this unique name should exist before we start it"
        );

        let mut child = Command::new(&link)
            .arg("30")
            .spawn()
            .expect("fixture process should start");
        let child_pid = child.id() as i32;

        // Poll rather than sleep a fixed amount: process visibility is
        // not instantaneous after spawn, and a fixed delay would either
        // be flaky or needlessly slow.
        let found = poll_until(|| find_running_pid(&name).is_some());
        assert!(found, "the fixture process should become visible to pgrep");
        assert_eq!(
            find_running_pid(&name),
            Some(child_pid),
            "should find the exact process we started, not merely something"
        );

        child.kill().expect("fixture process should be killable");
        child.wait().expect("fixture process should reap");

        let gone = poll_until(|| find_running_pid(&name).is_none());
        assert!(gone, "should stop finding the process once it has exited");
    }

    /// Retry a condition for up to ~2s. Returns whether it ever held.
    fn poll_until(mut condition: impl FnMut() -> bool) -> bool {
        for _ in 0..100 {
            if condition() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        false
    }

    /// ISC-280 smoke test: event construction and posting complete
    /// without panicking. Deliberately targets this test process's own
    /// pid with a harmless key.
    ///
    /// This asserts what is actually assertable. Whether a keystroke
    /// *reached and was acted on by* a target app is not observable from
    /// inside the process that sent it — CoreGraphics returns no signal —
    /// so verifying the real effect requires a running Teams call and a
    /// human watching the mute button. Same posture this project already
    /// takes with audio-device-dependent code: test the logic that can be
    /// tested, do not fake coverage of the part that cannot.
    #[test]
    fn post_key_combo_to_pid_constructs_and_posts_without_panicking() {
        let own_pid = std::process::id() as i32;
        let result = post_key_combo_to_pid(
            own_pid,
            core_graphics::event::KeyCode::ANSI_M,
            CGEventFlags::CGEventFlagCommand | CGEventFlags::CGEventFlagShift,
        );
        assert!(result.is_ok(), "event construction should succeed: {result:?}");
    }

    /// The permission probe must be callable and total — it answers, it
    /// never panics or blocks. Its actual value is environment-dependent
    /// (granted or not on this machine), so the value itself is not
    /// asserted; that would be asserting the machine's settings.
    #[test]
    fn accessibility_permission_probe_answers_without_panicking() {
        let _ = accessibility_permission_granted();
    }
}
