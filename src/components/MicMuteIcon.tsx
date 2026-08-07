/**
 * The mic / mic-muted glyph, shared by the badge overlay and the main
 * window's recording screen (ISC-275).
 *
 * Two genuinely DIFFERENT shapes, not one shape in two colors: the muted
 * state adds a slash across the capsule. A mute indicator that a user has
 * to remember the color code for is a mute indicator they will misread,
 * and misreading this one means either recording a private conversation
 * or losing half a meeting. Colour is layered on top as reinforcement, never
 * as the only signal — which is also what makes it legible to anyone with
 * a colour-vision deficiency.
 *
 * Lives in its own file so both windows draw the identical glyph; a second
 * hand-inlined copy would be free to drift.
 */
export function MicMuteIcon({ muted }: { muted: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {/* Mic capsule */}
      <rect x="6" y="1.5" width="4" height="7" rx="2" />
      {/* Pickup arc + stand */}
      <path d="M3.5 7.25a4.5 4.5 0 0 0 9 0" />
      <path d="M8 11.75v2.75" />
      {muted && <path d="M2 14 14 2" />}
    </svg>
  );
}
