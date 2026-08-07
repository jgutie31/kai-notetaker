/**
 * How a meeting was captured — the frontend mirror of Rust's
 * `storage::TriggerSource` (ISC-247). These four strings are the wire
 * contract: they are exactly what is stored in SQLite and exactly what
 * serde emits, so they must not be renamed on one side alone.
 *
 * `null` is a real, expected value: meetings recorded before
 * RecordingTriggerProvenance shipped genuinely have no stored trigger.
 * We show nothing for those rather than inventing a label.
 */
export type TriggerSource = "manual" | "calendar" | "presence" | "recovered";

/**
 * Full-sentence labels for the detail view. Each one names the actual
 * mechanism rather than the enum variant — "Presence" means nothing to
 * someone reading their own meeting history six months from now, but
 * "Ad-hoc Teams call" does.
 */
const DETAIL_LABELS: Record<TriggerSource, string> = {
  manual: "Manual — you clicked Start Recording",
  calendar: "Calendar (auto-join) — a scheduled meeting",
  presence: "Ad-hoc Teams call — detected automatically",
  recovered: "Recovered after a crash — original trigger unknown",
};

/** Terse tags for the library list, where many rows compete for space. */
const SHORT_LABELS: Record<TriggerSource, string> = {
  manual: "Manual",
  calendar: "Calendar",
  presence: "Ad-hoc",
  recovered: "Recovered",
};

const ICONS: Record<TriggerSource, string> = {
  manual: "●",
  calendar: "🗓",
  presence: "⚡",
  recovered: "⟲",
};

/**
 * All three helpers tolerate an unrecognized value rather than throwing:
 * the backend already degrades an unknown stored string to `null`, and a
 * label is never worth taking the library view down for.
 */
function isKnown(value: TriggerSource | null | undefined): value is TriggerSource {
  return value != null && value in DETAIL_LABELS;
}

export function triggerSourceDetailLabel(value: TriggerSource | null | undefined): string | null {
  return isKnown(value) ? DETAIL_LABELS[value] : null;
}

export function triggerSourceShortLabel(value: TriggerSource | null | undefined): string | null {
  return isKnown(value) ? SHORT_LABELS[value] : null;
}

export function triggerSourceIcon(value: TriggerSource | null | undefined): string | null {
  return isKnown(value) ? ICONS[value] : null;
}
