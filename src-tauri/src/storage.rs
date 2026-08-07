//! Real schema for persisted meetings, transcripts, summaries, action
//! items, and embeddings, on a database encrypted at rest via SQLCipher
//! (ISC-38/39) — the key is sourced from the OS keychain, never a
//! hardcoded literal or config file.

use crate::keychain;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("meeting {0} not found")]
    MeetingNotFound(i64),
    #[error("keychain error: {0}")]
    Keychain(#[from] keychain::KeychainError),
    #[error("database could not be opened with the stored encryption key")]
    WrongEncryptionKey,
    #[error("filesystem error archiving legacy database: {0}")]
    Io(#[from] std::io::Error),
}

// Looked up once per process (a Keychain query on every command call
// would be wasteful and unnecessary — the key never changes mid-session).
static DB_KEY: OnceLock<Vec<u8>> = OnceLock::new();

fn db_key() -> Result<&'static Vec<u8>, StorageError> {
    if let Some(key) = DB_KEY.get() {
        return Ok(key);
    }
    let key = keychain::get_or_create_db_key()?;
    Ok(DB_KEY.get_or_init(|| key))
}

/// If `db_path` already holds a pre-encryption dev-era plaintext database
/// (readable with no key at all), archive it rather than let SQLCipher
/// choke on it. Dev-only: safe only because no real user data existed
/// before encryption shipped — same discipline as `reset_if_schema_outdated`.
fn archive_if_plaintext(db_path: &Path) -> Result<(), StorageError> {
    if !db_path.exists() {
        return Ok(());
    }

    let is_plaintext = Connection::open(db_path)
        .and_then(|probe| probe.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(())))
        .is_ok();
    if !is_plaintext {
        return Ok(());
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    eprintln!(
        "storage: {db_path:?} is an unencrypted dev-era database — archiving before encryption takes over"
    );
    for suffix in ["", "-wal", "-shm"] {
        let sidecar = Path::new(&format!("{}{suffix}", db_path.display())).to_path_buf();
        if sidecar.exists() {
            let backup = Path::new(&format!(
                "{}.pre-encryption-backup-{timestamp}{suffix}",
                db_path.display()
            ))
            .to_path_buf();
            std::fs::rename(&sidecar, &backup)?;
        }
    }
    Ok(())
}

/// Open a connection configured for the multi-thread/multi-connection
/// access pattern this app actually has: the Tauri command handler, the
/// background pipeline-processing thread, and the periodic retention
/// sweep can all touch the same database file around the same time.
/// SQLite's default rollback-journal mode serializes writers strictly
/// enough that this produces real "database is locked" errors under
/// exactly that pattern. WAL mode allows one writer and many concurrent
/// readers without blocking, and `busy_timeout` makes any writer-vs-writer
/// contention that WAL doesn't eliminate retry for up to 5s instead of
/// failing immediately — this is the standard, documented fix for this
/// class of error, not a workaround for a bug in the schema itself.
pub fn open_connection(db_path: &Path) -> Result<Connection, StorageError> {
    archive_if_plaintext(db_path)?;

    let conn = Connection::open(db_path)?;
    let key = db_key()?;
    conn.execute_batch(&format!("PRAGMA key = \"x'{}'\";", keychain::to_hex(key)))?;
    conn.query_row("SELECT count(*) FROM sqlite_master;", [], |_| Ok(()))
        .map_err(|_| StorageError::WrongEncryptionKey)?;

    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

/// Root cause of a real bug hit during testing: `CREATE TABLE IF NOT
/// EXISTS` does not migrate an existing table's shape — if `meetings`
/// already exists from an older schema version (e.g. the original
/// placeholder table that only had `id`/`created_at`, from earlier in
/// this same development cycle), the new columns this module expects
/// silently never get added, and every query referencing them fails at
/// runtime instead of at schema-creation time.
///
/// This is a DEV-ONLY reset, not a real migration system: if the
/// `meetings` table exists but is missing a column this schema version
/// expects, every app table is dropped and recreated from scratch. That
/// is only acceptable because no real user data exists yet — the actual
/// StorageLayer feature (SQLCipher, real migrations) must replace this
/// before any real meeting recording anyone cares about could be lost.
fn reset_if_schema_outdated(conn: &Connection) -> Result<(), StorageError> {
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meetings'",
            [],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if !table_exists {
        return Ok(());
    }

    let has_title_column: bool = conn
        .prepare("SELECT title FROM meetings LIMIT 1")
        .is_ok();
    if has_title_column {
        return Ok(());
    }

    eprintln!("storage: detected outdated dev schema on 'meetings' table — resetting all app tables");
    conn.execute_batch(
        "DROP TABLE IF EXISTS transcript_segments;
         DROP TABLE IF EXISTS meeting_summaries;
         DROP TABLE IF EXISTS action_items;
         DROP TABLE IF EXISTS meeting_embeddings;
         DROP TABLE IF EXISTS embeddings;
         DROP TABLE IF EXISTS transcripts;
         DROP TABLE IF EXISTS meetings;",
    )?;
    Ok(())
}

pub fn ensure_schema(conn: &Connection) -> Result<(), StorageError> {
    reset_if_schema_outdated(conn)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            title TEXT,
            duration_secs INTEGER NOT NULL DEFAULT 0,
            audio_path TEXT,
            status TEXT NOT NULL DEFAULT 'processing',
            error_message TEXT
         );
         CREATE TABLE IF NOT EXISTS transcript_segments (
            id INTEGER PRIMARY KEY,
            meeting_id INTEGER NOT NULL,
            segment_index INTEGER NOT NULL,
            speaker INTEGER,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(meeting_id) REFERENCES meetings(id)
         );
         CREATE TABLE IF NOT EXISTS meeting_summaries (
            meeting_id INTEGER PRIMARY KEY,
            summary_text TEXT NOT NULL,
            FOREIGN KEY(meeting_id) REFERENCES meetings(id)
         );
         CREATE TABLE IF NOT EXISTS action_items (
            id INTEGER PRIMARY KEY,
            meeting_id INTEGER NOT NULL,
            description TEXT NOT NULL,
            owner TEXT,
            due_date TEXT,
            FOREIGN KEY(meeting_id) REFERENCES meetings(id)
         );
         CREATE TABLE IF NOT EXISTS meeting_embeddings (
            id INTEGER PRIMARY KEY,
            meeting_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY(meeting_id) REFERENCES meetings(id)
         );
         CREATE TABLE IF NOT EXISTS known_speakers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS known_speaker_embeddings (
            id INTEGER PRIMARY KEY,
            known_speaker_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            source_meeting_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(known_speaker_id) REFERENCES known_speakers(id)
         );
         CREATE TABLE IF NOT EXISTS meeting_speaker_labels (
            meeting_id INTEGER NOT NULL,
            speaker_index INTEGER NOT NULL,
            known_speaker_id INTEGER,
            label TEXT,
            PRIMARY KEY(meeting_id, speaker_index),
            FOREIGN KEY(meeting_id) REFERENCES meetings(id),
            FOREIGN KEY(known_speaker_id) REFERENCES known_speakers(id)
         );
         CREATE TABLE IF NOT EXISTS transcript_segment_speaker_overrides (
            segment_id INTEGER PRIMARY KEY,
            known_speaker_id INTEGER,
            label TEXT NOT NULL,
            FOREIGN KEY(segment_id) REFERENCES transcript_segments(id),
            FOREIGN KEY(known_speaker_id) REFERENCES known_speakers(id)
         );
         CREATE TABLE IF NOT EXISTS auto_join_log (
            event_id TEXT PRIMARY KEY,
            triggered_at INTEGER NOT NULL,
            subject TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
         );",
    )?;
    apply_column_migrations(conn)?;
    Ok(())
}

/// Real, non-destructive migrations — additive `ALTER TABLE ... ADD
/// COLUMN` only, never a drop/recreate. Now that real meeting data
/// exists (including real KCG client recordings), `reset_if_schema_outdated`'s
/// drop-everything approach is no longer an acceptable way to evolve the
/// schema; this is the actual migration path going forward. Each
/// migration checks for its own column first so re-running is always
/// safe (idempotent).
fn apply_column_migrations(conn: &Connection) -> Result<(), StorageError> {
    let has_deleted_at = conn.prepare("SELECT deleted_at FROM meetings LIMIT 1").is_ok();
    if !has_deleted_at {
        conn.execute_batch("ALTER TABLE meetings ADD COLUMN deleted_at TEXT;")?;
    }

    // ISC-245. Deliberately nullable with no default: rows created before
    // this column existed genuinely have no recorded trigger, and a
    // `NOT NULL DEFAULT 'manual'` would retroactively assert something
    // false about every historical meeting — including real KCG client
    // calls that were auto-joined. NULL reads back as "unknown", which is
    // the honest answer.
    let has_trigger_source = conn.prepare("SELECT trigger_source FROM meetings LIMIT 1").is_ok();
    if !has_trigger_source {
        conn.execute_batch("ALTER TABLE meetings ADD COLUMN trigger_source TEXT;")?;
    }
    Ok(())
}

/// How a recording actually started (ISC-247). Persisted per meeting so a
/// past meeting's capture method is never ambiguous after the fact.
///
/// An enum rather than a bare `String` on purpose: a typo'd or invented
/// value becomes a compile error instead of a silently-wrong row that
/// nobody notices until they're trying to reconstruct what happened
/// months later.
///
/// `Recovered` is its own honest value, not a guess at which of the other
/// three a crash-orphaned recording originally was — that information did
/// not survive the crash, and inventing it would defeat the entire point
/// of recording provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerSource {
    /// The Start Recording button.
    Manual,
    /// AutoJoinRecording — a scheduled calendar meeting, any provider.
    Calendar,
    /// TeamsPresenceAdhocRecording — an ad-hoc call detected via presence.
    Presence,
    /// Startup orphan recovery after a crash (ISC-217-221).
    Recovered,
}

impl TriggerSource {
    /// The exact string stored in SQLite. Kept identical to the serde
    /// representation so the DB value and the JSON the frontend switches
    /// on can never drift apart.
    pub fn as_str(self) -> &'static str {
        match self {
            TriggerSource::Manual => "manual",
            TriggerSource::Calendar => "calendar",
            TriggerSource::Presence => "presence",
            TriggerSource::Recovered => "recovered",
        }
    }

    /// Lenient on purpose: an unrecognized stored value degrades to
    /// `None` ("unknown") rather than failing the whole query or being
    /// coerced into some arbitrary variant. Reads should never be the
    /// place a bad value takes the library view down.
    pub fn from_db_str(value: &str) -> Option<Self> {
        match value {
            "manual" => Some(TriggerSource::Manual),
            "calendar" => Some(TriggerSource::Calendar),
            "presence" => Some(TriggerSource::Presence),
            "recovered" => Some(TriggerSource::Recovered),
            _ => None,
        }
    }
}

/// `NULL` (pre-migration row) and an unrecognized string both mean the
/// same thing to a reader: we don't know how this was captured.
fn trigger_source_from_column(value: Option<String>) -> Option<TriggerSource> {
    value.as_deref().and_then(TriggerSource::from_db_str)
}

#[derive(Debug, Clone, Serialize)]
pub struct MeetingListItem {
    pub id: i64,
    pub created_at: String,
    pub title: Option<String>,
    pub duration_secs: i64,
    pub status: String,
    /// `None` for meetings recorded before RecordingTriggerProvenance.
    pub trigger_source: Option<TriggerSource>,
}

/// The exact shape SQLite's own `datetime('now')` produces, and the shape
/// every reader in this codebase already expects from `meetings.created_at`
/// (UTC, no timezone suffix — `MeetingLibrary.tsx`'s `formatDate` appends
/// the "Z" itself). `create_meeting` now formats this value in Rust rather
/// than letting SQLite generate it, so the stored timestamp and the
/// fallback title's timestamp are provably the same instant — but the
/// format is unchanged, so nothing downstream can tell the difference.
const SQLITE_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// Deterministic name for a recording that has no real title to inherit
/// (ISC-263). Same inputs, same string, no DB, no LLM — which is the
/// entire point. The old behavior derived a title from the first six
/// words of an LLM summary, which meant a meeting had no name at all
/// until minutes of pipeline work finished, and then got a name that was
/// frequently unrelated to the call.
///
/// `created_at` is UTC (matches the column it came from); the displayed
/// stamp is converted to the system's local time before formatting, so it
/// agrees with `MeetingLibrary.tsx`'s own local-time date column instead
/// of contradicting it on the same row.
///
/// `Calendar` is not one of the three trigger sources this is specified
/// for — a calendar recording is supposed to arrive with the event's real
/// subject. Reaching here with `Calendar` means that subject was missing
/// or blank, so it gets the same neutral label as a manual recording:
/// claiming "Ad Hoc Call" or inventing a calendar-ish name would be
/// asserting something we do not actually know.
pub fn fallback_title(trigger_source: TriggerSource, created_at: &str) -> String {
    let label = match trigger_source {
        TriggerSource::Manual | TriggerSource::Calendar => "Recording",
        TriggerSource::Presence => "Ad Hoc Call",
        TriggerSource::Recovered => "Recovered Recording",
    };

    // An unparseable timestamp degrades to the raw stored string rather
    // than panicking or silently producing a wrong date — a badly-named
    // meeting is recoverable (rename exists), a crash in the recording
    // stop path is not.
    let stamp = chrono::NaiveDateTime::parse_from_str(created_at, SQLITE_DATETIME_FORMAT)
        .map(|dt| {
            chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc)
                .with_timezone(&chrono::Local)
                .format("%b %-d, %Y %-I:%M %p")
                .to_string()
        })
        .unwrap_or_else(|_| created_at.to_string());

    format!("{label} — {stamp}")
}

/// `trigger_source` is a required parameter, not an `Option` with a
/// default (ISC-246): every caller must state how the recording it is
/// persisting actually started. Rust has no default arguments, so a new
/// recording path that forgets to say is a compile error, not a row
/// quietly mislabeled `manual`.
///
/// `known_title` (ISC-258) is the real, already-known name of the thing
/// being recorded — in practice a calendar event's own subject, which the
/// auto-join path has had in hand the whole time. When it is present the
/// row is titled with it verbatim. When it is absent, the deterministic
/// fallback is computed HERE, once, rather than at each call site, so
/// there is exactly one place in the codebase that decides what an
/// untitled recording is called.
///
/// Either way `meetings.title` is non-NULL the moment the row exists.
pub fn create_meeting(
    conn: &Connection,
    audio_path: &str,
    duration_secs: u64,
    trigger_source: TriggerSource,
    known_title: Option<&str>,
) -> Result<i64, StorageError> {
    // ONE instant, used for both the stored `created_at` and the fallback
    // title's timestamp. Two independent "now" reads could disagree across
    // a second boundary and produce a row whose title contradicts its own
    // creation time.
    let created_at = chrono::Utc::now().format(SQLITE_DATETIME_FORMAT).to_string();

    // A blank subject is treated as no subject. Calendar events really can
    // carry an empty subject, and an empty title renders as a blank row in
    // the library — strictly worse than the deterministic fallback.
    let title = match known_title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(real) => real.to_string(),
        None => fallback_title(trigger_source, &created_at),
    };

    conn.execute(
        "INSERT INTO meetings (created_at, duration_secs, audio_path, status, trigger_source, title) VALUES (?1, ?2, ?3, 'processing', ?4, ?5)",
        rusqlite::params![created_at, duration_secs as i64, audio_path, trigger_source.as_str(), title],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Every non-null `audio_path` currently recorded in `meetings`, as one
/// query. Used by the startup orphan-recovery scan (ISC-217/ISC-219) to
/// diff on-disk recordings against the DB in memory rather than issuing
/// one query per file — and deliberately including soft-deleted rows, so
/// a deleted meeting's leftover audio file is never "recovered" back into
/// existence on the next launch.
pub fn all_audio_paths(conn: &Connection) -> Result<std::collections::HashSet<String>, StorageError> {
    let mut stmt = conn.prepare("SELECT audio_path FROM meetings WHERE audio_path IS NOT NULL")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut out = std::collections::HashSet::new();
    for row in rows {
        out.insert(row?);
    }
    Ok(out)
}

/// Status only (ISC-259). Titling used to happen here, at the very end of
/// the pipeline, derived from the LLM summary's first six words — which is
/// exactly the behavior being removed: every meeting now carries a real
/// title from `create_meeting` onward, so there is nothing left for this
/// function to name.
pub fn mark_meeting_ready(conn: &Connection, meeting_id: i64) -> Result<(), StorageError> {
    conn.execute("UPDATE meetings SET status = 'ready' WHERE id = ?1", [meeting_id])?;
    Ok(())
}

/// Resets a meeting back to `processing` for a reprocess run.
///
/// Deliberately does NOT clear `title` any more. It used to, because the
/// title was pipeline output and a stale one would have been overwritten
/// on the next `mark_meeting_ready`. Now that titles are set at creation
/// and never regenerated, clearing here would strand a reprocessed meeting
/// with a permanently NULL title — the "Processing…" placeholder forever
/// (ISC-265). A reprocess re-derives transcript/summary/speakers; it does
/// not re-derive what the meeting is called, and it must not silently
/// discard a name the user may have typed themselves via rename.
pub fn mark_meeting_processing(conn: &Connection, meeting_id: i64) -> Result<(), StorageError> {
    conn.execute("UPDATE meetings SET status = 'processing' WHERE id = ?1", [meeting_id])?;
    Ok(())
}

pub fn mark_meeting_failed(conn: &Connection, meeting_id: i64, error_message: &str) -> Result<(), StorageError> {
    conn.execute(
        "UPDATE meetings SET status = 'failed', error_message = ?1 WHERE id = ?2",
        rusqlite::params![error_message, meeting_id],
    )?;
    Ok(())
}

pub fn insert_transcript_segment(
    conn: &Connection,
    meeting_id: i64,
    segment_index: i64,
    speaker: Option<i32>,
    start_ms: i64,
    end_ms: i64,
    text: &str,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO transcript_segments (meeting_id, segment_index, speaker, start_ms, end_ms, text) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![meeting_id, segment_index, speaker, start_ms, end_ms, text],
    )?;
    Ok(())
}

pub fn insert_summary(conn: &Connection, meeting_id: i64, summary_text: &str) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO meeting_summaries (meeting_id, summary_text) VALUES (?1, ?2)",
        rusqlite::params![meeting_id, summary_text],
    )?;
    Ok(())
}

pub fn insert_action_item(
    conn: &Connection,
    meeting_id: i64,
    description: &str,
    owner: Option<&str>,
    due_date: Option<&str>,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO action_items (meeting_id, description, owner, due_date) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![meeting_id, description, owner, due_date],
    )?;
    Ok(())
}

fn f32_vec_to_bytes(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn insert_embedding(conn: &Connection, meeting_id: i64, chunk_text: &str, vector: &[f32]) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO meeting_embeddings (meeting_id, chunk_text, vector) VALUES (?1, ?2, ?3)",
        rusqlite::params![meeting_id, chunk_text, f32_vec_to_bytes(vector)],
    )?;
    Ok(())
}

/// Gets the existing known-speaker row by name, or creates one. Real
/// person identity is name-keyed (unique), separate from any one
/// meeting's raw diarization index — the same person can be "Speaker 2"
/// in one meeting and "Speaker 0" in another.
pub fn get_or_create_known_speaker(conn: &Connection, name: &str) -> Result<i64, StorageError> {
    if let Ok(id) = conn.query_row("SELECT id FROM known_speakers WHERE name = ?1", [name], |r| r.get(0)) {
        return Ok(id);
    }
    conn.execute(
        "INSERT INTO known_speakers (name, created_at) VALUES (?1, datetime('now'))",
        [name],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Stores one more voice sample for a known speaker — multiple samples
/// per person (across meetings) make future matching more robust than a
/// single embedding ever could.
pub fn add_speaker_embedding_sample(
    conn: &Connection,
    known_speaker_id: i64,
    embedding: &[f32],
    source_meeting_id: Option<i64>,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO known_speaker_embeddings (known_speaker_id, embedding, source_meeting_id, created_at)
         VALUES (?1, ?2, ?3, datetime('now'))",
        rusqlite::params![known_speaker_id, f32_vec_to_bytes(embedding), source_meeting_id],
    )?;
    Ok(())
}

/// Every stored voice sample across all known speakers, as (name,
/// embedding) pairs — the real source of truth for rebuilding an
/// in-memory `SpeakerEmbeddingManager` at app startup (the manager itself
/// has no persistence of its own).
pub fn load_all_speaker_embeddings(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT ks.name, kse.embedding FROM known_speaker_embeddings kse
         JOIN known_speakers ks ON ks.id = kse.known_speaker_id",
    )?;
    let rows = stmt.query_map([], |row| {
        let name: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        Ok((name, bytes_to_f32_vec(&bytes)))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(StorageError::from)
}

/// Sets (upserts) the display label for one raw diarization speaker
/// index within one meeting — either a link to a known, named person, or
/// just a one-off text label with no persistent identity attached.
pub fn label_meeting_speaker(
    conn: &Connection,
    meeting_id: i64,
    speaker_index: i32,
    known_speaker_id: Option<i64>,
    label: &str,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO meeting_speaker_labels (meeting_id, speaker_index, known_speaker_id, label)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(meeting_id, speaker_index) DO UPDATE SET known_speaker_id = ?3, label = ?4",
        rusqlite::params![meeting_id, speaker_index, known_speaker_id, label],
    )?;
    Ok(())
}

/// The resolved speaker_index -> display_label map for one meeting —
/// used both to render real names in the UI and to build the
/// LLM-facing labeled transcript with real names instead of "Speaker N".
pub fn get_meeting_speaker_labels(conn: &Connection, meeting_id: i64) -> Result<std::collections::HashMap<i32, String>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT speaker_index, label FROM meeting_speaker_labels WHERE meeting_id = ?1",
    )?;
    let rows = stmt.query_map([meeting_id], |row| Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?)))?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>().map_err(StorageError::from)
}

/// Labels specific transcript segments directly, overriding whatever the
/// raw diarization speaker_index would otherwise resolve to for those rows
/// only. Exists because diarization's clustering can (and, on real long
/// calls, does) merge two different real speakers into the same raw index —
/// an index-wide label would then mislabel one of them. Scoping to exact
/// segment ids lets a human correct just the wrongly-clustered stretch
/// without touching other segments that share the same raw index but are
/// actually a different person.
pub fn set_segment_speaker_labels(
    conn: &Connection,
    segment_ids: &[i64],
    known_speaker_id: Option<i64>,
    label: &str,
) -> Result<(), StorageError> {
    for segment_id in segment_ids {
        conn.execute(
            "INSERT INTO transcript_segment_speaker_overrides (segment_id, known_speaker_id, label)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(segment_id) DO UPDATE SET known_speaker_id = ?2, label = ?3",
            rusqlite::params![segment_id, known_speaker_id, label],
        )?;
    }
    Ok(())
}

/// Per-segment label overrides for one meeting, keyed by segment id — takes
/// precedence over the index-wide `meeting_speaker_labels` resolution in
/// `get_meeting_detail`.
pub fn get_segment_speaker_overrides(conn: &Connection, meeting_id: i64) -> Result<std::collections::HashMap<i64, String>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT tso.segment_id, tso.label FROM transcript_segment_speaker_overrides tso
         JOIN transcript_segments ts ON ts.id = tso.segment_id
         WHERE ts.meeting_id = ?1",
    )?;
    let rows = stmt.query_map([meeting_id], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>().map_err(StorageError::from)
}

/// Clears everything a reprocess regenerates from scratch: transcript,
/// summary, action items, embeddings, and index-wide speaker labels (which
/// are meaningless once diarization re-runs and produces new, unrelated
/// indices). Per-segment overrides are cleared too, since the transcript
/// segments they key off of are about to be deleted along with everything
/// else. Shared by the reprocess CLI and the known-speaker-count reprocess
/// command so both regenerate from the same clean slate.
pub fn clear_meeting_processing_data(conn: &Connection, meeting_id: i64) -> Result<(), StorageError> {
    conn.execute(
        "DELETE FROM transcript_segment_speaker_overrides WHERE segment_id IN (SELECT id FROM transcript_segments WHERE meeting_id = ?1)",
        [meeting_id],
    )?;
    conn.execute("DELETE FROM transcript_segments WHERE meeting_id = ?1", [meeting_id])?;
    conn.execute("DELETE FROM meeting_summaries WHERE meeting_id = ?1", [meeting_id])?;
    conn.execute("DELETE FROM action_items WHERE meeting_id = ?1", [meeting_id])?;
    conn.execute("DELETE FROM meeting_embeddings WHERE meeting_id = ?1", [meeting_id])?;
    conn.execute("DELETE FROM meeting_speaker_labels WHERE meeting_id = ?1", [meeting_id])?;
    Ok(())
}

pub fn list_known_speakers(conn: &Connection) -> Result<Vec<(i64, String)>, StorageError> {
    let mut stmt = conn.prepare("SELECT id, name FROM known_speakers ORDER BY name")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(StorageError::from)
}

pub fn rename_meeting(conn: &Connection, meeting_id: i64, title: &str) -> Result<(), StorageError> {
    let rows_affected = conn.execute(
        "UPDATE meetings SET title = ?1 WHERE id = ?2 AND deleted_at IS NULL",
        rusqlite::params![title, meeting_id],
    )?;
    if rows_affected == 0 {
        return Err(StorageError::MeetingNotFound(meeting_id));
    }
    Ok(())
}

/// Soft delete — hides the meeting from `list_meetings` immediately but
/// keeps its rows and audio file on disk, so `undelete_meeting` can
/// reverse an accidental delete. Real hard purging (if it ever happens)
/// is RetentionGate's job, not this command's.
pub fn delete_meeting(conn: &Connection, meeting_id: i64) -> Result<(), StorageError> {
    let rows_affected = conn.execute(
        "UPDATE meetings SET deleted_at = datetime('now') WHERE id = ?1 AND deleted_at IS NULL",
        [meeting_id],
    )?;
    if rows_affected == 0 {
        return Err(StorageError::MeetingNotFound(meeting_id));
    }
    Ok(())
}

pub fn undelete_meeting(conn: &Connection, meeting_id: i64) -> Result<(), StorageError> {
    let rows_affected = conn.execute("UPDATE meetings SET deleted_at = NULL WHERE id = ?1", [meeting_id])?;
    if rows_affected == 0 {
        return Err(StorageError::MeetingNotFound(meeting_id));
    }
    Ok(())
}

pub fn list_meetings(conn: &Connection) -> Result<Vec<MeetingListItem>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, title, duration_secs, status, trigger_source FROM meetings WHERE deleted_at IS NULL ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(MeetingListItem {
            id: row.get(0)?,
            created_at: row.get(1)?,
            title: row.get(2)?,
            duration_secs: row.get(3)?,
            status: row.get(4)?,
            trigger_source: trigger_source_from_column(row.get(5)?),
        })
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSegmentRow {
    pub id: i64,
    pub speaker: Option<i32>,
    /// Resolved display name for `speaker`. Resolution order: an explicit
    /// per-segment override (`transcript_segment_speaker_overrides`) wins
    /// first, since it corrects a specific diarization mistake; otherwise
    /// falls back to the index-wide `meeting_speaker_labels` entry, if any.
    /// `None` means the frontend should fall back to "Speaker N".
    pub speaker_label: Option<String>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionItemRow {
    pub description: String,
    pub owner: Option<String>,
    pub due_date: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeetingDetail {
    pub id: i64,
    pub created_at: String,
    pub title: Option<String>,
    pub duration_secs: i64,
    pub status: String,
    pub error_message: Option<String>,
    pub summary: Option<String>,
    pub transcript: Vec<TranscriptSegmentRow>,
    pub action_items: Vec<ActionItemRow>,
    pub audio_path: Option<String>,
    /// How this meeting was captured (ISC-252). `None` for meetings
    /// recorded before RecordingTriggerProvenance existed — the UI shows
    /// nothing rather than inventing a label.
    pub trigger_source: Option<TriggerSource>,
}

pub fn get_meeting_detail(conn: &Connection, meeting_id: i64) -> Result<MeetingDetail, StorageError> {
    let (created_at, title, duration_secs, status, error_message, audio_path, trigger_source) = conn
        .query_row(
            "SELECT created_at, title, duration_secs, status, error_message, audio_path, trigger_source FROM meetings WHERE id = ?1",
            [meeting_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    trigger_source_from_column(row.get(6)?),
                ))
            },
        )
        .map_err(|_| StorageError::MeetingNotFound(meeting_id))?;

    let summary: Option<String> = conn
        .query_row(
            "SELECT summary_text FROM meeting_summaries WHERE meeting_id = ?1",
            [meeting_id],
            |row| row.get(0),
        )
        .ok();

    let speaker_labels = get_meeting_speaker_labels(conn, meeting_id)?;
    let segment_overrides = get_segment_speaker_overrides(conn, meeting_id)?;

    let mut stmt = conn.prepare(
        "SELECT id, speaker, start_ms, end_ms, text FROM transcript_segments WHERE meeting_id = ?1 ORDER BY segment_index ASC",
    )?;
    let transcript = stmt
        .query_map([meeting_id], |row| {
            let id: i64 = row.get(0)?;
            let speaker: Option<i32> = row.get(1)?;
            let speaker_label = segment_overrides
                .get(&id)
                .cloned()
                .or_else(|| speaker.and_then(|s| speaker_labels.get(&s).cloned()));
            Ok(TranscriptSegmentRow {
                id,
                speaker,
                speaker_label,
                start_ms: row.get(2)?,
                end_ms: row.get(3)?,
                text: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut stmt = conn.prepare(
        "SELECT description, owner, due_date FROM action_items WHERE meeting_id = ?1 ORDER BY id ASC",
    )?;
    let action_items = stmt
        .query_map([meeting_id], |row| {
            Ok(ActionItemRow {
                description: row.get(0)?,
                owner: row.get(1)?,
                due_date: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(MeetingDetail {
        id: meeting_id,
        created_at,
        title,
        duration_secs,
        status,
        error_message,
        summary,
        transcript,
        action_items,
        audio_path,
        trigger_source,
    })
}

// ---------------------------------------------------------------------
// Settings + auto-join log
//
// `settings` is a deliberately generic key/value table rather than a
// column-per-setting: this app had no settings mechanism at all before
// AutoJoinRecording (checked before adding one), and a k/v table means
// the next boolean/small-string preference is an INSERT, not a schema
// migration on a database that now holds real client recordings.
// ---------------------------------------------------------------------

const AUTO_JOIN_ENABLED_KEY: &str = "auto_join_enabled";

pub fn get_setting(conn: &Connection, key: &str) -> Result<Option<String>, StorageError> {
    let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
    let mut rows = stmt.query([key])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub fn set_setting(conn: &Connection, key: &str, value: &str) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, value],
    )?;
    Ok(())
}

/// Absent row means OFF. This is the load-bearing default for ISC-164:
/// a fresh install — or a fresh Microsoft connection on an existing
/// install — must never auto-open a browser tab and start recording
/// until Jeremiah explicitly opts in. Anything other than the exact
/// string `"true"` is treated as off, so a corrupted/hand-edited value
/// fails closed rather than silently enabling background recording.
pub fn get_auto_join_enabled(conn: &Connection) -> Result<bool, StorageError> {
    Ok(get_setting(conn, AUTO_JOIN_ENABLED_KEY)?.as_deref() == Some("true"))
}

pub fn set_auto_join_enabled(conn: &Connection, enabled: bool) -> Result<(), StorageError> {
    set_setting(conn, AUTO_JOIN_ENABLED_KEY, if enabled { "true" } else { "false" })
}

/// `INSERT OR IGNORE`, not `INSERT OR REPLACE`: the row IS the
/// idempotency record, so re-logging the same event must preserve the
/// original `triggered_at` rather than sliding it forward on every poll.
pub fn log_auto_join(conn: &Connection, event_id: &str, subject: &str) -> Result<(), StorageError> {
    conn.execute(
        "INSERT OR IGNORE INTO auto_join_log (event_id, triggered_at, subject) VALUES (?1, ?2, ?3)",
        rusqlite::params![event_id, chrono::Utc::now().timestamp(), subject],
    )?;
    Ok(())
}

pub fn was_already_auto_joined(conn: &Connection, event_id: &str) -> Result<bool, StorageError> {
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM auto_join_log WHERE event_id = ?1",
        [event_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// `(event_id, triggered_at unix seconds, subject)`, most recent first.
pub fn list_auto_joined(conn: &Connection) -> Result<Vec<(String, i64, String)>, StorageError> {
    let mut stmt = conn
        .prepare("SELECT event_id, triggered_at, subject FROM auto_join_log ORDER BY triggered_at DESC")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(StorageError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn ensure_schema_resets_an_outdated_meetings_table_instead_of_erroring() {
        let conn = Connection::open_in_memory().unwrap();
        // Simulate the real bug: an old-shape meetings table already exists
        // (this exact statement is what the original placeholder schema
        // created, before `title`/`duration_secs`/etc. existed).
        conn.execute_batch("CREATE TABLE meetings (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);")
            .unwrap();
        conn.execute("INSERT INTO meetings (created_at) VALUES (datetime('now'))", [])
            .unwrap();

        // Must not error, and must leave a fully-usable new-shape table —
        // this is the exact query that broke in real testing.
        ensure_schema(&conn).unwrap();
        let list = list_meetings(&conn).unwrap();
        assert!(list.is_empty(), "old dev row is expected to be gone after a dev-schema reset");

        // And the schema now genuinely supports the new columns.
        let id = create_meeting(&conn, "/some/path.wav", 42, TriggerSource::Manual, Some("Test Meeting")).unwrap();
        mark_meeting_ready(&conn, id).unwrap();
        let list = list_meetings(&conn).unwrap();
        assert_eq!(list[0].title.as_deref(), Some("Test Meeting"));
    }

    #[test]
    fn ensure_schema_is_a_true_noop_when_schema_already_current() {
        let conn = test_db();
        let id = create_meeting(&conn, "/a.wav", 10, TriggerSource::Manual, Some("Keep Me")).unwrap();
        mark_meeting_ready(&conn, id).unwrap();

        // Calling ensure_schema again (e.g. every command handler does
        // this) must NOT wipe data when the schema is already correct.
        ensure_schema(&conn).unwrap();

        let list = list_meetings(&conn).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].title.as_deref(), Some("Keep Me"));
    }

    #[test]
    fn full_lifecycle_create_populate_read_back() {
        let conn = test_db();
        let id = create_meeting(&conn, "/path/to/audio.wav", 125, TriggerSource::Manual, Some("Quick Sync")).unwrap();

        insert_transcript_segment(&conn, id, 0, Some(0), 0, 5000, "Hello there.").unwrap();
        insert_transcript_segment(&conn, id, 1, Some(1), 5000, 9000, "General Kenobi.").unwrap();
        insert_summary(&conn, id, "A brief greeting exchange.").unwrap();
        insert_action_item(&conn, id, "Follow up with Nesta", Some("Jeremiah"), Some("2026-08-10")).unwrap();
        insert_embedding(&conn, id, "Hello there.", &[0.1, 0.2, 0.3]).unwrap();
        mark_meeting_ready(&conn, id).unwrap();

        let list = list_meetings(&conn).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].title.as_deref(), Some("Quick Sync"));
        assert_eq!(list[0].status, "ready");

        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.transcript.len(), 2);
        assert_eq!(detail.transcript[0].text, "Hello there.");
        assert_eq!(detail.summary.as_deref(), Some("A brief greeting exchange."));
        assert_eq!(detail.action_items.len(), 1);
        assert_eq!(detail.action_items[0].owner.as_deref(), Some("Jeremiah"));
        assert_eq!(detail.audio_path.as_deref(), Some("/path/to/audio.wav"));
    }

    #[test]
    fn failed_meeting_records_error_and_status() {
        let conn = test_db();
        let id = create_meeting(&conn, "/path/to/audio.wav", 60, TriggerSource::Manual, None).unwrap();
        mark_meeting_failed(&conn, id, "ASR model failed to load").unwrap();

        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.status, "failed");
        assert_eq!(detail.error_message.as_deref(), Some("ASR model failed to load"));
    }

    #[test]
    fn get_meeting_detail_errors_for_unknown_id() {
        let conn = test_db();
        let result = get_meeting_detail(&conn, 9999);
        assert!(matches!(result, Err(StorageError::MeetingNotFound(9999))));
    }

    #[test]
    fn list_meetings_orders_newest_first() {
        let conn = test_db();
        let id1 = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100)); // ensure distinct created_at timestamps
        let id2 = create_meeting(&conn, "b.wav", 20, TriggerSource::Manual, None).unwrap();

        let list = list_meetings(&conn).unwrap();
        assert_eq!(list[0].id, id2);
        assert_eq!(list[1].id, id1);
    }

    // ISC-38: real file, real key from the keychain — the on-disk bytes
    // must never contain the plaintext value, not just "PRAGMA key was set".
    #[test]
    fn open_connection_actually_encrypts_the_database_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        {
            let conn = open_connection(&path).unwrap();
            ensure_schema(&conn).unwrap();
            create_meeting(&conn, "unmistakable-plaintext-marker.wav", 42, TriggerSource::Manual, None).unwrap();
        }

        let raw = std::fs::read(&path).unwrap();
        let needle = b"unmistakable-plaintext-marker.wav";
        assert!(
            !raw.windows(needle.len()).any(|w| w == needle),
            "found the real audio path in plaintext on disk — database is not actually encrypted"
        );
    }

    // ISC-39: opening with the wrong key must fail, not silently succeed.
    #[test]
    fn wrong_key_cannot_read_data_written_with_the_real_key() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        {
            let conn = open_connection(&path).unwrap();
            ensure_schema(&conn).unwrap();
            create_meeting(&conn, "real-key-wrote-this.wav", 10, TriggerSource::Manual, None).unwrap();
        }

        let wrong_key_conn = Connection::open(&path).unwrap();
        wrong_key_conn
            .execute_batch("PRAGMA key = \"x'0000000000000000000000000000000000000000000000000000000000000000'\";")
            .unwrap();
        let result = wrong_key_conn.query_row::<i64, _, _>("SELECT count(*) FROM meetings", [], |r| r.get(0));
        assert!(result.is_err(), "wrong key should not be able to read the meetings table");
    }

    // A pre-encryption dev-era plaintext file must be archived, not
    // corrupted, when this code first runs against it.
    #[test]
    fn legacy_plaintext_database_is_archived_not_corrupted() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        {
            // Simulate the real pre-encryption dev DB: a plain, un-keyed
            // connection with real schema and a real row already in it.
            let plain = Connection::open(&path).unwrap();
            ensure_schema(&plain).unwrap();
            create_meeting(&plain, "pre-encryption-dev-data.wav", 5, TriggerSource::Manual, None).unwrap();
        }

        let conn = open_connection(&path).unwrap();
        ensure_schema(&conn).unwrap();
        let meetings = list_meetings(&conn).unwrap();
        assert!(meetings.is_empty(), "expected a fresh encrypted db, not the archived plaintext data");

        let parent = path.parent().unwrap();
        let stem = path.file_name().unwrap().to_string_lossy().to_string();
        let archived = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().starts_with(&format!("{stem}.pre-encryption-backup-")));
        assert!(archived, "expected the original plaintext file to be archived alongside the new one");
    }

    #[test]
    fn rename_meeting_updates_title() {
        let conn = test_db();
        let id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Original Title")).unwrap();
        mark_meeting_ready(&conn, id).unwrap();

        rename_meeting(&conn, id, "Strategy Call with Nesta").unwrap();

        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.title.as_deref(), Some("Strategy Call with Nesta"));
    }

    #[test]
    fn rename_meeting_errors_for_unknown_id() {
        let conn = test_db();
        let result = rename_meeting(&conn, 9999, "New Title");
        assert!(matches!(result, Err(StorageError::MeetingNotFound(9999))));
    }

    #[test]
    fn deleted_meeting_disappears_from_list_but_undelete_restores_it() {
        let conn = test_db();
        let id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Real Meeting")).unwrap();
        mark_meeting_ready(&conn, id).unwrap();
        assert_eq!(list_meetings(&conn).unwrap().len(), 1);

        delete_meeting(&conn, id).unwrap();
        assert!(list_meetings(&conn).unwrap().is_empty(), "deleted meeting should not appear in the list");

        // Deleted rows aren't gone — this is what makes undo possible.
        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.title.as_deref(), Some("Real Meeting"));

        undelete_meeting(&conn, id).unwrap();
        assert_eq!(list_meetings(&conn).unwrap().len(), 1, "undelete should restore it to the list");
    }

    #[test]
    fn delete_meeting_errors_for_unknown_id() {
        let conn = test_db();
        let result = delete_meeting(&conn, 9999);
        assert!(matches!(result, Err(StorageError::MeetingNotFound(9999))));
    }

    #[test]
    fn deleted_at_column_migration_is_idempotent_on_a_pre_existing_database() {
        // Simulate a real pre-migration database (schema created before
        // this column existed) — ensure_schema must add the column
        // without erroring, and running it again must not error either.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE meetings (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT,
                duration_secs INTEGER NOT NULL DEFAULT 0,
                audio_path TEXT,
                status TEXT NOT NULL DEFAULT 'processing',
                error_message TEXT
             );",
        )
        .unwrap();

        ensure_schema(&conn).unwrap();
        ensure_schema(&conn).unwrap();

        let id = create_meeting(&conn, "a.wav", 5, TriggerSource::Manual, None).unwrap();
        delete_meeting(&conn, id).unwrap();
        assert!(list_meetings(&conn).unwrap().is_empty());
    }

    /// ISC-245: the real migration case — a database whose `meetings`
    /// table predates `trigger_source` entirely, with a row already in it.
    /// `ensure_schema` must add the column without erroring, must be safe
    /// to re-run, and must leave the pre-existing row readable. That row
    /// reads back as `None` rather than a fabricated value: nobody recorded
    /// how it was captured, and guessing would be a lie.
    #[test]
    fn trigger_source_column_migration_is_idempotent_and_preserves_pre_existing_rows() {
        let conn = Connection::open_in_memory().unwrap();
        // The exact schema shape that shipped BEFORE this feature —
        // `deleted_at` present (so `reset_if_schema_outdated` leaves it
        // alone), `trigger_source` absent.
        conn.execute_batch(
            "CREATE TABLE meetings (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT,
                duration_secs INTEGER NOT NULL DEFAULT 0,
                audio_path TEXT,
                status TEXT NOT NULL DEFAULT 'processing',
                error_message TEXT,
                deleted_at TEXT
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO meetings (created_at, title, duration_secs, audio_path, status)
             VALUES (datetime('now'), 'Legacy KCG Call', 900, '/legacy.wav', 'ready')",
            [],
        )
        .unwrap();
        let legacy_id = conn.last_insert_rowid();

        // Twice: the migration must be idempotent, exactly like deleted_at's.
        ensure_schema(&conn).unwrap();
        ensure_schema(&conn).unwrap();

        let has_column = conn.prepare("SELECT trigger_source FROM meetings LIMIT 1").is_ok();
        assert!(has_column, "the trigger_source column should exist after migration");

        // The pre-migration row is untouched and still readable.
        let legacy = get_meeting_detail(&conn, legacy_id).unwrap();
        assert_eq!(legacy.title.as_deref(), Some("Legacy KCG Call"));
        assert_eq!(legacy.duration_secs, 900);
        assert_eq!(
            legacy.trigger_source, None,
            "a meeting recorded before this feature has no honest trigger value — None, never a guess"
        );

        // And the migrated table genuinely accepts new writes.
        let id = create_meeting(&conn, "/new.wav", 30, TriggerSource::Calendar, None).unwrap();
        assert_eq!(get_meeting_detail(&conn, id).unwrap().trigger_source, Some(TriggerSource::Calendar));
        assert_eq!(list_meetings(&conn).unwrap().len(), 2);
    }

    /// ISC-246/247: all four variants survive a real write/read cycle
    /// through SQLite as their own distinct value — no collapsing,
    /// no defaulting, `Recovered` staying `Recovered`.
    #[test]
    fn every_trigger_source_variant_round_trips_through_storage() {
        let conn = test_db();
        for expected in [
            TriggerSource::Manual,
            TriggerSource::Calendar,
            TriggerSource::Presence,
            TriggerSource::Recovered,
        ] {
            let id = create_meeting(&conn, &format!("/{}.wav", expected.as_str()), 10, expected, None).unwrap();
            let detail = get_meeting_detail(&conn, id).unwrap();
            assert_eq!(
                detail.trigger_source,
                Some(expected),
                "{expected:?} must read back as itself, not as another variant or a default"
            );
        }
    }

    /// ISC-247: the four on-the-wire strings are part of the contract —
    /// they are what lands in SQLite AND what the frontend switches on.
    /// A silent rename here would orphan every already-stored row and
    /// break the UI label at the same time.
    #[test]
    fn trigger_source_uses_the_exact_four_lowercase_wire_strings() {
        let cases = [
            (TriggerSource::Manual, "manual"),
            (TriggerSource::Calendar, "calendar"),
            (TriggerSource::Presence, "presence"),
            (TriggerSource::Recovered, "recovered"),
        ];
        for (variant, wire) in cases {
            assert_eq!(variant.as_str(), wire);
            assert_eq!(TriggerSource::from_db_str(wire), Some(variant));
            // Same string on the JSON side the frontend actually reads.
            assert_eq!(serde_json::to_string(&variant).unwrap(), format!("\"{wire}\""));
        }
        assert_eq!(
            TriggerSource::from_db_str("teleported"),
            None,
            "an unrecognized stored value must degrade to unknown, never be coerced into a wrong variant"
        );
    }

    /// ISC-253: the library list needs the same certainty as the detail
    /// view, so the value has to come back from `list_meetings` too —
    /// not only `get_meeting_detail`.
    #[test]
    fn list_meetings_carries_each_rows_trigger_source() {
        let conn = test_db();
        create_meeting(&conn, "/p.wav", 10, TriggerSource::Presence, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100)); // distinct created_at
        create_meeting(&conn, "/r.wav", 20, TriggerSource::Recovered, None).unwrap();

        let list = list_meetings(&conn).unwrap();
        assert_eq!(list[0].trigger_source, Some(TriggerSource::Recovered));
        assert_eq!(list[1].trigger_source, Some(TriggerSource::Presence));
    }

    // ---- MeetingTitleStandardization (ISC-258 / ISC-263) ----

    /// The stamp is the SYSTEM's local time, not UTC — that's the whole
    /// point of the conversion (a fallback title must agree with
    /// `MeetingLibrary.tsx`'s local-time date column on the same row).
    /// Since these tests run on whatever machine is running them, they
    /// can't assert a literal clock string; instead they independently
    /// re-derive the expected local stamp from the same UTC input via a
    /// structurally distinct call path, then check `fallback_title`
    /// agrees — still catches a wrong label, a dropped "—", or a format
    /// typo, just not a hardcoded machine-specific hour.
    fn expected_local_stamp(utc_naive: &str) -> String {
        use chrono::TimeZone;
        let naive = chrono::NaiveDateTime::parse_from_str(utc_naive, SQLITE_DATETIME_FORMAT).unwrap();
        chrono::Local.from_utc_datetime(&naive).format("%b %-d, %Y %-I:%M %p").to_string()
    }

    #[test]
    fn fallback_title_names_a_manual_recording() {
        let stamp = expected_local_stamp("2026-08-07 14:34:09");
        assert_eq!(fallback_title(TriggerSource::Manual, "2026-08-07 14:34:09"), format!("Recording — {stamp}"));
    }

    #[test]
    fn fallback_title_names_a_presence_triggered_ad_hoc_call() {
        let stamp = expected_local_stamp("2026-08-07 14:34:09");
        assert_eq!(
            fallback_title(TriggerSource::Presence, "2026-08-07 14:34:09"),
            format!("Ad Hoc Call — {stamp}")
        );
    }

    #[test]
    fn fallback_title_names_a_crash_recovered_recording() {
        let stamp = expected_local_stamp("2026-08-07 14:34:09");
        assert_eq!(
            fallback_title(TriggerSource::Recovered, "2026-08-07 14:34:09"),
            format!("Recovered Recording — {stamp}")
        );
    }

    /// The format is only useful if it's readable at the edges too: a
    /// midnight/single-digit-hour timestamp must not render as "0:05 AM"
    /// or "Aug 07" — in local time, once converted.
    #[test]
    fn fallback_title_formats_midnight_and_single_digit_dates_readably() {
        let stamp = expected_local_stamp("2026-01-03 00:05:00");
        assert_eq!(fallback_title(TriggerSource::Manual, "2026-01-03 00:05:00"), format!("Recording — {stamp}"));
    }

    /// Total over the enum, and non-panicking on junk: a recording being
    /// stopped must never fail because a timestamp looked odd.
    #[test]
    fn fallback_title_degrades_gracefully_on_an_unparseable_timestamp() {
        let title = fallback_title(TriggerSource::Manual, "not-a-timestamp");
        assert_eq!(title, "Recording — not-a-timestamp");
    }

    /// ISC-258, the calendar case — the whole point of the feature. The
    /// real event subject lands in the row verbatim, immediately, with no
    /// pipeline run involved.
    #[test]
    fn create_meeting_uses_a_known_calendar_subject_verbatim() {
        let conn = test_db();
        let id = create_meeting(
            &conn,
            "/cal.wav",
            600,
            TriggerSource::Calendar,
            Some("PCI-DSS Assessment — Smithville"),
        )
        .unwrap();

        // Read through the same accessors the UI uses, not a raw query.
        assert_eq!(
            get_meeting_detail(&conn, id).unwrap().title.as_deref(),
            Some("PCI-DSS Assessment — Smithville")
        );
        assert_eq!(
            list_meetings(&conn).unwrap()[0].title.as_deref(),
            Some("PCI-DSS Assessment — Smithville")
        );
    }

    /// ISC-258, the three fallback cases. Asserted against
    /// `fallback_title` applied to the row's OWN stored `created_at`,
    /// which is the real claim being made: the title and the timestamp in
    /// the same row describe the same instant.
    #[test]
    fn create_meeting_computes_the_fallback_title_for_every_untitled_trigger_source() {
        let conn = test_db();
        for source in [TriggerSource::Manual, TriggerSource::Presence, TriggerSource::Recovered] {
            let id =
                create_meeting(&conn, &format!("/{}.wav", source.as_str()), 10, source, None).unwrap();
            let detail = get_meeting_detail(&conn, id).unwrap();

            assert_eq!(
                detail.title.as_deref(),
                Some(fallback_title(source, &detail.created_at).as_str()),
                "{source:?} must be titled from its own stored created_at"
            );
        }
    }

    /// ISC-258's end state, stated as its own assertion: no newly-created
    /// row is ever title-less, for any trigger source, before any pipeline
    /// work runs. This is what makes MeetingLibrary's "Processing…"
    /// placeholder effectively unreachable (ISC-265).
    #[test]
    fn a_newly_created_meeting_never_has_a_null_title() {
        let conn = test_db();
        for source in [
            TriggerSource::Manual,
            TriggerSource::Calendar,
            TriggerSource::Presence,
            TriggerSource::Recovered,
        ] {
            let id =
                create_meeting(&conn, &format!("/n-{}.wav", source.as_str()), 10, source, None).unwrap();
            let title = get_meeting_detail(&conn, id).unwrap().title;
            assert!(
                title.as_deref().is_some_and(|t| !t.trim().is_empty()),
                "{source:?} produced an empty title: {title:?}"
            );
        }
    }

    /// A calendar event really can carry a blank subject. Treating it as a
    /// real title would render an empty row in the library — worse than
    /// the deterministic fallback, so blank is treated as absent.
    #[test]
    fn create_meeting_treats_a_blank_known_title_as_no_title() {
        let conn = test_db();
        let id = create_meeting(&conn, "/blank.wav", 10, TriggerSource::Calendar, Some("   ")).unwrap();
        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(
            detail.title.as_deref(),
            Some(fallback_title(TriggerSource::Calendar, &detail.created_at).as_str())
        );
    }

    /// ISC-259: marking a meeting ready is a status change and nothing
    /// else. The title it was born with survives — including one the user
    /// typed themselves.
    #[test]
    fn mark_meeting_ready_leaves_the_title_untouched() {
        let conn = test_db();
        let id = create_meeting(&conn, "/r.wav", 10, TriggerSource::Calendar, Some("Strategy Call")).unwrap();
        rename_meeting(&conn, id, "Strategy Call with Nesta").unwrap();

        mark_meeting_ready(&conn, id).unwrap();

        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.title.as_deref(), Some("Strategy Call with Nesta"));
        assert_eq!(detail.status, "ready");
    }

    /// Regression guard for the gap ISC-259 opens: titles used to be
    /// (re)written at the end of every pipeline run, so clearing the title
    /// on reprocess was harmless. It is not any more — nothing would ever
    /// write it back.
    #[test]
    fn reprocessing_a_meeting_does_not_wipe_its_title() {
        let conn = test_db();
        let id = create_meeting(&conn, "/a.wav", 10, TriggerSource::Calendar, Some("Smithville Scoping")).unwrap();
        mark_meeting_ready(&conn, id).unwrap();

        mark_meeting_processing(&conn, id).unwrap();

        let detail = get_meeting_detail(&conn, id).unwrap();
        assert_eq!(detail.status, "processing");
        assert_eq!(
            detail.title.as_deref(),
            Some("Smithville Scoping"),
            "a reprocess re-derives transcript and summary, never the meeting's name"
        );
    }

    /// The stored `created_at` must stay in SQLite's own shape now that
    /// Rust generates it — every reader downstream (including
    /// `MeetingLibrary.tsx`'s formatDate, which appends its own "Z")
    /// already depends on exactly this format.
    #[test]
    fn create_meeting_stores_created_at_in_sqlites_own_datetime_format() {
        let conn = test_db();
        let id = create_meeting(&conn, "/f.wav", 10, TriggerSource::Manual, None).unwrap();
        let created_at = get_meeting_detail(&conn, id).unwrap().created_at;

        assert!(
            chrono::NaiveDateTime::parse_from_str(&created_at, SQLITE_DATETIME_FORMAT).is_ok(),
            "created_at must stay parseable as 'YYYY-MM-DD HH:MM:SS', got {created_at:?}"
        );

        // And it must agree with what SQLite itself would have written —
        // same UTC clock, not a local-time value that would read hours off.
        let sqlite_now: String =
            conn.query_row("SELECT datetime('now')", [], |row| row.get(0)).unwrap();
        assert_eq!(
            created_at[..10],
            sqlite_now[..10],
            "Rust-generated created_at must be the same UTC date SQLite would produce"
        );
    }

    #[test]
    fn get_or_create_known_speaker_is_idempotent_by_name() {
        let conn = test_db();
        let id1 = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        let id2 = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        assert_eq!(id1, id2, "the same name should resolve to the same known_speaker row");

        let id3 = get_or_create_known_speaker(&conn, "Nesta").unwrap();
        assert_ne!(id1, id3);
    }

    #[test]
    fn speaker_embedding_samples_round_trip_through_storage() {
        let conn = test_db();
        let speaker_id = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        let embedding = vec![0.1_f32, -0.2, 0.3, 0.4];
        add_speaker_embedding_sample(&conn, speaker_id, &embedding, Some(1)).unwrap();

        let all = load_all_speaker_embeddings(&conn).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "Jeremiah");
        assert_eq!(all[0].1, embedding, "float bytes must round-trip exactly through the BLOB encoding");
    }

    #[test]
    fn label_meeting_speaker_upserts_and_resolves_in_meeting_detail() {
        let conn = test_db();
        let meeting_id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Real Meeting")).unwrap();
        mark_meeting_ready(&conn, meeting_id).unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "hello").unwrap();
        insert_transcript_segment(&conn, meeting_id, 1, Some(1), 1000, 2000, "hi there").unwrap();

        let jeremiah_id = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        label_meeting_speaker(&conn, meeting_id, 0, Some(jeremiah_id), "Jeremiah").unwrap();
        label_meeting_speaker(&conn, meeting_id, 1, None, "Unknown caller").unwrap();

        let detail = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.transcript[0].speaker_label.as_deref(), Some("Jeremiah"));
        assert_eq!(detail.transcript[1].speaker_label.as_deref(), Some("Unknown caller"));

        // Upsert: relabeling the same (meeting, speaker_index) updates in place.
        label_meeting_speaker(&conn, meeting_id, 0, Some(jeremiah_id), "J. Gutierrez").unwrap();
        let updated = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(updated.transcript[0].speaker_label.as_deref(), Some("J. Gutierrez"));
        assert_eq!(
            get_meeting_speaker_labels(&conn, meeting_id).unwrap().len(),
            2,
            "relabeling should update the existing row, not add a duplicate"
        );
    }

    #[test]
    fn unlabeled_speaker_falls_back_to_none() {
        let conn = test_db();
        let meeting_id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Real Meeting")).unwrap();
        mark_meeting_ready(&conn, meeting_id).unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "hello").unwrap();

        let detail = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.transcript[0].speaker_label, None);
    }

    #[test]
    fn segment_override_corrects_a_diarization_merge_without_touching_the_other_speaker() {
        // Reproduces the real bug: a raw diarization index (0) that
        // actually contains two different people — one early in the call,
        // one later — because the clustering step never detected a turn
        // boundary between them. An index-wide label would mislabel
        // whichever person didn't get to type their own name.
        let conn = test_db();
        let meeting_id = create_meeting(&conn, "a.wav", 3000, TriggerSource::Manual, Some("Merged Speakers Call")).unwrap();
        mark_meeting_ready(&conn, meeting_id).unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "Jeremiah's first line").unwrap();
        insert_transcript_segment(&conn, meeting_id, 1, Some(0), 1000, 2000, "Jeremiah's second line").unwrap();
        insert_transcript_segment(&conn, meeting_id, 2, Some(0), 420_000, 430_000, "Dave's first line, misclustered as the same index").unwrap();
        insert_transcript_segment(&conn, meeting_id, 3, Some(0), 430_000, 440_000, "Dave's second line, misclustered as the same index").unwrap();

        let before = get_meeting_detail(&conn, meeting_id).unwrap();
        let dave_segment_ids: Vec<i64> = before.transcript[2..4].iter().map(|s| s.id).collect();

        let dave_id = get_or_create_known_speaker(&conn, "Dave").unwrap();
        set_segment_speaker_labels(&conn, &dave_segment_ids, Some(dave_id), "Dave").unwrap();

        let after = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(after.transcript[0].speaker_label, None, "Jeremiah's first line must stay unlabeled, not become Dave");
        assert_eq!(after.transcript[1].speaker_label, None, "Jeremiah's second line must stay unlabeled, not become Dave");
        assert_eq!(after.transcript[2].speaker_label.as_deref(), Some("Dave"));
        assert_eq!(after.transcript[3].speaker_label.as_deref(), Some("Dave"));

        // Now label Jeremiah's two lines by segment id too — proves the two
        // real people sharing raw index 0 can each be corrected
        // independently, which is the whole point of the fix.
        let jeremiah_id = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        let jeremiah_segment_ids: Vec<i64> = before.transcript[0..2].iter().map(|s| s.id).collect();
        set_segment_speaker_labels(&conn, &jeremiah_segment_ids, Some(jeremiah_id), "Jeremiah").unwrap();

        let final_detail = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(final_detail.transcript[0].speaker_label.as_deref(), Some("Jeremiah"));
        assert_eq!(final_detail.transcript[1].speaker_label.as_deref(), Some("Jeremiah"));
        assert_eq!(final_detail.transcript[2].speaker_label.as_deref(), Some("Dave"));
        assert_eq!(final_detail.transcript[3].speaker_label.as_deref(), Some("Dave"));
    }

    #[test]
    fn segment_override_takes_precedence_over_index_wide_label() {
        let conn = test_db();
        let meeting_id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Real Meeting")).unwrap();
        mark_meeting_ready(&conn, meeting_id).unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "hello").unwrap();
        insert_transcript_segment(&conn, meeting_id, 1, Some(0), 1000, 2000, "actually someone else").unwrap();

        // Index-wide label says the whole index is "Jeremiah"...
        let jeremiah_id = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        label_meeting_speaker(&conn, meeting_id, 0, Some(jeremiah_id), "Jeremiah").unwrap();

        // ...but a specific segment override corrects just the second line.
        let detail = get_meeting_detail(&conn, meeting_id).unwrap();
        let second_segment_id = detail.transcript[1].id;
        let dave_id = get_or_create_known_speaker(&conn, "Dave").unwrap();
        set_segment_speaker_labels(&conn, &[second_segment_id], Some(dave_id), "Dave").unwrap();

        let resolved = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(resolved.transcript[0].speaker_label.as_deref(), Some("Jeremiah"), "unoverridden segment still resolves via the index-wide label");
        assert_eq!(resolved.transcript[1].speaker_label.as_deref(), Some("Dave"), "segment override wins over the index-wide label");
    }

    #[test]
    fn clear_meeting_processing_data_removes_everything_a_reprocess_regenerates() {
        let conn = test_db();
        let meeting_id = create_meeting(&conn, "a.wav", 10, TriggerSource::Manual, Some("Real Meeting")).unwrap();
        mark_meeting_ready(&conn, meeting_id).unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "hello").unwrap();
        insert_summary(&conn, meeting_id, "a summary").unwrap();
        insert_action_item(&conn, meeting_id, "do a thing", None, None).unwrap();
        insert_embedding(&conn, meeting_id, "hello", &[0.1, 0.2]).unwrap();
        let jeremiah_id = get_or_create_known_speaker(&conn, "Jeremiah").unwrap();
        label_meeting_speaker(&conn, meeting_id, 0, Some(jeremiah_id), "Jeremiah").unwrap();
        let seg_id = get_meeting_detail(&conn, meeting_id).unwrap().transcript[0].id;
        set_segment_speaker_labels(&conn, &[seg_id], Some(jeremiah_id), "Jeremiah").unwrap();

        clear_meeting_processing_data(&conn, meeting_id).unwrap();

        let detail = get_meeting_detail(&conn, meeting_id).unwrap();
        assert!(detail.transcript.is_empty());
        assert_eq!(detail.summary, None);
        assert!(detail.action_items.is_empty());
        assert_eq!(get_meeting_speaker_labels(&conn, meeting_id).unwrap().len(), 0);
        assert_eq!(get_segment_speaker_overrides(&conn, meeting_id).unwrap().len(), 0);
    }

    #[test]
    fn list_known_speakers_returns_all_enrolled_names_sorted() {
        let conn = test_db();
        get_or_create_known_speaker(&conn, "Nesta").unwrap();
        get_or_create_known_speaker(&conn, "Dave").unwrap();
        get_or_create_known_speaker(&conn, "Jeremiah").unwrap();

        let names: Vec<String> = list_known_speakers(&conn).unwrap().into_iter().map(|(_, n)| n).collect();
        assert_eq!(names, vec!["Dave", "Jeremiah", "Nesta"]);
    }

    /// ISC-160: the idempotency record must be a real persisted row, not
    /// an in-memory `HashSet` that resets on relaunch — otherwise every
    /// app restart re-opens and re-records meetings it already handled.
    /// Deliberately a real on-disk file (not `open_in_memory`) closed and
    /// reopened, because an in-memory database cannot distinguish "we
    /// persisted it" from "it never left RAM".
    #[test]
    fn auto_join_log_persists_across_reconnect() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_path_buf();

        {
            let conn = Connection::open(&db_path).unwrap();
            ensure_schema(&conn).unwrap();
            log_auto_join(&conn, "AAMkAGI2TG93AAA=_ScopingCall", "Smithville PCI-DSS Scoping Call").unwrap();
            assert!(was_already_auto_joined(&conn, "AAMkAGI2TG93AAA=_ScopingCall").unwrap());
        } // connection dropped — simulates the app quitting

        let conn = Connection::open(&db_path).unwrap();
        ensure_schema(&conn).unwrap();
        assert!(
            was_already_auto_joined(&conn, "AAMkAGI2TG93AAA=_ScopingCall").unwrap(),
            "a meeting already auto-joined before a restart must still count as auto-joined after it"
        );
        assert!(!was_already_auto_joined(&conn, "some-other-event-id").unwrap());

        let logged = list_auto_joined(&conn).unwrap();
        assert_eq!(logged.len(), 1);
        assert_eq!(logged[0].0, "AAMkAGI2TG93AAA=_ScopingCall");
        assert_eq!(logged[0].2, "Smithville PCI-DSS Scoping Call");
        assert!(logged[0].1 > 0, "triggered_at must be a real unix timestamp");
    }

    /// ISC-160 (second half): re-logging the same event must not slide
    /// `triggered_at` forward — the row is the idempotency record, so
    /// the timestamp has to mean "when we FIRST auto-joined this".
    #[test]
    fn log_auto_join_is_idempotent_and_preserves_the_first_trigger_time() {
        let conn = test_db();
        log_auto_join(&conn, "event-1", "First Subject").unwrap();
        let first = list_auto_joined(&conn).unwrap();

        log_auto_join(&conn, "event-1", "A Renamed Subject").unwrap();
        let after = list_auto_joined(&conn).unwrap();

        assert_eq!(after.len(), 1, "logging the same event twice must not create a second row");
        assert_eq!(after[0].1, first[0].1, "the original triggered_at must be preserved");
        assert_eq!(after[0].2, "First Subject");
    }

    /// ISC-164: OFF by default. A fresh database — a fresh install — must
    /// report the toggle as false with no row present at all.
    #[test]
    fn auto_join_is_disabled_by_default_on_a_fresh_database() {
        let conn = test_db();
        assert!(get_setting(&conn, "auto_join_enabled").unwrap().is_none(), "no row should exist yet");
        assert!(!get_auto_join_enabled(&conn).unwrap(), "auto-join must default to OFF");

        set_auto_join_enabled(&conn, true).unwrap();
        assert!(get_auto_join_enabled(&conn).unwrap());
        set_auto_join_enabled(&conn, false).unwrap();
        assert!(!get_auto_join_enabled(&conn).unwrap());

        // Fails closed: anything that isn't exactly "true" is off.
        set_setting(&conn, "auto_join_enabled", "yes-please").unwrap();
        assert!(!get_auto_join_enabled(&conn).unwrap(), "a non-\"true\" value must fail closed to OFF");
    }
}
