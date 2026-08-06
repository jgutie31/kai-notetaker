//! Real schema for persisted meetings, transcripts, summaries, action
//! items, and embeddings, on a database encrypted at rest via SQLCipher
//! (ISC-38/39) — the key is sourced from the OS keychain, never a
//! hardcoded literal or config file.

use crate::keychain;
use rusqlite::Connection;
use serde::Serialize;
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
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct MeetingListItem {
    pub id: i64,
    pub created_at: String,
    pub title: Option<String>,
    pub duration_secs: i64,
    pub status: String,
}

pub fn create_meeting(conn: &Connection, audio_path: &str, duration_secs: u64) -> Result<i64, StorageError> {
    conn.execute(
        "INSERT INTO meetings (created_at, duration_secs, audio_path, status) VALUES (datetime('now'), ?1, ?2, 'processing')",
        rusqlite::params![duration_secs as i64, audio_path],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn mark_meeting_ready(conn: &Connection, meeting_id: i64, title: &str) -> Result<(), StorageError> {
    conn.execute(
        "UPDATE meetings SET status = 'ready', title = ?1 WHERE id = ?2",
        rusqlite::params![title, meeting_id],
    )?;
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
        "SELECT id, created_at, title, duration_secs, status FROM meetings WHERE deleted_at IS NULL ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(MeetingListItem {
            id: row.get(0)?,
            created_at: row.get(1)?,
            title: row.get(2)?,
            duration_secs: row.get(3)?,
            status: row.get(4)?,
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
    pub speaker: Option<i32>,
    /// Resolved display name for `speaker`, if this meeting's speaker
    /// index has been labeled (auto-matched or manually assigned).
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
}

pub fn get_meeting_detail(conn: &Connection, meeting_id: i64) -> Result<MeetingDetail, StorageError> {
    let (created_at, title, duration_secs, status, error_message, audio_path) = conn
        .query_row(
            "SELECT created_at, title, duration_secs, status, error_message, audio_path FROM meetings WHERE id = ?1",
            [meeting_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
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

    let mut stmt = conn.prepare(
        "SELECT speaker, start_ms, end_ms, text FROM transcript_segments WHERE meeting_id = ?1 ORDER BY segment_index ASC",
    )?;
    let transcript = stmt
        .query_map([meeting_id], |row| {
            let speaker: Option<i32> = row.get(0)?;
            Ok(TranscriptSegmentRow {
                speaker,
                speaker_label: speaker.and_then(|s| speaker_labels.get(&s).cloned()),
                start_ms: row.get(1)?,
                end_ms: row.get(2)?,
                text: row.get(3)?,
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
    })
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
        let id = create_meeting(&conn, "/some/path.wav", 42).unwrap();
        mark_meeting_ready(&conn, id, "Test Meeting").unwrap();
        let list = list_meetings(&conn).unwrap();
        assert_eq!(list[0].title.as_deref(), Some("Test Meeting"));
    }

    #[test]
    fn ensure_schema_is_a_true_noop_when_schema_already_current() {
        let conn = test_db();
        let id = create_meeting(&conn, "/a.wav", 10).unwrap();
        mark_meeting_ready(&conn, id, "Keep Me").unwrap();

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
        let id = create_meeting(&conn, "/path/to/audio.wav", 125).unwrap();

        insert_transcript_segment(&conn, id, 0, Some(0), 0, 5000, "Hello there.").unwrap();
        insert_transcript_segment(&conn, id, 1, Some(1), 5000, 9000, "General Kenobi.").unwrap();
        insert_summary(&conn, id, "A brief greeting exchange.").unwrap();
        insert_action_item(&conn, id, "Follow up with Nesta", Some("Jeremiah"), Some("2026-08-10")).unwrap();
        insert_embedding(&conn, id, "Hello there.", &[0.1, 0.2, 0.3]).unwrap();
        mark_meeting_ready(&conn, id, "Quick Sync").unwrap();

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
        let id = create_meeting(&conn, "/path/to/audio.wav", 60).unwrap();
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
        let id1 = create_meeting(&conn, "a.wav", 10).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100)); // ensure distinct created_at timestamps
        let id2 = create_meeting(&conn, "b.wav", 20).unwrap();

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
            create_meeting(&conn, "unmistakable-plaintext-marker.wav", 42).unwrap();
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
            create_meeting(&conn, "real-key-wrote-this.wav", 10).unwrap();
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
            create_meeting(&plain, "pre-encryption-dev-data.wav", 5).unwrap();
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
        let id = create_meeting(&conn, "a.wav", 10).unwrap();
        mark_meeting_ready(&conn, id, "Original Title").unwrap();

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
        let id = create_meeting(&conn, "a.wav", 10).unwrap();
        mark_meeting_ready(&conn, id, "Real Meeting").unwrap();
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

        let id = create_meeting(&conn, "a.wav", 5).unwrap();
        delete_meeting(&conn, id).unwrap();
        assert!(list_meetings(&conn).unwrap().is_empty());
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
        let meeting_id = create_meeting(&conn, "a.wav", 10).unwrap();
        mark_meeting_ready(&conn, meeting_id, "Real Meeting").unwrap();
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
        let meeting_id = create_meeting(&conn, "a.wav", 10).unwrap();
        mark_meeting_ready(&conn, meeting_id, "Real Meeting").unwrap();
        insert_transcript_segment(&conn, meeting_id, 0, Some(0), 0, 1000, "hello").unwrap();

        let detail = get_meeting_detail(&conn, meeting_id).unwrap();
        assert_eq!(detail.transcript[0].speaker_label, None);
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
}
