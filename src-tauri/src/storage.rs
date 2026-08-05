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
         );",
    )?;
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

pub fn insert_embedding(conn: &Connection, meeting_id: i64, chunk_text: &str, vector: &[f32]) -> Result<(), StorageError> {
    let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
    conn.execute(
        "INSERT INTO meeting_embeddings (meeting_id, chunk_text, vector) VALUES (?1, ?2, ?3)",
        rusqlite::params![meeting_id, chunk_text, bytes],
    )?;
    Ok(())
}

pub fn list_meetings(conn: &Connection) -> Result<Vec<MeetingListItem>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, title, duration_secs, status FROM meetings ORDER BY created_at DESC",
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

    let mut stmt = conn.prepare(
        "SELECT speaker, start_ms, end_ms, text FROM transcript_segments WHERE meeting_id = ?1 ORDER BY segment_index ASC",
    )?;
    let transcript = stmt
        .query_map([meeting_id], |row| {
            Ok(TranscriptSegmentRow {
                speaker: row.get(0)?,
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
}
