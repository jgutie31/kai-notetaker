//! Real schema for persisted meetings, transcripts, summaries, action
//! items, and embeddings. Still an unencrypted SQLite database — the
//! SQLCipher key-from-keychain wiring (ISC-38/39) remains deferred and
//! tracked separately; this module is about giving the Library/Detail
//! screens real, structured, queryable content to display, not about
//! encryption-at-rest (that's the StorageLayer Feature's job when it's
//! actually picked up).

use rusqlite::Connection;
use serde::Serialize;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("meeting {0} not found")]
    MeetingNotFound(i64),
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
    let conn = Connection::open(db_path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

pub fn ensure_schema(conn: &Connection) -> Result<(), StorageError> {
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
}

pub fn get_meeting_detail(conn: &Connection, meeting_id: i64) -> Result<MeetingDetail, StorageError> {
    let (created_at, title, duration_secs, status, error_message) = conn
        .query_row(
            "SELECT created_at, title, duration_secs, status, error_message FROM meetings WHERE id = ?1",
            [meeting_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
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
}
