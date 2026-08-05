//! Code-enforced retention and deletion.
//!
//! The policy is deliberately not exposed as anything a UI or settings
//! screen can disable. There is no `skip_retention` or `disable()` function
//! anywhere in this module's public API. `max_age_days` has a hard ceiling
//! (`MAX_ALLOWED_AGE_DAYS`) enforced at construction time specifically so
//! nobody — including future-Jeremiah in a hurry — can quietly set
//! retention to "never" by passing a huge number.
//!
//! Every deletion this module performs is written to the audit log via
//! `AuditLog::append` as part of the same operation, not as an afterthought.
//! That ordering matters: the moment data is deleted is the single highest-
//! risk moment for an audit trail to go dark, so the deletion path is
//! exactly where the audit log dependency is least optional.

use crate::audit_log::{AuditLog, AuditLogError};
use rusqlite::Connection;
use serde_json::json;
use thiserror::Error;

/// Ten years. Anything at or above this is refused — retention "forever"
/// is not a real product decision, it's a compliance liability wearing a
/// large number.
pub const MAX_ALLOWED_AGE_DAYS: u32 = 3650;

#[derive(Debug, Error)]
pub enum RetentionError {
    #[error("max_age_days ({0}) exceeds the hard ceiling of {MAX_ALLOWED_AGE_DAYS} days — retention cannot be effectively disabled")]
    AgeExceedsCeiling(u32),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("audit log error: {0}")]
    Audit(#[from] AuditLogError),
}

#[derive(Debug, Clone, Copy)]
pub struct RetentionPolicy {
    pub max_age_days: u32,
}

impl RetentionPolicy {
    /// The only constructor. Rejects anything at/above the hard ceiling.
    pub fn new(max_age_days: u32) -> Result<Self, RetentionError> {
        if max_age_days >= MAX_ALLOWED_AGE_DAYS {
            return Err(RetentionError::AgeExceedsCeiling(max_age_days));
        }
        Ok(Self { max_age_days })
    }

    /// A reasonable default for a compliance-adjacent tool: 2 years,
    /// comfortably under the ceiling and long enough to be useful for a
    /// consultancy's own engagement history.
    pub fn default_policy() -> Self {
        Self { max_age_days: 730 }
    }
}

/// One record identified as expired by a sweep.
#[derive(Debug, Clone)]
pub struct ExpiredMeeting {
    pub id: i64,
    pub age_days: i64,
}

/// Find every meeting row older than the policy's `max_age_days`.
pub fn find_expired(conn: &Connection, policy: RetentionPolicy) -> Result<Vec<ExpiredMeeting>, RetentionError> {
    let mut stmt = conn.prepare(
        "SELECT id, CAST(julianday('now') - julianday(created_at) AS INTEGER) AS age_days \
         FROM meetings \
         WHERE julianday('now') - julianday(created_at) >= ?1",
    )?;
    let rows = stmt.query_map([policy.max_age_days as i64], |row| {
        Ok(ExpiredMeeting {
            id: row.get(0)?,
            age_days: row.get(1)?,
        })
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Hard-delete every expired meeting's rows (meetings, transcripts,
/// speakers, action_items, embeddings all cascade on `meeting_id`), and
/// write one audit log entry per deleted meeting BEFORE returning.
///
/// This is the function the scheduled background task calls. There is no
/// parameter or code path here that skips the audit-log write.
pub fn retention_sweep(
    conn: &Connection,
    audit: &AuditLog,
    policy: RetentionPolicy,
) -> Result<usize, RetentionError> {
    let expired = find_expired(conn, policy)?;
    let mut deleted_count = 0;

    for meeting in &expired {
        // Audit-log-first, delete-second — deliberate ordering, not
        // incidental. The audit log and the SQLCipher DB are separate
        // storage primitives (by architectural requirement — the audit log
        // must be verifiable without the DB key), so there is no single
        // transaction that can cover both writes atomically. If the process
        // crashes between these two calls, this ordering guarantees the
        // failure mode is "audit log claims a deletion that didn't fully
        // happen" (visible, safe to reconcile on next sweep by re-deleting)
        // rather than "data was deleted with zero record of it" (invisible,
        // the actual compliance failure). A future session should add a
        // reconciliation pass that re-attempts deletion for any
        // "retention_delete" audit entry whose meeting_id still exists.
        audit.append(
            "retention_delete",
            "system:retention_sweep",
            json!({
                "meeting_id": meeting.id,
                "age_days": meeting.age_days,
                "policy_max_age_days": policy.max_age_days,
            }),
        )?;

        conn.execute("DELETE FROM meetings WHERE id = ?1", [meeting.id])?;
        // Child tables assumed to have ON DELETE CASCADE in the schema
        // (see storage.rs migrations, not yet built); explicit deletes
        // below are defense-in-depth in case cascade isn't set on a table.
        conn.execute("DELETE FROM transcripts WHERE meeting_id = ?1", [meeting.id])?;
        conn.execute("DELETE FROM action_items WHERE meeting_id = ?1", [meeting.id])?;
        conn.execute("DELETE FROM embeddings WHERE meeting_id = ?1", [meeting.id])?;

        deleted_count += 1;
    }

    Ok(deleted_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit_log::AuditLog;
    use tempfile::NamedTempFile;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE meetings (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);
             CREATE TABLE transcripts (id INTEGER PRIMARY KEY, meeting_id INTEGER);
             CREATE TABLE action_items (id INTEGER PRIMARY KEY, meeting_id INTEGER);
             CREATE TABLE embeddings (id INTEGER PRIMARY KEY, meeting_id INTEGER);",
        )
        .unwrap();
        conn
    }

    fn insert_meeting_aged_days(conn: &Connection, id: i64, age_days: i64) {
        conn.execute(
            "INSERT INTO meetings (id, created_at) VALUES (?1, datetime('now', ?2))",
            rusqlite::params![id, format!("-{age_days} days")],
        )
        .unwrap();
    }

    #[test]
    fn rejects_disabling_retention_via_huge_max_age() {
        let result = RetentionPolicy::new(MAX_ALLOWED_AGE_DAYS);
        assert!(result.is_err());
        let result = RetentionPolicy::new(u32::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn accepts_reasonable_max_age() {
        assert!(RetentionPolicy::new(365).is_ok());
        assert!(RetentionPolicy::new(730).is_ok());
    }

    #[test]
    fn find_expired_identifies_correct_subset() {
        let conn = setup_db();
        insert_meeting_aged_days(&conn, 1, 10); // fresh
        insert_meeting_aged_days(&conn, 2, 800); // expired
        insert_meeting_aged_days(&conn, 3, 5); // fresh
        insert_meeting_aged_days(&conn, 4, 900); // expired

        let policy = RetentionPolicy::new(730).unwrap();
        let expired = find_expired(&conn, policy).unwrap();
        let mut ids: Vec<i64> = expired.iter().map(|e| e.id).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 4]);
    }

    #[test]
    fn sweep_deletes_expired_and_writes_audit_entries() {
        let conn = setup_db();
        insert_meeting_aged_days(&conn, 1, 10);
        insert_meeting_aged_days(&conn, 2, 800);
        insert_meeting_aged_days(&conn, 3, 900);

        let tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(tmp.path()).ok();
        let audit = AuditLog::new(tmp.path());

        let policy = RetentionPolicy::new(730).unwrap();
        let deleted = retention_sweep(&conn, &audit, policy).unwrap();
        assert_eq!(deleted, 2);

        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM meetings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1);

        let audit_entries = audit.read_all().unwrap();
        let retention_entries: Vec<_> = audit_entries
            .iter()
            .filter(|e| e.event_type == "retention_delete")
            .collect();
        assert_eq!(retention_entries.len(), 2);
        assert!(audit.verify_chain().is_ok());
    }

    #[test]
    fn sweep_leaves_no_orphaned_child_rows() {
        let conn = setup_db();
        insert_meeting_aged_days(&conn, 1, 900);
        conn.execute(
            "INSERT INTO transcripts (id, meeting_id) VALUES (1, 1)",
            [],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(tmp.path()).ok();
        let audit = AuditLog::new(tmp.path());
        let policy = RetentionPolicy::new(730).unwrap();
        retention_sweep(&conn, &audit, policy).unwrap();

        let orphaned: i64 = conn
            .query_row("SELECT COUNT(*) FROM transcripts WHERE meeting_id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(orphaned, 0);
    }
}
