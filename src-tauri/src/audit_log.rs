//! Hash-chained, append-only, tamper-evident audit log.
//!
//! This is deliberately a separate storage primitive from the SQLCipher
//! database (see `storage.rs`). Encryption-at-rest and tamper-evidence are
//! different properties: SQLCipher proves the DB file can't be *read*
//! without the key; this module proves the audit trail can't be *silently
//! rewritten* even by someone who does have the key. Each entry embeds a
//! BLAKE3 hash of the previous entry's hash plus its own payload, so any
//! mutation, deletion, or reordering breaks the chain in a way `verify_chain`
//! detects deterministically.
//!
//! Only `append`, `verify_chain`, and `read_all` are public. There is no
//! `update` or `delete` — that absence is itself part of ISC-16.
//!
//! **Honest limitation, not overclaimed:** this chain detects accidental
//! corruption and naive edits (a mutated field, a deleted line, reordered
//! lines) because those break the recomputed hash relative to what's
//! already on disk. It does NOT resist a motivated actor with file-system
//! access who is willing to re-chain the log from any point forward — the
//! anchor (`GENESIS_SEED`) and every subsequent hash are derivable from the
//! file's own contents alone, with no external secret. A future hardening
//! pass (v2, not this session) should either HMAC each entry with a key
//! held outside this file (OS keychain) or persist `(head_hash, count)` to
//! a separate keychain-backed location checked on startup. Until then, this
//! module proves "wasn't accidentally corrupted or casually edited," not
//! "cryptographically impossible to forge by someone with disk access" —
//! do not market or rely on the stronger claim.

use blake3::Hash;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Fixed genesis value hashed to seed the very first entry's `prev_hash`.
/// Any verifier must reproduce this exact constant.
const GENESIS_SEED: &[u8] = b"KAI-NOTETAKER-AUDIT-LOG-GENESIS-V1";

#[derive(Debug, thiserror::Error)]
pub enum AuditLogError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid hash encoding at line {0}: {1}")]
    InvalidHash(usize, String),
    #[error("chain broken at entry index {index}: {reason}")]
    ChainBroken { index: usize, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub event_type: String,
    pub actor: String,
    /// Free-form event payload (e.g. meeting id, target vendor, reason).
    pub payload: serde_json::Value,
    pub payload_hash: String,
    pub prev_hash: String,
    pub entry_hash: String,
}

/// The subset of fields that go into computing `payload_hash` /
/// `entry_hash` — i.e. everything except the hashes themselves. Serialized
/// with fixed field order so the same logical record always hashes
/// identically (avoids the "different structs hash the same" ambiguity).
#[derive(Serialize)]
struct HashableRecord<'a> {
    timestamp: &'a str,
    event_type: &'a str,
    actor: &'a str,
    payload: &'a serde_json::Value,
}

pub struct AuditLog {
    path: PathBuf,
}

impl AuditLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn genesis_hash() -> Hash {
        blake3::hash(GENESIS_SEED)
    }

    fn last_hash(&self) -> Result<Hash, AuditLogError> {
        let entries = self.read_all()?;
        match entries.last() {
            Some(e) => Hash::from_hex(&e.entry_hash)
                .map_err(|err| AuditLogError::InvalidHash(entries.len(), err.to_string())),
            None => Ok(Self::genesis_hash()),
        }
    }

    /// Append one entry to the log. This is the ONLY write operation.
    pub fn append(
        &self,
        event_type: &str,
        actor: &str,
        payload: serde_json::Value,
    ) -> Result<AuditEntry, AuditLogError> {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let prev_hash = self.last_hash()?;

        let hashable = HashableRecord {
            timestamp: &timestamp,
            event_type,
            actor,
            payload: &payload,
        };
        let payload_bytes = serde_json::to_vec(&hashable)?;
        let payload_hash = blake3::hash(&payload_bytes);

        let entry_hash = blake3::Hasher::new()
            .update(prev_hash.as_bytes())
            .update(payload_hash.as_bytes())
            .finalize();

        let entry = AuditEntry {
            timestamp,
            event_type: event_type.to_string(),
            actor: actor.to_string(),
            payload,
            payload_hash: payload_hash.to_string(),
            prev_hash: prev_hash.to_string(),
            entry_hash: entry_hash.to_string(),
        };

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(&entry)?;
        writeln!(file, "{line}")?;

        Ok(entry)
    }

    /// Read every entry in file order. Does not validate the chain.
    pub fn read_all(&self) -> Result<Vec<AuditEntry>, AuditLogError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            entries.push(serde_json::from_str(&line)?);
        }
        Ok(entries)
    }

    /// Recompute every entry's hash from its payload + declared prev_hash
    /// and confirm it matches (a) the entry's own declared `entry_hash` and
    /// (b) the next entry's declared `prev_hash`. Detects mutation,
    /// deletion, and reordering.
    pub fn verify_chain(&self) -> Result<(), AuditLogError> {
        let entries = self.read_all()?;
        let mut expected_prev = Self::genesis_hash();

        for (i, entry) in entries.iter().enumerate() {
            let declared_prev = Hash::from_hex(&entry.prev_hash)
                .map_err(|e| AuditLogError::InvalidHash(i, e.to_string()))?;
            if declared_prev != expected_prev {
                return Err(AuditLogError::ChainBroken {
                    index: i,
                    reason: format!(
                        "prev_hash mismatch: expected {expected_prev}, entry declares {declared_prev}"
                    ),
                });
            }

            let hashable = HashableRecord {
                timestamp: &entry.timestamp,
                event_type: &entry.event_type,
                actor: &entry.actor,
                payload: &entry.payload,
            };
            let payload_bytes = serde_json::to_vec(&hashable)
                .map_err(AuditLogError::Serde)?;
            let recomputed_payload_hash = blake3::hash(&payload_bytes);
            if recomputed_payload_hash.to_string() != entry.payload_hash {
                return Err(AuditLogError::ChainBroken {
                    index: i,
                    reason: "payload_hash does not match recomputed hash of stored fields — payload was mutated after write".into(),
                });
            }

            let recomputed_entry_hash = blake3::Hasher::new()
                .update(declared_prev.as_bytes())
                .update(recomputed_payload_hash.as_bytes())
                .finalize();
            if recomputed_entry_hash.to_string() != entry.entry_hash {
                return Err(AuditLogError::ChainBroken {
                    index: i,
                    reason: "entry_hash does not match recomputed hash — entry_hash field was tampered".into(),
                });
            }

            expected_prev = recomputed_entry_hash;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::NamedTempFile;

    fn temp_log() -> (AuditLog, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(tmp.path()).ok(); // append() creates it fresh
        (AuditLog::new(tmp.path()), tmp)
    }

    #[test]
    fn append_writes_all_required_fields() {
        let (log, _tmp) = temp_log();
        log.append("meeting_created", "jeremiah", json!({"meeting_id": "m1"}))
            .unwrap();
        let entries = log.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert!(!e.timestamp.is_empty());
        assert_eq!(e.event_type, "meeting_created");
        assert_eq!(e.actor, "jeremiah");
        assert!(!e.payload_hash.is_empty());
        assert!(!e.prev_hash.is_empty());
        assert!(!e.entry_hash.is_empty());
    }

    #[test]
    fn first_entry_uses_genesis_prev_hash() {
        let (log, _tmp) = temp_log();
        log.append("meeting_created", "jeremiah", json!({})).unwrap();
        let entries = log.read_all().unwrap();
        assert_eq!(entries[0].prev_hash, AuditLog::genesis_hash().to_string());
    }

    #[test]
    fn verify_chain_passes_untampered() {
        let (log, _tmp) = temp_log();
        for i in 0..5 {
            log.append("event", "system", json!({"n": i})).unwrap();
        }
        assert!(log.verify_chain().is_ok());
    }

    #[test]
    fn verify_chain_detects_mutation() {
        let (log, tmp) = temp_log();
        for i in 0..5 {
            log.append("event", "system", json!({"n": i})).unwrap();
        }
        // Mutate entry index 2's payload directly on disk.
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        let mut entry: AuditEntry = serde_json::from_str(&lines[2]).unwrap();
        entry.payload = json!({"n": 9999});
        lines[2] = serde_json::to_string(&entry).unwrap();
        std::fs::write(tmp.path(), lines.join("\n") + "\n").unwrap();

        let result = log.verify_chain();
        assert!(result.is_err());
        if let Err(AuditLogError::ChainBroken { index, .. }) = result {
            assert_eq!(index, 2);
        } else {
            panic!("expected ChainBroken at index 2, got {result:?}");
        }
    }

    #[test]
    fn verify_chain_detects_deletion() {
        let (log, tmp) = temp_log();
        for i in 0..5 {
            log.append("event", "system", json!({"n": i})).unwrap();
        }
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        lines.remove(2); // delete the middle entry
        std::fs::write(tmp.path(), lines.join("\n") + "\n").unwrap();

        assert!(log.verify_chain().is_err());
    }

    #[test]
    fn verify_chain_detects_reordering() {
        let (log, tmp) = temp_log();
        for i in 0..5 {
            log.append("event", "system", json!({"n": i})).unwrap();
        }
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        lines.swap(1, 2);
        std::fs::write(tmp.path(), lines.join("\n") + "\n").unwrap();

        assert!(log.verify_chain().is_err());
    }

    #[test]
    fn audit_log_file_path_is_distinct_from_a_hypothetical_db_path() {
        let (log, _tmp) = temp_log();
        let fake_db_path = PathBuf::from("some/other/database.sqlite3");
        assert_ne!(log.path(), fake_db_path.as_path());
    }
}
