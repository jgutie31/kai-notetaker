//! BAA-on-file gate for any future cloud sync path.
//!
//! v1 has zero real sync targets implemented — that's intentional (see
//! ISA.md ISC-34). This module exists so that whenever a sync target IS
//! implemented later, it has no way to reach the network without first
//! passing a programmatic check for a current, non-expired Business
//! Associate Agreement on file for that exact vendor. The gate is written
//! now, ungated by anything, specifically so nobody can add a sync feature
//! later that forgets to call it — the pattern is: no `SyncClient` type may
//! be constructed without first calling `is_sync_allowed()`.
//!
//! There is deliberately no default-allow path and no wildcard match.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyncTarget {
    pub vendor_name: String,
}

impl SyncTarget {
    pub fn new(vendor_name: impl Into<String>) -> Self {
        Self {
            vendor_name: vendor_name.into(),
        }
    }
}

/// A signed Business Associate Agreement on file for a specific vendor.
/// No field is optional — a BAA record with an unknown expiration or no
/// document reference is not a real BAA record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaaRecord {
    pub vendor_name: String,
    pub signed_date: DateTime<Utc>,
    pub expiration_date: DateTime<Utc>,
    pub document_reference: String,
}

impl BaaRecord {
    fn is_current(&self, now: DateTime<Utc>) -> bool {
        self.expiration_date > now
    }
}

/// In-memory store for this session's purposes. A real implementation would
/// back this with the SQLCipher database; the gate logic itself (the part
/// that actually matters for compliance) doesn't change based on where the
/// records are persisted, which is why the store is behind a trait.
pub trait BaaStore {
    fn find_for_vendor(&self, vendor_name: &str) -> Option<BaaRecord>;
}

pub struct CloudSyncGate<'a, S: BaaStore> {
    store: &'a S,
}

impl<'a, S: BaaStore> CloudSyncGate<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }

    /// The single function every future sync code path MUST call before
    /// opening any connection to `target`. Returns `false` unless a
    /// non-expired BAA record exists for that exact vendor.
    pub fn is_sync_allowed(&self, target: &SyncTarget) -> bool {
        self.is_sync_allowed_at(target, Utc::now())
    }

    /// Testable variant that takes an explicit "now" so expiry behavior
    /// doesn't depend on wall-clock time during tests.
    pub fn is_sync_allowed_at(&self, target: &SyncTarget, now: DateTime<Utc>) -> bool {
        match self.store.find_for_vendor(&target.vendor_name) {
            Some(record) if record.vendor_name == target.vendor_name => record.is_current(now),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use std::collections::HashMap;

    struct InMemoryStore(HashMap<String, BaaRecord>);

    impl BaaStore for InMemoryStore {
        fn find_for_vendor(&self, vendor_name: &str) -> Option<BaaRecord> {
            self.0.get(vendor_name).cloned()
        }
    }

    fn record(vendor: &str, expires_in_days: i64) -> BaaRecord {
        let now = Utc::now();
        BaaRecord {
            vendor_name: vendor.to_string(),
            signed_date: now - Duration::days(30),
            expiration_date: now + Duration::days(expires_in_days),
            document_reference: "DOC-001".to_string(),
        }
    }

    #[test]
    fn blocked_by_default_with_no_baa_record() {
        let store = InMemoryStore(HashMap::new());
        let gate = CloudSyncGate::new(&store);
        let target = SyncTarget::new("some-cloud-vendor");
        assert!(!gate.is_sync_allowed(&target));
    }

    #[test]
    fn allowed_with_current_baa_record() {
        let mut map = HashMap::new();
        map.insert("nave-security".to_string(), record("nave-security", 365));
        let store = InMemoryStore(map);
        let gate = CloudSyncGate::new(&store);
        let target = SyncTarget::new("nave-security");
        assert!(gate.is_sync_allowed(&target));
    }

    #[test]
    fn blocked_when_baa_record_expired() {
        let mut map = HashMap::new();
        map.insert("nave-security".to_string(), record("nave-security", -1));
        let store = InMemoryStore(map);
        let gate = CloudSyncGate::new(&store);
        let target = SyncTarget::new("nave-security");
        assert!(!gate.is_sync_allowed(&target));
    }

    #[test]
    fn blocked_exactly_at_expiration_boundary() {
        let now = Utc::now();
        let mut map = HashMap::new();
        map.insert(
            "vendor".to_string(),
            BaaRecord {
                vendor_name: "vendor".to_string(),
                signed_date: now - Duration::days(30),
                expiration_date: now,
                document_reference: "DOC-002".to_string(),
            },
        );
        let store = InMemoryStore(map);
        let gate = CloudSyncGate::new(&store);
        let target = SyncTarget::new("vendor");
        assert!(!gate.is_sync_allowed_at(&target, now));
    }

    #[test]
    fn no_default_allow_for_unmatched_vendor() {
        let mut map = HashMap::new();
        map.insert("nave-security".to_string(), record("nave-security", 365));
        let store = InMemoryStore(map);
        let gate = CloudSyncGate::new(&store);
        // A target that doesn't match ANY stored record must never fall
        // through to an allow.
        let target = SyncTarget::new("completely-different-vendor");
        assert!(!gate.is_sync_allowed(&target));
    }
}
