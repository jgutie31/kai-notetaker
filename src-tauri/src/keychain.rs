//! Cross-platform secure secret storage via the `keyring` crate — macOS
//! Keychain, Windows Credential Manager, or Linux Secret Service,
//! depending on the OS, behind one API. Used for the SQLCipher database
//! key (ISC-38) and, going forward, OAuth refresh tokens for calendar
//! providers — one storage layer for both, rather than macOS-only code
//! that would need redoing when Windows support lands.

use keyring::Entry;
use thiserror::Error;

const SERVICE: &str = "com.kairoscompliance.kainotetaker";
const DB_KEY_ACCOUNT: &str = "sqlcipher-db-key";
const KEY_LEN: usize = 32;

#[derive(Debug, Error)]
pub enum KeychainError {
    #[error("secure storage access failed: {0}")]
    Access(String),
    #[error("failed to generate random key material: {0}")]
    Random(String),
}

/// Fetches a previously-stored secret, or `None` if nothing is stored
/// under this account name yet. `pub(crate)` so `oauth.rs` can reuse the
/// same secure-storage primitive for calendar tokens — one storage layer,
/// not a second copy of the Keychain/Credential-Manager/Secret-Service
/// wiring.
pub(crate) fn get_secret(account: &str) -> Result<Option<Vec<u8>>, KeychainError> {
    let entry = Entry::new(SERVICE, account).map_err(|e| KeychainError::Access(e.to_string()))?;
    match entry.get_secret() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(KeychainError::Access(e.to_string())),
    }
}

pub(crate) fn set_secret(account: &str, value: &[u8]) -> Result<(), KeychainError> {
    let entry = Entry::new(SERVICE, account).map_err(|e| KeychainError::Access(e.to_string()))?;
    entry.set_secret(value).map_err(|e| KeychainError::Access(e.to_string()))
}

/// Idempotent: deleting an account that was never stored is success, not
/// an error — the caller's intent ("this account should not exist") is
/// already satisfied either way.
#[cfg(test)]
pub(crate) fn delete_secret(account: &str) -> Result<(), KeychainError> {
    let entry = Entry::new(SERVICE, account).map_err(|e| KeychainError::Access(e.to_string()))?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(KeychainError::Access(e.to_string())),
    }
}

/// Returns the app's database encryption key, generating and persisting a
/// fresh random one on first run. Never derives a key from a
/// user-supplied passphrase — a generated key avoids weak-password risk
/// entirely and needs no re-entry once secure storage has it.
pub fn get_or_create_db_key() -> Result<Vec<u8>, KeychainError> {
    if let Some(key) = get_secret(DB_KEY_ACCOUNT)? {
        return Ok(key);
    }
    let mut key = vec![0u8; KEY_LEN];
    getrandom::fill(&mut key).map_err(|e| KeychainError::Random(e.to_string()))?;
    set_secret(DB_KEY_ACCOUNT, &key)?;
    Ok(key)
}

pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
