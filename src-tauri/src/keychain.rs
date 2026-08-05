//! Sources the database's SQLCipher key from the macOS Keychain, so the
//! app never asks for a password more than once (system TCC prompt on
//! first access, silent thereafter) and the key never lives in a config
//! file or hardcoded literal — ISC-38.

use security_framework::passwords::{get_generic_password, set_generic_password};
use security_framework_sys::base::errSecItemNotFound;
use thiserror::Error;

const SERVICE: &str = "com.kairoscompliance.kainotetaker";
const ACCOUNT: &str = "sqlcipher-db-key";
const KEY_LEN: usize = 32;

#[derive(Debug, Error)]
pub enum KeychainError {
    #[error("keychain access failed: {0}")]
    Access(String),
    #[error("failed to generate random key material: {0}")]
    Random(String),
}

/// Returns the app's database encryption key, generating and persisting a
/// fresh random one on first run. Never derives a key from a
/// user-supplied passphrase — a generated key avoids weak-password risk
/// entirely and needs no re-entry once the OS Keychain has it.
pub fn get_or_create_db_key() -> Result<Vec<u8>, KeychainError> {
    match get_generic_password(SERVICE, ACCOUNT) {
        Ok(key) => Ok(key),
        Err(e) if e.code() == errSecItemNotFound => {
            let mut key = vec![0u8; KEY_LEN];
            getrandom::fill(&mut key).map_err(|e| KeychainError::Random(e.to_string()))?;
            set_generic_password(SERVICE, ACCOUNT, &key)
                .map_err(|e| KeychainError::Access(e.to_string()))?;
            Ok(key)
        }
        Err(e) => Err(KeychainError::Access(e.to_string())),
    }
}

pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
