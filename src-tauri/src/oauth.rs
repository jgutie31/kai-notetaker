//! Generic OAuth 2.0 Authorization Code + PKCE (RFC 7636) flow, shared by
//! every calendar provider (Google, Microsoft, Zoom all support this
//! exact flow for native/desktop clients). Provider-specific pieces
//! (URLs, scopes, quirky extra params, the calendar API itself) live in
//! their own provider modules; this module only knows the OAuth
//! mechanics, so adding a new provider is wiring a config, not
//! reimplementing a flow.
//!
//! No bot, no server of ours in the loop: the user signs in through their
//! own already-logged-in browser, the provider redirects back to a
//! `127.0.0.1` port this app is listening on for exactly that one
//! request, and tokens are stored via the same OS-native secure storage
//! (`keychain.rs`) already used for the database encryption key — nothing
//! is ever sent to a server this app runs.

use crate::keychain;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OAuthError {
    #[error("failed to bind local redirect listener on 127.0.0.1:{0}: {1}")]
    ListenerBind(u16, String),
    #[error("redirect listener timed out waiting for the browser to redirect back")]
    Timeout,
    #[error("redirect request did not include an authorization code")]
    MissingCode,
    #[error("the provider reported an authorization error: {0}")]
    AuthorizeDenied(String),
    #[error("token endpoint request failed: {0}")]
    TokenRequest(#[from] reqwest::Error),
    #[error("token endpoint returned an error response: {0}")]
    TokenResponse(String),
    #[error("secure storage error: {0}")]
    Keychain(#[from] keychain::KeychainError),
    #[error("no stored tokens for provider '{0}' — the user hasn't connected this account")]
    NotConnected(String),
    #[error("redirect state did not match the value this app sent — possible CSRF, aborting")]
    StateMismatch,
    #[error("failed to (de)serialize stored tokens: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("failed to generate PKCE random material: {0}")]
    Random(String),
}

/// Everything a provider needs beyond the generic OAuth2+PKCE mechanics.
/// `extra_authorize_params`: provider-specific query params on the
/// authorize URL — e.g. Google requires `access_type=offline` +
/// `prompt=consent` to guarantee a refresh_token on first consent;
/// Microsoft and Zoom don't use these at all. Keeping them here, not
/// baked into the generic URL builder, is what keeps this module
/// provider-agnostic instead of accidentally Google-shaped.
#[derive(Debug, Clone)]
pub struct OAuthProviderConfig {
    pub authorize_url: String,
    pub token_url: String,
    pub client_id: String,
    pub scope: String,
    pub extra_authorize_params: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix seconds — when `access_token` expires. Refresh proactively
    /// before this, not reactively on a 401 from the calendar API.
    pub expires_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: i64,
}

pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

impl PkcePair {
    /// RFC 7636 §4.1: verifier is 43-128 chars from `[A-Za-z0-9-._~]`.
    /// 32 random bytes base64url-encoded (no padding) is 43 chars from
    /// exactly that alphabet — satisfies the spec at the minimum length,
    /// not an approximation of it.
    pub fn generate() -> Result<Self, OAuthError> {
        let mut raw = [0u8; 32];
        getrandom::fill(&mut raw).map_err(|e| OAuthError::Random(e.to_string()))?;
        let verifier = base64_url_no_pad(&raw);
        let challenge = base64_url_no_pad(&Sha256::digest(verifier.as_bytes()));
        Ok(Self { verifier, challenge })
    }
}

/// A random anti-CSRF `state` value (RFC 6749 §10.12) — distinct from the
/// PKCE verifier, which proves possession of the code, not defense against
/// a forged redirect.
pub fn generate_state() -> Result<String, OAuthError> {
    let mut raw = [0u8; 16];
    getrandom::fill(&mut raw).map_err(|e| OAuthError::Random(e.to_string()))?;
    Ok(base64_url_no_pad(&raw))
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

pub fn build_authorize_url(config: &OAuthProviderConfig, pkce: &PkcePair, redirect_uri: &str, state: &str) -> String {
    let mut url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&code_challenge={}&code_challenge_method=S256",
        config.authorize_url,
        urlencode(&config.client_id),
        urlencode(redirect_uri),
        urlencode(&config.scope),
        urlencode(state),
        urlencode(&pkce.challenge),
    );
    for (k, v) in &config.extra_authorize_params {
        url.push('&');
        url.push_str(&urlencode(k));
        url.push('=');
        url.push_str(&urlencode(v));
    }
    url
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Percent-decodes a query-parameter value. Deliberately ASCII-only: the
/// values this is ever called on (`code`, `state`, `error`) are defined by
/// RFC 6749 §4.1.2/§4.1.2.1 as VSCHAR (`%x20-7E`, printable ASCII only),
/// so a full UTF-8-aware percent-decoder would be handling input the spec
/// says can't occur here — not a shortcut, a match to the actual contract.
fn percent_decode_ascii(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                out.push(byte as char);
                continue;
            }
        }
        if c == '+' {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// Returns `(code, state)` — `state` is returned rather than checked here
/// so the caller (who generated the expected value) does the actual
/// anti-CSRF comparison; this function only knows how to parse a query
/// string, not what the correct state was for this particular flow.
fn parse_code_from_query(path_and_query: &str) -> Result<(String, Option<String>), OAuthError> {
    let query = path_and_query.split_once('?').map(|(_, q)| q).unwrap_or("");
    let mut code = None;
    let mut error = None;
    let mut state = None;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            let decoded = percent_decode_ascii(v);
            match k {
                "code" => code = Some(decoded),
                "error" => error = Some(decoded),
                "state" => state = Some(decoded),
                _ => {}
            }
        }
    }
    if let Some(e) = error {
        return Err(OAuthError::AuthorizeDenied(e));
    }
    Ok((code.ok_or(OAuthError::MissingCode)?, state))
}

fn read_redirect_request(mut stream: std::net::TcpStream) -> Result<String, String> {
    let mut buf = [0u8; 4096];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    let request = String::from_utf8_lossy(&buf[..n]).into_owned();
    let first_line = request.lines().next().ok_or("empty redirect request")?;
    // "GET /callback?code=XYZ&state=ABC HTTP/1.1"
    let path_and_query = first_line
        .split_whitespace()
        .nth(1)
        .ok_or("malformed redirect request line")?
        .to_string();

    let body = "<html><body>Signed in — you can close this window and return to kai-notetaker.</body></html>";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    Ok(path_and_query)
}

/// Blocks until the OAuth provider's browser redirect hits
/// `http://127.0.0.1:<port>/...?code=...` (or `?error=...`), then returns
/// `(code, state)`. A real TCP accept loop against a real socket —
/// exactly what a real browser redirect connects to — not a mocked
/// callback. Enforces `timeout` via a background thread + channel, since
/// `std::net::TcpListener::accept` has no built-in timeout and a user who
/// closes the browser tab without finishing shouldn't hang the app.
pub fn await_authorization_code(port: u16, timeout: Duration) -> Result<(String, Option<String>), OAuthError> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| OAuthError::ListenerBind(port, e.to_string()))?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = listener
            .accept()
            .map_err(|e| e.to_string())
            .and_then(|(stream, _)| read_redirect_request(stream));
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(path_and_query)) => parse_code_from_query(&path_and_query),
        Ok(Err(e)) => Err(OAuthError::AuthorizeDenied(e)),
        Err(_) => Err(OAuthError::Timeout),
    }
}

fn post_token_request(token_url: &str, params: &[(&str, &str)]) -> Result<TokenResponse, OAuthError> {
    let client = reqwest::blocking::Client::new();
    let resp = client.post(token_url).form(params).send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(OAuthError::TokenResponse(format!("{status}: {body}")));
    }
    resp.json::<TokenResponse>().map_err(OAuthError::from)
}

pub fn exchange_code_for_tokens(
    config: &OAuthProviderConfig,
    code: &str,
    pkce_verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse, OAuthError> {
    post_token_request(
        &config.token_url,
        &[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", &config.client_id),
            ("code_verifier", pkce_verifier),
        ],
    )
}

pub fn refresh_access_token(config: &OAuthProviderConfig, refresh_token: &str) -> Result<TokenResponse, OAuthError> {
    let mut token = post_token_request(
        &config.token_url,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", &config.client_id),
        ],
    )?;
    // Google and Microsoft both omit refresh_token on a refresh response
    // (it doesn't rotate) — preserve the original so callers never lose it.
    if token.refresh_token.is_none() {
        token.refresh_token = Some(refresh_token.to_string());
    }
    Ok(token)
}

fn keychain_account(provider: &str) -> String {
    format!("calendar-tokens:{provider}")
}

fn client_id_keychain_account(provider: &str) -> String {
    format!("calendar-client-id:{provider}")
}

/// The OAuth client ID isn't a secret in the traditional sense (it's
/// visible in every authorize URL a browser sends), but it's still
/// per-user configuration with no other natural home in this app, and
/// keychain storage means Jeremiah pastes it once, not on every launch.
pub fn store_client_id(provider: &str, client_id: &str) -> Result<(), OAuthError> {
    keychain::set_secret(&client_id_keychain_account(provider), client_id.as_bytes())?;
    Ok(())
}

pub fn load_client_id(provider: &str) -> Result<Option<String>, OAuthError> {
    match keychain::get_secret(&client_id_keychain_account(provider))? {
        Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
        None => Ok(None),
    }
}

pub fn store_tokens(provider: &str, tokens: &TokenResponse) -> Result<(), OAuthError> {
    let stored = StoredTokens {
        access_token: tokens.access_token.clone(),
        refresh_token: tokens.refresh_token.clone(),
        expires_at: now_unix() + tokens.expires_in,
    };
    let json = serde_json::to_vec(&stored)?;
    keychain::set_secret(&keychain_account(provider), &json)?;
    Ok(())
}

pub fn load_tokens(provider: &str) -> Result<Option<StoredTokens>, OAuthError> {
    match keychain::get_secret(&keychain_account(provider))? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Disconnects a provider by forgetting its stored tokens — a fresh
/// `connect_*` (browser sign-in) is required to use it again. Deliberately
/// leaves the stored client ID untouched: re-connecting is then just the
/// consent flow, not re-pasting an Azure/Google/Zoom client ID that hasn't
/// changed. Real production path, driven by the UI's Disconnect button —
/// not test-only.
pub fn delete_tokens(provider: &str) -> Result<(), OAuthError> {
    keychain::delete_secret(&keychain_account(provider))?;
    Ok(())
}

/// Test-only alias: real Keychain/Credential-Manager/Secret-Service entries
/// persist across separate `cargo test` invocations (this is real OS
/// storage, not an in-memory fixture that resets itself) — any test
/// asserting a "nothing stored yet" precondition must actively clear that
/// state first, or it only passes once, on a machine that's never run it
/// before.
#[cfg(test)]
pub(crate) fn delete_tokens_for_test(provider: &str) -> Result<(), OAuthError> {
    delete_tokens(provider)
}

#[cfg(test)]
pub(crate) fn delete_client_id_for_test(provider: &str) -> Result<(), OAuthError> {
    keychain::delete_secret(&client_id_keychain_account(provider))?;
    Ok(())
}

/// The function calendar-polling code actually calls: returns a valid,
/// non-expired access token, transparently refreshing (and re-storing)
/// first if the cached one is within 60 seconds of expiring. Refreshing
/// proactively rather than reacting to a 401 avoids a wasted round-trip to
/// the calendar API on every call once a token's near end-of-life.
pub fn get_valid_access_token(provider: &str, config: &OAuthProviderConfig) -> Result<String, OAuthError> {
    let stored = load_tokens(provider)?.ok_or_else(|| OAuthError::NotConnected(provider.to_string()))?;
    if stored.expires_at - now_unix() > 60 {
        return Ok(stored.access_token);
    }
    let refresh_token = stored
        .refresh_token
        .ok_or_else(|| OAuthError::TokenResponse("access token expired and no refresh_token was ever stored".to_string()))?;
    let refreshed = refresh_access_token(config, &refresh_token)?;
    store_tokens(provider, &refreshed)?;
    Ok(refreshed.access_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_is_the_real_sha256_of_the_verifier() {
        let pair = PkcePair::generate().unwrap();
        assert_eq!(pair.verifier.len(), 43, "32 random bytes base64url-no-pad-encoded is always 43 chars");
        assert!(
            pair.verifier.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "verifier must be from RFC 7636's unreserved alphabet, got {}",
            pair.verifier
        );
        let expected_challenge = base64_url_no_pad(&Sha256::digest(pair.verifier.as_bytes()));
        assert_eq!(pair.challenge, expected_challenge);
    }

    #[test]
    fn two_generated_pairs_are_never_the_same() {
        let a = PkcePair::generate().unwrap();
        let b = PkcePair::generate().unwrap();
        assert_ne!(a.verifier, b.verifier, "PKCE verifiers must be unpredictable per auth attempt");
    }

    #[test]
    fn build_authorize_url_includes_pkce_and_provider_extras() {
        let config = OAuthProviderConfig {
            authorize_url: "https://example.com/authorize".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test-client".to_string(),
            scope: "calendar.readonly".to_string(),
            extra_authorize_params: vec![("access_type".to_string(), "offline".to_string())],
        };
        let pkce = PkcePair::generate().unwrap();
        let url = build_authorize_url(&config, &pkce, "http://127.0.0.1:48291/callback", "xyz-state");

        assert!(url.starts_with("https://example.com/authorize?"));
        assert!(url.contains("client_id=test-client"));
        assert!(url.contains(&format!("code_challenge={}", pkce.challenge)));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=xyz-state"));
        assert!(url.contains("access_type=offline"), "provider-specific extra param must be present");
    }

    #[test]
    fn parse_code_from_query_extracts_a_real_code_and_state() {
        let (code, state) = parse_code_from_query("/callback?state=abc&code=4%2F0AY0e-real-code&scope=email").unwrap();
        assert_eq!(code, "4/0AY0e-real-code", "percent-encoded '/' must be decoded");
        assert_eq!(state, Some("abc".to_string()));
    }

    #[test]
    fn parse_code_from_query_surfaces_a_real_denial() {
        let err = parse_code_from_query("/callback?error=access_denied&state=abc").unwrap_err();
        assert!(matches!(err, OAuthError::AuthorizeDenied(msg) if msg == "access_denied"));
    }

    #[test]
    fn parse_code_from_query_errors_when_neither_code_nor_error_present() {
        let err = parse_code_from_query("/callback?state=abc").unwrap_err();
        assert!(matches!(err, OAuthError::MissingCode));
    }

    // Real socket, real bytes over loopback TCP — simulates exactly what
    // a browser's redirect does, not a mocked function call.
    #[test]
    fn await_authorization_code_receives_a_real_redirect_over_a_real_socket() {
        let port = 48391;
        let handle = std::thread::spawn(move || await_authorization_code(port, Duration::from_secs(5)));

        // Give the listener a moment to bind before connecting.
        std::thread::sleep(Duration::from_millis(100));
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.write_all(b"GET /callback?code=real-test-code&state=xyz HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).ok();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "listener must respond so the browser tab shows success, got: {response}");

        let (code, state) = handle.join().unwrap().unwrap();
        assert_eq!(code, "real-test-code");
        assert_eq!(state, Some("xyz".to_string()));
    }

    #[test]
    fn await_authorization_code_times_out_if_nothing_connects() {
        let port = 48392;
        let result = await_authorization_code(port, Duration::from_millis(300));
        assert!(matches!(result, Err(OAuthError::Timeout)));
    }

    // Real HTTP POST over loopback against a real (fake) token endpoint —
    // proves the client's request shape and response parsing are correct
    // without needing a live third-party OAuth provider.
    fn spawn_fake_token_endpoint(port: u16, response_body: &'static str) {
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
    }

    #[test]
    fn exchange_code_for_tokens_parses_a_real_http_response() {
        let port = 48393;
        spawn_fake_token_endpoint(port, r#"{"access_token":"real-access","refresh_token":"real-refresh","expires_in":3600,"token_type":"Bearer"}"#);
        std::thread::sleep(Duration::from_millis(100));

        let config = OAuthProviderConfig {
            authorize_url: "unused".to_string(),
            token_url: format!("http://127.0.0.1:{port}/token"),
            client_id: "test-client".to_string(),
            scope: "unused".to_string(),
            extra_authorize_params: vec![],
        };
        let tokens = exchange_code_for_tokens(&config, "some-code", "some-verifier", "http://127.0.0.1:9999/callback").unwrap();
        assert_eq!(tokens.access_token, "real-access");
        assert_eq!(tokens.refresh_token, Some("real-refresh".to_string()));
        assert_eq!(tokens.expires_in, 3600);
    }

    #[test]
    fn refresh_preserves_the_original_refresh_token_when_the_response_omits_it() {
        let port = 48394;
        // Real behavior on Google/Microsoft: a refresh response omits
        // refresh_token entirely since it doesn't rotate.
        spawn_fake_token_endpoint(port, r#"{"access_token":"new-access","expires_in":3600}"#);
        std::thread::sleep(Duration::from_millis(100));

        let config = OAuthProviderConfig {
            authorize_url: "unused".to_string(),
            token_url: format!("http://127.0.0.1:{port}/token"),
            client_id: "test-client".to_string(),
            scope: "unused".to_string(),
            extra_authorize_params: vec![],
        };
        let refreshed = refresh_access_token(&config, "original-refresh-token").unwrap();
        assert_eq!(refreshed.access_token, "new-access");
        assert_eq!(refreshed.refresh_token, Some("original-refresh-token".to_string()), "must preserve the caller's refresh_token when the response omits it");
    }

    #[test]
    fn store_and_load_tokens_round_trip_through_real_secure_storage() {
        let provider = "test-provider-oauth-roundtrip";
        let tokens = TokenResponse { access_token: "stored-access".to_string(), refresh_token: Some("stored-refresh".to_string()), expires_in: 1800 };
        store_tokens(provider, &tokens).unwrap();

        let loaded = load_tokens(provider).unwrap().expect("tokens should have been stored");
        assert_eq!(loaded.access_token, "stored-access");
        assert_eq!(loaded.refresh_token, Some("stored-refresh".to_string()));
        assert!(loaded.expires_at > now_unix(), "expires_at should be computed from expires_in, in the future");
    }

    #[test]
    fn client_id_round_trips_through_real_secure_storage() {
        let provider = "test-provider-client-id-roundtrip";
        // Real Keychain/Credential-Manager/Secret-Service entries persist
        // across separate `cargo test` runs — clear first so this test is
        // actually rerunnable, not just "passes once on a clean machine."
        delete_client_id_for_test(provider).unwrap();
        assert_eq!(load_client_id(provider).unwrap(), None);
        store_client_id(provider, "00001111-aaaa-2222-bbbb-3333cccc4444").unwrap();
        assert_eq!(load_client_id(provider).unwrap(), Some("00001111-aaaa-2222-bbbb-3333cccc4444".to_string()));
    }

    #[test]
    fn get_valid_access_token_errors_clearly_when_never_connected() {
        let config = OAuthProviderConfig {
            authorize_url: "unused".to_string(),
            token_url: "unused".to_string(),
            client_id: "unused".to_string(),
            scope: "unused".to_string(),
            extra_authorize_params: vec![],
        };
        let err = get_valid_access_token("test-provider-never-connected", &config).unwrap_err();
        assert!(matches!(err, OAuthError::NotConnected(p) if p == "test-provider-never-connected"));
    }
}
