//! Frontier-model ("polish this summary") gating, per the Council's
//! non-negotiable: never a default pipeline step, explicit user request
//! only, capped at exactly one call per meeting, fed the local-generated
//! summary text — never the raw transcript.
//!
//! Ordering discipline matches the lesson learned in `retention.rs`:
//! audit the intent to call out BEFORE the network call happens, not
//! after. A crash mid-call must fail toward "attempt logged, nothing sent"
//! or "attempt logged, response received" — never toward "data left the
//! machine, zero record of it."

use crate::audit_log::AuditLog;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Haiku is the deliberate choice for a single "polish this summary" call —
/// cheap and fast, appropriate for light editorial polish of text a local
/// model already produced, not a task requiring frontier-tier reasoning.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

#[derive(Debug, Error)]
pub enum FrontierError {
    #[error("frontier polish already used for meeting {0} — capped at one call per meeting")]
    AlreadyCalledForMeeting(String),
    #[error("ANTHROPIC_API_KEY environment variable not set")]
    MissingApiKey,
    #[error("audit log error: {0}")]
    Audit(#[from] crate::audit_log::AuditLogError),
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("anthropic api returned an error: {0}")]
    ApiError(String),
    #[error("unexpected response shape from anthropic api: {0}")]
    UnexpectedResponse(String),
}

/// Tracks which meetings have already consumed their one allowed frontier
/// call. A real implementation backs this with the SQLCipher database (a
/// `frontier_polish_used` column on `meetings`, or a dedicated table); the
/// trait boundary keeps the cap-enforcement logic testable without a live
/// database, same pattern as `cloud_sync_gate::BaaStore`.
pub trait FrontierCallTracker {
    fn has_been_called(&self, meeting_id: &str) -> bool;
    fn mark_called(&mut self, meeting_id: &str);
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: &'static str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

pub struct FrontierClient {
    api_key: String,
    http: reqwest::blocking::Client,
}

impl FrontierClient {
    /// Reads the API key from the environment at call time — never
    /// hardcoded, never logged, never passed through the frontend.
    pub fn from_env() -> Result<Self, FrontierError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| FrontierError::MissingApiKey)?;
        Ok(Self {
            api_key,
            http: reqwest::blocking::Client::new(),
        })
    }

    fn polish_text(&self, summary_text: &str) -> Result<String, FrontierError> {
        let body = AnthropicRequest {
            model: DEFAULT_MODEL,
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: "user",
                content: format!(
                    "Polish this meeting summary for clarity and professional tone. \
                     Keep all factual content unchanged — only improve wording and flow:\n\n{summary_text}"
                ),
            }],
        };

        let response = self
            .http
            .post(ANTHROPIC_API_URL)
            .header("content-type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(FrontierError::ApiError(format!("{status}: {text}")));
        }

        let parsed: AnthropicResponse = response.json()?;
        parsed
            .content
            .into_iter()
            .find(|b| b.block_type == "text")
            .and_then(|b| b.text)
            .ok_or_else(|| FrontierError::UnexpectedResponse("no text content block in response".into()))
    }
}

/// The single entry point every UI/command layer must call — enforces the
/// cap, audits before and after, and never accepts raw transcript text as
/// input (the type signature only accepts `summary_text`, not a transcript).
pub fn request_polish(
    tracker: &mut impl FrontierCallTracker,
    client: &FrontierClient,
    audit: &AuditLog,
    meeting_id: &str,
    summary_text: &str,
) -> Result<String, FrontierError> {
    if tracker.has_been_called(meeting_id) {
        return Err(FrontierError::AlreadyCalledForMeeting(meeting_id.to_string()));
    }

    audit.append(
        "frontier_call_attempted",
        "user_explicit_request",
        json!({
            "meeting_id": meeting_id,
            "vendor": "anthropic",
            "model": DEFAULT_MODEL,
            "payload_char_count": summary_text.len(),
        }),
    )?;

    let result = client.polish_text(summary_text);

    match &result {
        Ok(_) => {
            tracker.mark_called(meeting_id);
            audit.append(
                "frontier_call_succeeded",
                "system",
                json!({ "meeting_id": meeting_id }),
            )?;
        }
        Err(e) => {
            audit.append(
                "frontier_call_failed",
                "system",
                json!({ "meeting_id": meeting_id, "error": e.to_string() }),
            )?;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::NamedTempFile;

    struct InMemoryTracker(HashSet<String>);

    impl FrontierCallTracker for InMemoryTracker {
        fn has_been_called(&self, meeting_id: &str) -> bool {
            self.0.contains(meeting_id)
        }
        fn mark_called(&mut self, meeting_id: &str) {
            self.0.insert(meeting_id.to_string());
        }
    }

    fn temp_audit() -> (AuditLog, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::remove_file(tmp.path()).ok();
        (AuditLog::new(tmp.path()), tmp)
    }

    #[test]
    fn second_request_for_same_meeting_is_rejected_without_a_network_call() {
        let mut tracker = InMemoryTracker(HashSet::new());
        tracker.mark_called("meeting-1"); // simulate a call already used

        let (audit, _tmp) = temp_audit();
        // A client with an obviously-invalid key — if this test somehow
        // reached the network call, it would fail differently (ApiError),
        // not AlreadyCalledForMeeting. Asserting the specific error variant
        // proves the cap check short-circuits before any HTTP attempt.
        let client = FrontierClient {
            api_key: "invalid".to_string(),
            http: reqwest::blocking::Client::new(),
        };

        let result = request_polish(&mut tracker, &client, &audit, "meeting-1", "some summary");
        assert!(matches!(result, Err(FrontierError::AlreadyCalledForMeeting(_))));

        // No audit entry should have been written for a request that was
        // rejected before the "about to call out" checkpoint.
        let entries = audit.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn missing_api_key_is_a_clean_error_not_a_panic() {
        // SAFETY: test-only env manipulation, single-threaded within this
        // test's scope for this specific variable.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
        let result = FrontierClient::from_env();
        assert!(matches!(result, Err(FrontierError::MissingApiKey)));
    }

    #[test]
    fn failed_call_is_audited_but_does_not_consume_the_cap() {
        let mut tracker = InMemoryTracker(HashSet::new());
        let (audit, _tmp) = temp_audit();
        let client = FrontierClient {
            api_key: "invalid-key-guaranteed-to-be-rejected".to_string(),
            http: reqwest::blocking::Client::new(),
        };

        let result = request_polish(&mut tracker, &client, &audit, "meeting-2", "a summary");
        assert!(result.is_err());
        assert!(!tracker.has_been_called("meeting-2"), "a failed call must not consume the cap");

        let entries = audit.read_all().unwrap();
        let event_types: Vec<&str> = entries.iter().map(|e| e.event_type.as_str()).collect();
        assert!(event_types.contains(&"frontier_call_attempted"));
        assert!(event_types.contains(&"frontier_call_failed"));
        assert!(audit.verify_chain().is_ok());
    }
}
