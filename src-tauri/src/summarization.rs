//! Chunked map-reduce meeting summarization + structured action-item
//! extraction, per the Council's converged recommendation: chunk at
//! ~2K-token windows with 200-token overlap, summarize each chunk
//! (map), combine chunk summaries into one meeting summary (reduce),
//! then extract action items as structured JSON — never freeform text
//! parsed by regex.
//!
//! Everything in this module runs through the already-loaded local
//! `LlmEngine` (Llama-3.1-8B-Instruct). No frontier-model calls happen
//! here — that's `frontier.rs`, invoked separately and only on explicit
//! user request (ISC-68/69).

use crate::llm::{LlmEngine, LlmError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SummarizationError {
    #[error("llm error: {0}")]
    Llm(#[from] LlmError),
    #[error("failed to parse action items JSON from model output: {0}")]
    ActionItemsParse(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionItem {
    pub description: String,
    pub owner: Option<String>,
    pub due_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSummary {
    pub chunk_summaries: Vec<String>,
    pub meeting_summary: String,
    pub action_items: Vec<ActionItem>,
}

const CHUNK_WINDOW_TOKENS: usize = 2000;
const CHUNK_OVERLAP_TOKENS: usize = 200;

fn chunk_summary_prompt(chunk: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
         Summarize the following meeting transcript excerpt in 2-4 sentences. \
         Focus on decisions, facts, and commitments — skip filler and small talk. \
         The excerpt may be very short, informal, or a test recording — that is fine. \
         Describe plainly whatever was actually said, even if it's just one sentence \
         or sounds like a sound check. Do NOT say the transcript is missing, unclear, \
         insufficient, or that you are unable to summarize it — there is always \
         something to report, even if it's simply that someone did a brief test \
         recording.\n\n\
         Excerpt:\n{chunk}\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

fn reduce_prompt(combined_chunk_summaries: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
         The following are summaries of consecutive excerpts from one meeting. \
         Combine them into a single coherent meeting summary of 4-8 sentences. \
         Do not repeat the same point twice; merge overlapping content.\n\n\
         Excerpt summaries:\n{combined_chunk_summaries}\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

fn action_items_prompt(meeting_summary: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
         Extract action items from this meeting summary as a JSON array. \
         Each item must have exactly these fields: \"description\" (string, required), \
         \"owner\" (string or null), \"due_date\" (string or null). \
         If there are no action items, return an empty array []. \
         Return ONLY the JSON array, no other text, no markdown code fences.\n\n\
         Meeting summary:\n{meeting_summary}\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

/// Local models reliably ignore "return only JSON" instructions and add
/// commentary, markdown fences, or leading/trailing prose. Extract the
/// first well-formed `[...]` substring rather than trusting the whole
/// response is clean JSON.
fn extract_json_array(raw: &str) -> Result<Vec<ActionItem>, SummarizationError> {
    let start = raw
        .find('[')
        .ok_or_else(|| SummarizationError::ActionItemsParse(format!("no '[' found in output: {raw:?}")))?;
    let end = raw
        .rfind(']')
        .ok_or_else(|| SummarizationError::ActionItemsParse(format!("no ']' found in output: {raw:?}")))?;
    if end < start {
        return Err(SummarizationError::ActionItemsParse(format!(
            "malformed brackets in output: {raw:?}"
        )));
    }
    let json_slice = &raw[start..=end];
    serde_json::from_str(json_slice)
        .map_err(|e| SummarizationError::ActionItemsParse(format!("{e} — extracted slice: {json_slice:?}")))
}

/// Run the full map-reduce pipeline: chunk -> per-chunk summary (map) ->
/// combined meeting summary (reduce) -> structured action-item extraction.
pub fn summarize_meeting(
    engine: &LlmEngine,
    transcript: &str,
    n_ctx: u32,
) -> Result<MeetingSummary, SummarizationError> {
    let chunks = engine.chunk_by_tokens(transcript, CHUNK_WINDOW_TOKENS, CHUNK_OVERLAP_TOKENS)?;

    let mut chunk_summaries = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let prompt = chunk_summary_prompt(chunk);
        let summary = engine.complete(&prompt, 200, n_ctx)?;
        chunk_summaries.push(summary.trim().to_string());
    }

    // Real bug fixed here: running the "combine multiple summaries into
    // one" reduce prompt when there was only ever ONE chunk gives the
    // model nothing real to reduce across, and — especially for a short,
    // thin transcript (a few seconds of audio) — it tends to respond with
    // meta-commentary ("there is no meeting transcript provided...")
    // instead of a plain answer. Skip the redundant reduce pass entirely
    // when there's nothing to reduce; the single chunk summary already
    // IS the meeting summary in that case.
    let meeting_summary = if chunk_summaries.len() <= 1 {
        chunk_summaries.first().cloned().unwrap_or_default()
    } else {
        let combined = chunk_summaries.join("\n\n");
        engine.complete(&reduce_prompt(&combined), 400, n_ctx)?.trim().to_string()
    };

    let action_items_raw = engine.complete(&action_items_prompt(&meeting_summary), 500, n_ctx)?;
    let action_items = extract_json_array(&action_items_raw)?;

    Ok(MeetingSummary {
        chunk_summaries,
        meeting_summary,
        action_items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmEngine;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    }

    #[test]
    fn extract_json_array_handles_clean_output() {
        let raw = r#"[{"description": "Send the proposal", "owner": "Jeremiah", "due_date": "2026-08-10"}]"#;
        let items = extract_json_array(raw).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "Send the proposal");
        assert_eq!(items[0].owner, Some("Jeremiah".to_string()));
    }

    #[test]
    fn extract_json_array_handles_model_commentary_around_json() {
        let raw = "Sure! Here are the action items:\n```json\n[{\"description\": \"Follow up with Brian\", \"owner\": null, \"due_date\": null}]\n```\nLet me know if you need anything else.";
        let items = extract_json_array(raw).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "Follow up with Brian");
        assert_eq!(items[0].owner, None);
    }

    #[test]
    fn extract_json_array_handles_empty_array() {
        let items = extract_json_array("[]").unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn extract_json_array_errors_on_no_brackets() {
        let result = extract_json_array("There are no action items in this meeting.");
        assert!(result.is_err());
    }

    #[test]
    fn short_thin_transcript_gets_a_direct_summary_not_meta_commentary() {
        // Reproduces a real bug hit in production testing: a ~5-second
        // recording (one short sentence, plus a garbled non-English
        // fragment) produced a "summary" that was actually the model
        // refusing to summarize ("It appears there is no meeting
        // transcript excerpt provided...") instead of a plain answer.
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: LLM model not present in this environment");
            return;
        }
        let engine = LlmEngine::load(&path, 1000).unwrap();

        let transcript = "Speaker 1: Test in one, two, three recording.\nSpeaker 1: For a van do uno doz tres.";
        let result = summarize_meeting(&engine, transcript, 4096).unwrap();

        assert!(!result.meeting_summary.trim().is_empty());
        let lower = result.meeting_summary.to_lowercase();
        assert!(
            !lower.contains("no meeting transcript") && !lower.contains("unable to create"),
            "summary regressed into meta-commentary/refusal instead of a direct answer: {}",
            result.meeting_summary
        );
    }

    #[test]
    fn full_pipeline_on_real_model_produces_summary_and_valid_action_items() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: LLM model not present in this environment");
            return;
        }
        let engine = LlmEngine::load(&path, 1000).unwrap();

        let transcript = "Jeremiah: Let's review the Smithville PCI-DSS scoping call. \
             Nesta: I've drafted the PAR worksheet, Dave needs to fill it out before Thursday. \
             Jeremiah: Good, can you send it to him by end of day tomorrow? \
             Nesta: Yes, I'll send it tomorrow morning. \
             Jeremiah: Also we need Brian to review the subcontractor agreement before we sign anything.";

        let result = summarize_meeting(&engine, transcript, 4096).unwrap();

        assert!(!result.chunk_summaries.is_empty());
        assert!(!result.meeting_summary.trim().is_empty());
        // Real content check, not just non-empty — the summary should be
        // about the actual meeting, not generic filler.
        let lower = result.meeting_summary.to_lowercase();
        assert!(
            lower.contains("smithville") || lower.contains("par") || lower.contains("worksheet") || lower.contains("dave"),
            "meeting summary didn't reference the actual content: {}",
            result.meeting_summary
        );
        // Action items is a valid Vec (possibly empty) — the real
        // assertion is that JSON parsing succeeded at all against real
        // model output, not that the model perfectly extracted every item.
        for item in &result.action_items {
            assert!(!item.description.trim().is_empty());
        }
    }
}
