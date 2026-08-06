//! First-run model provisioning: downloads the five local AI models this
//! app depends on into the app's own data directory, instead of requiring
//! a manual `curl` per model (that was the real gap before this module —
//! see ISA Decisions, 2026-08-05). Every URL below was verified by an
//! exact byte-size match against the real model files already on this
//! dev machine, not assumed from the filename alone — multiple different
//! HuggingFace repos publish files under the identical filename with
//! different content, so a filename match alone is not real provenance.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProvisioningError {
    #[error("download failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),
    #[error("archive extraction failed: {0}")]
    Extract(String),
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    pub url: &'static str,
    /// Relative to the models directory.
    pub dest_relative: &'static str,
    pub expected_bytes: u64,
    /// Some models ship inside a compressed archive alongside files this
    /// app never uses (an int8 variant, python export scripts) — this
    /// names the one archive member actually needed; everything else in
    /// the archive is discarded after extraction.
    pub archive_member: Option<&'static str>,
}

pub const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec {
        name: "Whisper (multilingual ASR)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        dest_relative: "ggml-base.bin",
        expected_bytes: 147_951_465,
        archive_member: None,
    },
    ModelSpec {
        name: "Diarization (speaker segmentation)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        dest_relative: "diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        expected_bytes: 6_958_444,
        archive_member: Some("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    },
    ModelSpec {
        name: "Diarization (speaker embedding)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        dest_relative: "diarization/speaker-embedding.onnx",
        expected_bytes: 29_596_978,
        archive_member: None,
    },
    ModelSpec {
        name: "Summarization LLM (Llama-3.1-8B-Instruct)",
        url: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        dest_relative: "llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        expected_bytes: 4_920_739_232,
        archive_member: None,
    },
    ModelSpec {
        name: "Embeddings (bge-small-en-v1.5)",
        url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf",
        dest_relative: "embeddings/bge-small-en-v1.5-f16.gguf",
        expected_bytes: 67_308_128,
        archive_member: None,
    },
];

/// Which of the 5 models are missing from `models_dir`.
pub fn missing_models(models_dir: &Path) -> Vec<&'static ModelSpec> {
    MODEL_SPECS.iter().filter(|spec| !models_dir.join(spec.dest_relative).exists()).collect()
}

/// Downloads one model, calling `on_progress(bytes_downloaded, total_bytes)`
/// as data arrives. Downloads to a temp path and only lands at the final
/// destination on full success, so a crash or interrupted download never
/// leaves a partial file that `missing_models` would mistake for present.
pub fn download_model(
    spec: &ModelSpec,
    models_dir: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<(), ProvisioningError> {
    let final_dest = models_dir.join(spec.dest_relative);
    if let Some(parent) = final_dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let download_target = models_dir.join(format!("{}.download-tmp", spec.name.replace([' ', '(', ')'], "-")));

    let client = reqwest::blocking::Client::new();
    let mut response = client.get(spec.url).send()?.error_for_status()?;
    let total = response.content_length().unwrap_or(spec.expected_bytes);

    {
        let mut file = std::fs::File::create(&download_target)?;
        let mut buf = [0u8; 64 * 1024];
        let mut downloaded: u64 = 0;
        loop {
            let n = response.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            on_progress(downloaded, total);
        }
    }

    match spec.archive_member {
        Some(member) => {
            extract_archive_member(&download_target, member, &final_dest)?;
            std::fs::remove_file(&download_target)?;
        }
        None => {
            std::fs::rename(&download_target, &final_dest)?;
        }
    }

    Ok(())
}

/// Extracts exactly one member from a `.tar.bz2` archive, discarding the
/// rest — shells out to the system `tar` (present on macOS by default).
/// Windows support is deferred along with the rest of the Windows build,
/// per this project's own explicit build ordering.
fn extract_archive_member(archive_path: &Path, member: &str, dest: &Path) -> Result<(), ProvisioningError> {
    let extract_dir = archive_path.with_extension("extracted");
    std::fs::create_dir_all(&extract_dir)?;

    let status = std::process::Command::new("tar")
        .arg("-xjf")
        .arg(archive_path)
        .arg("-C")
        .arg(&extract_dir)
        .status()?;
    if !status.success() {
        return Err(ProvisioningError::Extract(format!("tar exited with status {status}")));
    }

    let extracted_member = extract_dir.join(member);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(&extracted_member, dest)?;
    std::fs::remove_dir_all(&extract_dir).ok();
    Ok(())
}

/// Resolves the real models directory for this run. Production/installed
/// builds always use `$APPDATA/models`. Dev builds fall back to the
/// already-populated source-tree `models/` directory when `$APPDATA/models`
/// hasn't been provisioned yet, so a working dev machine never needs to
/// re-download several GB of models it already has on disk. Release
/// builds never take the fallback branch — `cfg!(debug_assertions)` is
/// false in `--release`, and `CARGO_MANIFEST_DIR` wouldn't exist on an
/// installed machine's disk anyway.
pub fn resolve_models_dir(app_data_dir: &Path) -> PathBuf {
    let production_dir = app_data_dir.join("models");
    if cfg!(debug_assertions) {
        let dev_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        if missing_models(&production_dir).len() == MODEL_SPECS.len() && missing_models(&dev_dir).is_empty() {
            return dev_dir;
        }
    }
    production_dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_models_reports_all_five_for_an_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = missing_models(tmp.path());
        assert_eq!(missing.len(), MODEL_SPECS.len());
    }

    #[test]
    fn missing_models_is_empty_once_every_file_exists() {
        let tmp = tempfile::tempdir().unwrap();
        for spec in MODEL_SPECS {
            let path = tmp.path().join(spec.dest_relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"placeholder").unwrap();
        }
        assert!(missing_models(tmp.path()).is_empty());
    }

    #[test]
    fn resolve_models_dir_prefers_dev_source_tree_when_appdata_is_empty() {
        let tmp = tempfile::tempdir().unwrap();
        // Neither directory populated except the real dev source tree
        // (which genuinely has all 5 models on this machine) — this test
        // only makes a real assertion when that's true; skips otherwise.
        let dev_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        if !missing_models(&dev_dir).is_empty() {
            eprintln!("skipping: dev source tree doesn't have all 5 models in this environment");
            return;
        }
        let resolved = resolve_models_dir(tmp.path());
        assert_eq!(resolved, dev_dir);
    }
}
