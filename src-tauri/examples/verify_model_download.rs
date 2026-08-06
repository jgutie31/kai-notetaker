//! Real end-to-end verification of the first-run model download flow
//! (ISC-111, previously DEFERRED-VERIFY since it would force re-downloading
//! ~5.3GB already present in the dev source tree). Downloads all 5 real
//! models into the actual production directory ($APPDATA/models) — the
//! same directory `resolve_models_dir` falls back to once it's fully
//! populated — so this both proves the download path works AND leaves
//! the app in the same state a real installed copy would be in.
//!
//! `cargo run --example verify_model_download`

use kai_notetaker_lib::model_provisioning::{self, MODEL_SPECS};
use std::path::PathBuf;

fn main() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let models_dir = PathBuf::from(&home)
        .join("Library/Application Support/com.kairoscompliance.kainotetaker/models");

    let missing_before = model_provisioning::missing_models(&models_dir);
    println!("{} of {} models missing before this run", missing_before.len(), MODEL_SPECS.len());

    let mut failed = 0;
    for spec in MODEL_SPECS {
        let dest = models_dir.join(spec.dest_relative);
        if dest.exists() {
            println!("--- {} already present, skipping ---", spec.name);
            continue;
        }

        println!("--- downloading {} (expected {} bytes) ---", spec.name, spec.expected_bytes);
        let mut last_percent = 0u64;
        let result = model_provisioning::download_model(spec, &models_dir, |downloaded, total| {
            let percent = if total > 0 { downloaded * 100 / total } else { 0 };
            if percent >= last_percent + 10 {
                println!("  {} : {}% ({}/{} bytes)", spec.name, percent, downloaded, total);
                last_percent = percent;
            }
        });

        match result {
            Ok(()) => {
                let actual_bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                if spec.archive_member.is_some() {
                    // `expected_bytes` is the compressed archive's download
                    // size for these specs, not the extracted file's size —
                    // comparing them directly is an apples-to-oranges bug,
                    // not a real integrity check. Just confirm the
                    // extracted file is non-empty here.
                    if actual_bytes > 0 {
                        println!("OK: {} — extracted {} bytes (archive member)", spec.name, actual_bytes);
                    } else {
                        println!("MISMATCH: {} — extracted file is empty", spec.name);
                        failed += 1;
                    }
                } else if actual_bytes == spec.expected_bytes {
                    println!("OK: {} — {} bytes, exact match", spec.name, actual_bytes);
                } else {
                    println!(
                        "MISMATCH: {} — expected {} bytes, got {}",
                        spec.name, spec.expected_bytes, actual_bytes
                    );
                    failed += 1;
                }
            }
            Err(e) => {
                eprintln!("FAILED: {} — {}", spec.name, e);
                failed += 1;
            }
        }
    }

    let missing_after = model_provisioning::missing_models(&models_dir);
    println!(
        "\n=== verification complete: {} failed, {} of {} still missing ===",
        failed,
        missing_after.len(),
        MODEL_SPECS.len()
    );
    if failed > 0 || !missing_after.is_empty() {
        std::process::exit(1);
    }
}
