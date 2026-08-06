# kai-notetaker

Local-first AI meeting notetaker. Everything — recording, transcription
(whisper.cpp), speaker diarization (sherpa-onnx), summarization and action
items (a local Llama-3.1-8B model) — runs on-device. No cloud calls happen
by default; a capped, explicit frontier-model "polish" call is opt-in per
meeting. The database is encrypted at rest (SQLCipher, key in the OS
Keychain).

macOS only for now (Apple Silicon). Windows support is planned but not
started.

## Prerequisites

- **Rust** (via [rustup](https://rustup.rs)) — stable toolchain
- **[Bun](https://bun.sh)** — this project uses `bun`, not `npm`/`yarn`
- **Xcode Command Line Tools** — `xcode-select --install`
- **CMake** — `brew install cmake` (needed to build the vendored
  whisper.cpp/llama.cpp/sherpa-onnx C++ dependencies)

## First-time setup

```bash
git clone https://github.com/jgutie31/kai-notetaker.git
cd kai-notetaker
bun install
bun run tauri dev
```

The first `bun run tauri dev` compiles whisper.cpp, llama.cpp, and
sherpa-onnx's ONNX Runtime bindings from source — this takes several
minutes. Subsequent runs are fast (incremental Rust builds).

## First launch: model download

The app needs five local AI models (~5.3GB total) that are **not** stored
in this repo — they're downloaded automatically on first launch into
`~/Library/Application Support/com.kairoscompliance.kainotetaker/models/`:

| Model | Purpose | Size |
|---|---|---|
| Whisper (multilingual, `ggml-base.bin`) | Speech-to-text | ~148MB |
| pyannote-segmentation-3.0 (via sherpa-onnx) | Speaker segmentation | ~7MB |
| 3D-Speaker CAM++ embedding | Speaker identity | ~30MB |
| Llama-3.1-8B-Instruct (Q4_K_M GGUF) | Summarization, action items | ~4.6GB |
| bge-small-en-v1.5 (GGUF) | Embeddings for search/Q&A | ~67MB |

On first launch you'll see a "Setting up kai-notetaker" screen with a
real per-model progress bar. This only happens once — after that, the
app checks the models directory and skips straight to the normal UI.

No manual `curl` commands, no bundling multi-GB files into the app
installer — see `src-tauri/src/model_provisioning.rs` for the real,
byte-verified download URLs.

## Encryption

The SQLite database is encrypted with SQLCipher. The encryption key is a
generated 256-bit value (never a passphrase) stored in the macOS
Keychain — macOS may prompt for Keychain access the first time (and
occasionally again across rebuilds of an unsigned dev build; this
stabilizes once the app is properly code-signed).

## Development

```bash
cd src-tauri
cargo test --lib      # Rust unit + integration tests (needs models downloaded)
cd ..
bunx tsc --noEmit      # Frontend typecheck
```

The project's `ISA.md` is the canonical system-of-record for what's
built, what's tested, and what's still open — read that before making
architectural changes.
