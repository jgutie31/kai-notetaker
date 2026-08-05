//! Microphone capture via `cpal`. Writes raw audio to a WAV file inside the
//! app's own data directory — never system `/tmp`, never anywhere outside
//! app-controlled storage (ISC-47). The recording's temp file is deleted
//! once ASR has consumed it successfully (`RecordingSession::cleanup`),
//! unless the caller explicitly opts to retain raw audio.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioCaptureError {
    #[error("no input device available")]
    NoInputDevice,
    // cpal 0.18 unified BuildStreamError/PlayStreamError/
    // DefaultStreamConfigError into a single `cpal::Error` — confirmed
    // against the installed crate's source, not assumed from stale docs.
    #[error("cpal error: {0}")]
    Cpal(#[from] cpal::Error),
    #[error("wav write error: {0}")]
    Wav(#[from] hound::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize)]
pub struct InputDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// List available input devices. Real cpal call, no mocking — this either
/// reflects actual hardware or it doesn't (ISC-43).
pub fn list_input_devices() -> Result<Vec<InputDeviceInfo>, AudioCaptureError> {
    let host = cpal::default_host();
    // cpal 0.18 removed `Device::name()` in favor of `Display` — use
    // `to_string()`, confirmed against the installed crate's source.
    let default_name = host.default_input_device().map(|d| d.to_string());

    let mut out = Vec::new();
    for device in host.input_devices().map_err(|_| AudioCaptureError::NoInputDevice)? {
        let name = device.to_string();
        let is_default = default_name.as_deref() == Some(name.as_str());
        out.push(InputDeviceInfo { name, is_default });
    }
    Ok(out)
}

/// A recording session: owns the temp file path (inside `data_dir`, never
/// outside it) and the cpal stream while it's live.
pub struct RecordingSession {
    path: PathBuf,
    stream: Option<cpal::Stream>,
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

impl RecordingSession {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Start recording from the default input device into a WAV file at
    /// `data_dir/recordings/<id>.wav`. `data_dir` MUST be the app's own
    /// data directory — this function does not resolve or validate that
    /// itself (the caller, `lib.rs`, owns that resolution), but it never
    /// falls back to a system temp directory on its own.
    pub fn start(data_dir: &Path, recording_id: &str) -> Result<Self, AudioCaptureError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioCaptureError::NoInputDevice)?;
        let config = device.default_input_config()?;
        // cpal 0.18's `SampleRate` is a plain `u32` type alias now, not a
        // tuple struct — confirmed against the installed crate's source.
        let sample_rate = config.sample_rate();
        let channels = config.channels() as usize;

        let recordings_dir = data_dir.join("recordings");
        std::fs::create_dir_all(&recordings_dir)?;
        let path = recordings_dir.join(format!("{recording_id}.wav"));

        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
        let buffer_clone = buffer.clone();

        let stream_config: cpal::StreamConfig = config.into();
        let stream = device.build_input_stream(
            stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                let mut buf = buffer_clone.lock().unwrap();
                if channels > 1 {
                    // Downmix to mono by averaging channels.
                    for frame in data.chunks(channels) {
                        let avg = frame.iter().sum::<f32>() / channels as f32;
                        buf.push(avg);
                    }
                } else {
                    buf.extend_from_slice(data);
                }
            },
            move |err| eprintln!("audio input stream error: {err}"),
            Some(Duration::from_secs(5)),
        )?;

        stream.play()?;

        Ok(Self {
            path,
            stream: Some(stream),
            buffer,
            sample_rate,
        })
    }

    /// Stop recording and flush the buffered samples to disk as a 16-bit
    /// PCM mono WAV at whatever sample rate the device actually captured
    /// at (resampling to whisper's required 16kHz is ASR's job at load
    /// time, not this module's — keeps the raw capture format honest).
    pub fn stop_and_write(mut self) -> Result<PathBuf, AudioCaptureError> {
        // Dropping the stream stops capture.
        self.stream.take();

        let samples = self.buffer.lock().unwrap();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&self.path, spec)?;
        for &s in samples.iter() {
            let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(clamped)?;
        }
        writer.finalize()?;

        Ok(self.path.clone())
    }
}

/// Delete a recording's temp file. Called after ASR has successfully
/// consumed it, unless the caller opted to retain raw audio (ISC-46).
pub fn delete_recording(path: &Path) -> Result<(), AudioCaptureError> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_input_devices_does_not_error_on_this_machine() {
        // Real hardware probe. If this environment genuinely has no input
        // device, that's a legitimate empty Vec, not an error — only a
        // host-level enumeration failure should Err.
        let result = list_input_devices();
        assert!(result.is_ok(), "device enumeration should not error: {result:?}");
    }

    #[test]
    fn recording_path_is_inside_provided_data_dir_never_system_temp() {
        let tmp_data_dir = std::env::temp_dir().join(format!("kai-notetaker-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_data_dir).unwrap();

        // Only run the real recording if a device exists — CI/headless
        // environments may have none, and that's not this test's concern.
        if list_input_devices().map(|d| d.is_empty()).unwrap_or(true) {
            eprintln!("skipping: no input device available in this environment");
            std::fs::remove_dir_all(&tmp_data_dir).ok();
            return;
        }

        let session = RecordingSession::start(&tmp_data_dir, "test-recording").unwrap();
        let path = session.path().to_path_buf();
        assert!(path.starts_with(&tmp_data_dir), "recording path {path:?} escaped data dir {tmp_data_dir:?}");

        std::thread::sleep(Duration::from_millis(200));
        let written_path = session.stop_and_write().unwrap();
        assert!(written_path.exists());
        assert_eq!(written_path, path);

        delete_recording(&written_path).unwrap();
        assert!(!written_path.exists());

        std::fs::remove_dir_all(&tmp_data_dir).ok();
    }
}
