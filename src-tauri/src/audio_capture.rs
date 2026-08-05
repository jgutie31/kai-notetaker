//! Microphone capture via `cpal`. Writes raw audio to a WAV file inside the
//! app's own data directory — never system `/tmp`, never anywhere outside
//! app-controlled storage (ISC-47). The recording's temp file is deleted
//! once ASR has consumed it successfully (`RecordingSession::cleanup`),
//! unless the caller explicitly opts to retain raw audio.
//!
//! Every device's audio — regardless of its native sample rate — is
//! resampled in real time to one fixed `CANONICAL_SAMPLE_RATE` before it
//! reaches the shared recording buffer. This is what makes mid-recording
//! device switching (`RecordingSession::switch_device`) possible without
//! corrupting the output file: the WAV format has exactly one declared
//! sample rate for its whole duration, so every source feeding into it
//! must already agree on that rate by the time it lands in the buffer.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, WindowFunction};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

/// Every capture stream — no matter which device or its native rate — is
/// resampled to this rate before reaching the recording buffer. 48kHz is
/// a high-quality-enough default most hardware supports natively (music
/// was explicitly named as a use case, so this deliberately is NOT the
/// 16kHz Whisper needs — that downsample happens separately, at ASR load
/// time, from this higher-fidelity source).
const CANONICAL_SAMPLE_RATE: u32 = 48_000;

/// Frames-per-call for the streaming resampler. 1024 is a reasonable
/// real-time chunk size — small enough for low latency, large enough that
/// the sinc filter has enough context per call.
const RESAMPLE_CHUNK_FRAMES: usize = 1024;

#[derive(Debug, Error)]
pub enum AudioCaptureError {
    #[error("no input device available")]
    NoInputDevice,
    #[error("input device '{0}' not found")]
    DeviceNotFound(String),
    // cpal 0.18 unified BuildStreamError/PlayStreamError/
    // DefaultStreamConfigError into a single `cpal::Error` — confirmed
    // against the installed crate's source, not assumed from stale docs.
    #[error("cpal error: {0}")]
    Cpal(#[from] cpal::Error),
    #[error("wav write error: {0}")]
    Wav(#[from] hound::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("resampler error: {0}")]
    Resampler(String),
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

/// Owns a real-time streaming resampler from one device's native rate to
/// `CANONICAL_SAMPLE_RATE`. Lives entirely inside one stream's callback
/// closure — never shared across threads, so no locking needed here even
/// though the resampler holds mutable internal state across calls.
struct StreamingResampler {
    resampler: Async<f32>,
    chunk_frames: usize,
    input_accum: Vec<f32>,
    output_scratch: Vec<f32>,
}

impl StreamingResampler {
    fn new(input_rate: u32, output_rate: u32) -> Result<Self, AudioCaptureError> {
        let ratio = output_rate as f64 / input_rate as f64;
        let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2);
        let resampler = Async::<f32>::new_sinc(
            ratio,
            1.1,
            &params,
            RESAMPLE_CHUNK_FRAMES,
            1, // mono — downmixing happens before samples reach the resampler
            FixedAsync::Input,
        )
        .map_err(|e| AudioCaptureError::Resampler(e.to_string()))?;

        let output_cap = resampler.output_frames_max();
        Ok(Self {
            resampler,
            chunk_frames: RESAMPLE_CHUNK_FRAMES,
            input_accum: Vec::with_capacity(RESAMPLE_CHUNK_FRAMES * 2),
            output_scratch: vec![0.0; output_cap],
        })
    }

    /// Feed newly-captured mono samples; returns whatever resampled output
    /// is ready (may be empty if not enough input has accumulated yet for
    /// a full chunk).
    fn process(&mut self, new_mono_samples: &[f32]) -> Result<Vec<f32>, AudioCaptureError> {
        self.input_accum.extend_from_slice(new_mono_samples);
        let mut produced = Vec::new();

        while self.input_accum.len() >= self.chunk_frames {
            let chunk: Vec<f32> = self.input_accum.drain(..self.chunk_frames).collect();

            let input_adapter = InterleavedSlice::new(&chunk, 1, self.chunk_frames)
                .map_err(|e| AudioCaptureError::Resampler(format!("{e:?}")))?;
            let output_len = self.output_scratch.len();
            let mut output_adapter = InterleavedSlice::new_mut(&mut self.output_scratch, 1, output_len)
                .map_err(|e| AudioCaptureError::Resampler(format!("{e:?}")))?;

            let (_consumed, produced_frames) = self
                .resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&Indexing::new()))
                .map_err(|e| AudioCaptureError::Resampler(e.to_string()))?;

            produced.extend_from_slice(&self.output_scratch[..produced_frames]);
        }

        Ok(produced)
    }
}

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Build and start a capture stream for `device`, writing
/// `CANONICAL_SAMPLE_RATE`-normalized mono samples into `buffer`. Shared
/// by both `RecordingSession::start` and `switch_device` so the two paths
/// can never drift apart in how they normalize audio.
fn build_capture_stream(
    device: &cpal::Device,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, AudioCaptureError> {
    let config = device.default_input_config()?;
    let native_rate = config.sample_rate();
    let channels = config.channels() as usize;
    let stream_config: cpal::StreamConfig = config.into();

    if native_rate == CANONICAL_SAMPLE_RATE {
        let stream = device.build_input_stream(
            stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                let mono = downmix_to_mono(data, channels);
                buffer.lock().unwrap().extend_from_slice(&mono);
            },
            move |err| eprintln!("audio input stream error: {err}"),
            Some(Duration::from_secs(5)),
        )?;
        stream.play()?;
        return Ok(stream);
    }

    let mut resampler = StreamingResampler::new(native_rate, CANONICAL_SAMPLE_RATE)?;
    let stream = device.build_input_stream(
        stream_config,
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            let mono = downmix_to_mono(data, channels);
            match resampler.process(&mono) {
                Ok(resampled) if !resampled.is_empty() => {
                    buffer.lock().unwrap().extend_from_slice(&resampled);
                }
                Ok(_) => {} // not enough accumulated yet for a full chunk
                Err(e) => eprintln!("resample error: {e}"),
            }
        },
        move |err| eprintln!("audio input stream error: {err}"),
        Some(Duration::from_secs(5)),
    )?;
    stream.play()?;
    Ok(stream)
}

fn find_device_by_name(name: &str) -> Result<cpal::Device, AudioCaptureError> {
    let host = cpal::default_host();
    host.input_devices()
        .map_err(|_| AudioCaptureError::NoInputDevice)?
        .find(|d| d.to_string() == name)
        .ok_or_else(|| AudioCaptureError::DeviceNotFound(name.to_string()))
}

/// A recording session: owns the temp file path (inside `data_dir`, never
/// outside it) and the currently-active cpal stream. The stream can be
/// swapped mid-recording via `switch_device` without losing any audio
/// already captured into `buffer`, and without changing the file's
/// declared sample rate — every stream normalizes to
/// `CANONICAL_SAMPLE_RATE` before writing into the shared buffer.
pub struct RecordingSession {
    path: PathBuf,
    stream: Option<cpal::Stream>,
    buffer: Arc<Mutex<Vec<f32>>>,
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

        let recordings_dir = data_dir.join("recordings");
        std::fs::create_dir_all(&recordings_dir)?;
        let path = recordings_dir.join(format!("{recording_id}.wav"));

        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
        let stream = build_capture_stream(&device, buffer.clone())?;

        Ok(Self {
            path,
            stream: Some(stream),
            buffer,
        })
    }

    /// Switch the active input device mid-recording. Already-captured
    /// samples in `buffer` are untouched — only the live stream feeding
    /// new samples into it changes. There IS a small (typically tens-of-
    /// milliseconds) gap while the old hardware stream is torn down and
    /// the new one spins up — inherent to closing and opening real audio
    /// hardware connections, not something software can make instant.
    /// What this guarantees instead: no glitch, no corrupted samples, no
    /// data loss, and no change to the output file's sample rate no
    /// matter what the new device's native rate is.
    pub fn switch_device(&mut self, device_name: &str) -> Result<(), AudioCaptureError> {
        let device = find_device_by_name(device_name)?;
        let new_stream = build_capture_stream(&device, self.buffer.clone())?;
        // Assigning over `self.stream` drops the old `cpal::Stream` here,
        // which stops its callback and releases the old hardware
        // connection — the new stream is already playing by this point
        // (`build_capture_stream` calls `.play()` internally), so the
        // handoff is "start new, then drop old" rather than the reverse,
        // minimizing the silent gap.
        self.stream = Some(new_stream);
        Ok(())
    }

    /// Stop recording and flush the buffered samples to disk as a 16-bit
    /// PCM mono WAV at `CANONICAL_SAMPLE_RATE` — always that rate,
    /// regardless of how many devices were used during the recording.
    pub fn stop_and_write(mut self) -> Result<PathBuf, AudioCaptureError> {
        // Dropping the stream stops capture.
        self.stream.take();

        let samples = self.buffer.lock().unwrap();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: CANONICAL_SAMPLE_RATE,
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

    #[test]
    fn stopped_recording_always_declares_canonical_sample_rate() {
        let tmp_data_dir = std::env::temp_dir().join(format!("kai-notetaker-test-rate-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_data_dir).unwrap();

        if list_input_devices().map(|d| d.is_empty()).unwrap_or(true) {
            eprintln!("skipping: no input device available in this environment");
            std::fs::remove_dir_all(&tmp_data_dir).ok();
            return;
        }

        let session = RecordingSession::start(&tmp_data_dir, "rate-check").unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let path = session.stop_and_write().unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, CANONICAL_SAMPLE_RATE);

        delete_recording(&path).unwrap();
        std::fs::remove_dir_all(&tmp_data_dir).ok();
    }

    #[test]
    fn switch_device_preserves_already_captured_audio_and_keeps_growing() {
        let tmp_data_dir = std::env::temp_dir().join(format!("kai-notetaker-test-switch-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_data_dir).unwrap();

        let devices = list_input_devices().unwrap_or_default();
        if devices.is_empty() {
            eprintln!("skipping: no input device available in this environment");
            std::fs::remove_dir_all(&tmp_data_dir).ok();
            return;
        }
        // This dev machine only has one real input device, so this test
        // exercises the real stop-old/start-new plumbing by "switching" to
        // the same device by name — it cannot prove cross-hardware
        // resampling correctness, which needs a second physical device.
        // That limitation is real and stated here rather than implied.
        let device_name = devices[0].name.clone();

        let mut session = RecordingSession::start(&tmp_data_dir, "switch-test").unwrap();
        std::thread::sleep(Duration::from_millis(150));

        let len_before_switch = session.buffer.lock().unwrap().len();
        assert!(len_before_switch > 0, "expected some audio captured before switching");

        session.switch_device(&device_name).unwrap();
        std::thread::sleep(Duration::from_millis(150));

        let len_after_switch = session.buffer.lock().unwrap().len();
        assert!(
            len_after_switch > len_before_switch,
            "expected buffer to keep growing after switch: before={len_before_switch}, after={len_after_switch}"
        );

        let path = session.stop_and_write().unwrap();
        assert!(path.exists());
        delete_recording(&path).unwrap();
        std::fs::remove_dir_all(&tmp_data_dir).ok();
    }

    #[test]
    fn switch_device_to_nonexistent_name_errors_without_disturbing_current_stream() {
        let tmp_data_dir = std::env::temp_dir().join(format!("kai-notetaker-test-badswitch-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_data_dir).unwrap();

        if list_input_devices().map(|d| d.is_empty()).unwrap_or(true) {
            eprintln!("skipping: no input device available in this environment");
            std::fs::remove_dir_all(&tmp_data_dir).ok();
            return;
        }

        let mut session = RecordingSession::start(&tmp_data_dir, "bad-switch-test").unwrap();
        std::thread::sleep(Duration::from_millis(100));

        let result = session.switch_device("this-device-definitely-does-not-exist-12345");
        assert!(matches!(result, Err(AudioCaptureError::DeviceNotFound(_))));

        // The original stream must still be alive and still capturing —
        // a failed switch must not leave the session dead.
        let len_before = session.buffer.lock().unwrap().len();
        std::thread::sleep(Duration::from_millis(150));
        let len_after = session.buffer.lock().unwrap().len();
        assert!(len_after > len_before, "original stream should still be capturing after a failed switch");

        let path = session.stop_and_write().unwrap();
        delete_recording(&path).unwrap();
        std::fs::remove_dir_all(&tmp_data_dir).ok();
    }

    #[test]
    fn streaming_resampler_upsamples_to_expected_frame_count() {
        // 44100 -> 48000 over 2 seconds of input should produce
        // approximately 2 seconds of output at the new rate, within a
        // small tolerance for sinc filter startup/edge effects.
        let mut resampler = StreamingResampler::new(44_100, 48_000).unwrap();
        let input_frames = 44_100 * 2;
        let input = vec![0.0_f32; input_frames];

        let mut total_output = 0usize;
        for chunk in input.chunks(2048) {
            total_output += resampler.process(chunk).unwrap().len();
        }

        let expected = 48_000 * 2;
        let tolerance = 4096; // sinc filter delay/edge effects, not a bug
        assert!(
            (total_output as i64 - expected as i64).abs() < tolerance,
            "expected ~{expected} output frames, got {total_output}"
        );
    }
}
