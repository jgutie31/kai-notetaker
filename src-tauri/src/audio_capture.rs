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
pub const CANONICAL_SAMPLE_RATE: u32 = 48_000;

/// The sample rate ASR (`whisper-rs`) and diarization (`sherpa-onnx`
/// pyannote-segmentation-3.0) both hard-require. Recordings are captured
/// at `CANONICAL_SAMPLE_RATE` (48kHz, chosen for music-quality capture),
/// so this batch resample is the explicit, visible step that bridges the
/// two — never a silent default inside the ASR/diarization layers
/// themselves (see `asr::read_wav_as_f32_mono_16k`'s doc comment).
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

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

/// One-shot (not real-time streaming) resample of a complete mono buffer
/// from `input_rate` to `output_rate`. Reuses the same `StreamingResampler`
/// already tested for live device switching, feeding it the whole input
/// plus a zero-padded flush chunk at the end so the sinc filter's trailing
/// context gets processed — otherwise the last fractional chunk (at most
/// `RESAMPLE_CHUNK_FRAMES` frames, a few milliseconds) would be silently
/// dropped rather than resampled.
pub fn one_shot_resample(samples: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>, AudioCaptureError> {
    if input_rate == output_rate {
        return Ok(samples.to_vec());
    }
    let mut resampler = StreamingResampler::new(input_rate, output_rate)?;
    let mut output = resampler.process(samples)?;

    let flush_padding = vec![0.0_f32; RESAMPLE_CHUNK_FRAMES];
    output.extend(resampler.process(&flush_padding)?);

    Ok(output)
}

/// RMS (root-mean-square) energy of the trailing `window_secs` of
/// `samples`, at `CANONICAL_SAMPLE_RATE`. `None` when the buffer doesn't
/// yet hold a full window — an honest "not enough audio to judge yet"
/// rather than a misleadingly-quiet reading computed from a partial
/// window.
///
/// Split out from `RecordingSession::trailing_rms` as a free function on a
/// plain slice so the real math is unit-testable against a known
/// sine-wave/silence buffer without needing live capture hardware — the
/// same pure-core/side-effecting-shell split `auto_join.rs` uses.
pub fn trailing_rms_of(samples: &[f32], window_secs: f32) -> Option<f32> {
    if !window_secs.is_finite() || window_secs <= 0.0 {
        return None;
    }
    let window_len = (window_secs as f64 * CANONICAL_SAMPLE_RATE as f64).round() as usize;
    if window_len == 0 || samples.len() < window_len {
        return None;
    }
    let tail = &samples[samples.len() - window_len..];
    // f64 accumulation: a 60-second window at 48kHz is 2.88M terms, enough
    // that f32 summation would lose real precision on quiet audio — which
    // is exactly the regime this measurement has to be trustworthy in.
    let sum_squares: f64 = tail.iter().map(|&s| (s as f64) * (s as f64)).sum();
    Some((sum_squares / window_len as f64).sqrt() as f32)
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

    /// RMS energy over the trailing `window_secs` of already-captured
    /// audio, **without disturbing the live capture stream** (ISC-200):
    /// this only takes a read lock on the shared buffer, exactly as the
    /// capture callback itself does to append — no stream teardown, no
    /// draining, no mutation of a single sample.
    ///
    /// `None` while fewer than `window_secs` of audio have been captured.
    ///
    /// Cost note, since this runs on an interval against a live recording:
    /// a 60-second window is ~2.88M samples (~11.5MB), so the buffer mutex
    /// is held for roughly a millisecond per call. That's the same lock the
    /// cpal callback takes to append — and that callback already performs
    /// a potentially-reallocating `extend_from_slice` under it — so this
    /// stays well inside the contention envelope the capture path already
    /// accepts, at one call per check interval rather than per audio chunk.
    pub fn trailing_rms(&self, window_secs: f32) -> Option<f32> {
        let buffer = self.buffer.lock().ok()?;
        trailing_rms_of(&buffer, window_secs)
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
    fn one_shot_resample_downsamples_48k_to_16k_within_tolerance() {
        let input = vec![0.0_f32; CANONICAL_SAMPLE_RATE as usize * 2]; // 2s @ 48kHz
        let output = one_shot_resample(&input, CANONICAL_SAMPLE_RATE, PIPELINE_SAMPLE_RATE).unwrap();
        let expected = PIPELINE_SAMPLE_RATE as usize * 2; // 2s @ 16kHz
        let tolerance = RESAMPLE_CHUNK_FRAMES; // one chunk's worth, from the flush padding
        assert!(
            (output.len() as i64 - expected as i64).abs() < tolerance as i64,
            "expected ~{expected} frames, got {}",
            output.len()
        );
    }

    #[test]
    fn one_shot_resample_is_passthrough_when_rates_match() {
        let input = vec![0.5_f32; 1000];
        let output = one_shot_resample(&input, 16_000, 16_000).unwrap();
        assert_eq!(output, input);
    }

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

    /// ISC-200: the computed RMS is the real, textbook value for a known
    /// signal — checked against the closed-form answer for a sine wave
    /// (amplitude / sqrt(2)), not merely "some number came back."
    #[test]
    fn trailing_rms_matches_the_closed_form_value_for_known_signals() {
        let window_secs = 1.0_f32;
        let n = CANONICAL_SAMPLE_RATE as usize;

        // Pure silence: exactly zero energy.
        let silence = vec![0.0_f32; n];
        let rms = trailing_rms_of(&silence, window_secs).expect("a full window of silence is measurable");
        assert!(rms.abs() < 1e-6, "silence must read as ~0 RMS, got {rms}");

        // A 440Hz sine at amplitude 0.5 — RMS is analytically 0.5/sqrt(2).
        let amplitude = 0.5_f32;
        let sine: Vec<f32> = (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / CANONICAL_SAMPLE_RATE as f32).sin())
            .collect();
        let expected = amplitude / 2.0_f32.sqrt();
        let rms = trailing_rms_of(&sine, window_secs).expect("a full window of tone is measurable");
        assert!(
            (rms - expected).abs() < 1e-3,
            "a 0.5-amplitude sine must read ~{expected} RMS (amplitude/sqrt(2)), got {rms}"
        );

        // A DC-offset constant: RMS equals the magnitude itself.
        let constant = vec![-0.25_f32; n];
        let rms = trailing_rms_of(&constant, window_secs).unwrap();
        assert!((rms - 0.25).abs() < 1e-5, "RMS of a constant is its magnitude, got {rms}");
    }

    /// ISC-200: it reads the TRAILING window, not the whole buffer. This is
    /// the property the entire silence detector rests on — a call that was
    /// loud for 50 minutes and has now been silent for the last minute must
    /// read as silent, or the prompt could never fire.
    #[test]
    fn trailing_rms_reads_only_the_tail_not_the_whole_buffer() {
        let one_second = CANONICAL_SAMPLE_RATE as usize;

        // 10 seconds of loud audio followed by 2 seconds of silence.
        let mut buffer = vec![0.8_f32; one_second * 10];
        buffer.extend(std::iter::repeat(0.0_f32).take(one_second * 2));

        let tail_rms = trailing_rms_of(&buffer, 2.0).expect("2s window fits in a 12s buffer");
        assert!(tail_rms.abs() < 1e-6, "the trailing 2s are silent, so RMS must be ~0, got {tail_rms}");

        // Widen the window past the silence and the earlier loud audio
        // reappears in the measurement — proving the window bound is real.
        let wide_rms = trailing_rms_of(&buffer, 12.0).unwrap();
        assert!(wide_rms > 0.5, "a 12s window covers the loud section, expected >0.5, got {wide_rms}");
    }

    /// ISC-200: not-enough-audio-yet is `None`, never a falsely-quiet
    /// reading from a partial window — otherwise a recording would look
    /// "silent" for its first minute and could be prompted to stop
    /// immediately.
    #[test]
    fn trailing_rms_is_none_until_a_full_window_is_available() {
        let one_second = CANONICAL_SAMPLE_RATE as usize;
        let almost = vec![0.0_f32; one_second * 60 - 1];
        assert_eq!(trailing_rms_of(&almost, 60.0), None, "one sample short of a full window is not yet measurable");

        let exact = vec![0.0_f32; one_second * 60];
        assert!(trailing_rms_of(&exact, 60.0).is_some(), "exactly a full window is measurable");

        assert_eq!(trailing_rms_of(&[], 60.0), None);
        assert_eq!(trailing_rms_of(&exact, 0.0), None, "a zero-length window is meaningless");
        assert_eq!(trailing_rms_of(&exact, -5.0), None);
        assert_eq!(trailing_rms_of(&exact, f32::NAN), None);
    }

    /// The real method on a real, live `RecordingSession` — proves
    /// `trailing_rms` genuinely reads the live capture buffer and leaves
    /// the stream running, not just that the free function's math works.
    #[test]
    fn trailing_rms_reads_a_live_session_without_disturbing_capture() {
        let tmp_data_dir = std::env::temp_dir().join(format!("kai-notetaker-test-rms-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_data_dir).unwrap();

        if list_input_devices().map(|d| d.is_empty()).unwrap_or(true) {
            eprintln!("skipping: no input device available in this environment");
            std::fs::remove_dir_all(&tmp_data_dir).ok();
            return;
        }

        let session = RecordingSession::start(&tmp_data_dir, "rms-live-test").unwrap();
        std::thread::sleep(Duration::from_millis(300));

        // A 60-second window can't be satisfied by a 300ms recording.
        assert_eq!(session.trailing_rms(60.0), None, "a fresh session has nowhere near 60s of audio");

        // A window short enough to actually be filled returns a real,
        // finite, non-negative reading.
        let len_before = session.buffer.lock().unwrap().len();
        if len_before > (0.05 * CANONICAL_SAMPLE_RATE as f32) as usize {
            let rms = session.trailing_rms(0.05).expect("50ms of audio should be measurable by now");
            assert!(rms.is_finite() && rms >= 0.0, "RMS must be a real non-negative value, got {rms}");
        }

        // The stream must still be capturing after being measured — the
        // "without disturbing the live capture stream" half of ISC-200.
        std::thread::sleep(Duration::from_millis(200));
        let len_after = session.buffer.lock().unwrap().len();
        assert!(len_after > len_before, "capture must keep running after an RMS read: before={len_before}, after={len_after}");

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
