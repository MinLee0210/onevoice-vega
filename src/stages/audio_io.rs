//! Mic capture, PTT gating, and playback. CPU-side (Kryo).
//!
//! Current implementation reads/writes WAV files rather than live
//! audio -- there's no PTT hardware or mic array to capture from yet
//! (Rubik Pi 3 not in hand). This still exercises the real pipeline
//! end-to-end; swapping this for cpal-based live capture/playback,
//! gated on an actual PTT button, is the only change needed once real
//! hardware shows up (see AGENTS.md -> Tooling).

use std::path::PathBuf;

use crate::audio::AudioBuffer;
use crate::error::OneVoiceError;

pub struct AudioCapture {
    sample_rate_hz: u32,
    input_wav_path: PathBuf,
    output_wav_path: PathBuf,
}

impl AudioCapture {
    pub fn new(
        sample_rate_hz: u32,
        input_wav_path: impl Into<PathBuf>,
        output_wav_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            sample_rate_hz,
            input_wav_path: input_wav_path.into(),
            output_wav_path: output_wav_path.into(),
        }
    }

    /// Reads `input_wav_path` as the "captured" utterance -- stands in
    /// for blocking on PTT + live mic capture until real hardware
    /// exists. Requires the file to already be at `sample_rate_hz`
    /// (RNNoise's fixed input rate, see `src/stages/denoise.rs`); real
    /// mic capture will configure the ADC for this rate directly rather
    /// than resampling.
    pub fn capture_on_ptt(&mut self) -> Result<AudioBuffer, OneVoiceError> {
        let mut reader = hound::WavReader::open(&self.input_wav_path).map_err(|source| {
            OneVoiceError::AudioIo(format!(
                "failed to open {}: {source}",
                self.input_wav_path.display()
            ))
        })?;
        let spec = reader.spec();

        if spec.sample_rate != self.sample_rate_hz {
            return Err(OneVoiceError::AudioIo(format!(
                "{} is {}Hz, expected {}Hz (RNNoise's fixed input rate -- see src/stages/denoise.rs)",
                self.input_wav_path.display(),
                spec.sample_rate,
                self.sample_rate_hz
            )));
        }
        if spec.channels != 1 {
            return Err(OneVoiceError::AudioIo(format!(
                "{} has {} channels, expected mono",
                self.input_wav_path.display(),
                spec.channels
            )));
        }

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<_, _>>(),
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
        }
        .map_err(|source| OneVoiceError::AudioIo(format!("failed to read samples: {source}")))?;

        tracing::debug!(
            path = %self.input_wav_path.display(),
            samples = samples.len(),
            "captured audio from wav file"
        );
        Ok(AudioBuffer::new(samples, spec.sample_rate))
    }

    /// Writes the synthesized audio to `output_wav_path` -- stands in
    /// for live speaker playback until real hardware exists.
    pub fn play(&mut self, audio: &AudioBuffer) -> Result<(), OneVoiceError> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: audio.sample_rate_hz,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(&self.output_wav_path, spec).map_err(|source| {
                OneVoiceError::AudioIo(format!(
                    "failed to create {}: {source}",
                    self.output_wav_path.display()
                ))
            })?;
        for &sample in &audio.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            writer
                .write_sample((clamped * i16::MAX as f32) as i16)
                .map_err(|source| {
                    OneVoiceError::AudioIo(format!("failed to write sample: {source}"))
                })?;
        }
        writer.finalize().map_err(|source| {
            OneVoiceError::AudioIo(format!(
                "failed to finalize {}: {source}",
                self.output_wav_path.display()
            ))
        })?;

        tracing::debug!(
            path = %self.output_wav_path.display(),
            samples = audio.samples.len(),
            "wrote synthesized audio to wav file"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_wav(path: &std::path::Path, sample_rate: u32, samples: &[f32]) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn capture_then_play_round_trips_through_a_wav_file() {
        let dir =
            std::env::temp_dir().join(format!("onevoice-audio-io-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input_path = dir.join("input.wav");
        let output_path = dir.join("output.wav");

        let original: Vec<f32> = (0..480).map(|i| (i as f32 / 480.0) - 0.5).collect();
        write_test_wav(&input_path, 48_000, &original);

        let mut audio_io = AudioCapture::new(48_000, &input_path, &output_path);
        let captured = audio_io.capture_on_ptt().expect("capture failed");
        assert_eq!(captured.sample_rate_hz, 48_000);
        assert_eq!(captured.samples.len(), original.len());

        audio_io.play(&captured).expect("play failed");
        assert!(output_path.exists());

        let mut reader = hound::WavReader::open(&output_path).unwrap();
        let written: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(written.len(), original.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn capture_rejects_wrong_sample_rate() {
        let dir = std::env::temp_dir().join(format!(
            "onevoice-audio-io-test-rate-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input_path = dir.join("input.wav");
        write_test_wav(&input_path, 16_000, &[0.0; 100]);

        let mut audio_io = AudioCapture::new(48_000, &input_path, dir.join("output.wav"));
        let err = audio_io.capture_on_ptt().unwrap_err();
        assert!(matches!(err, OneVoiceError::AudioIo(_)));

        std::fs::remove_dir_all(&dir).ok();
    }
}
