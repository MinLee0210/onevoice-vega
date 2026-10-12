//! Denoise stage: RNNoise via the `nnnoiseless` crate (pure Rust,
//! BSD-3-Clause, weights compiled in — no ONNX file to load, unlike
//! every other ML-backed stage). Runs on CPU during capture (zero added
//! latency — see AGENTS.md -> Pipeline).
//!
//! RNNoise's feature/band structure is hardcoded for a fixed 48kHz
//! input processed in fixed 480-sample (10ms) frames — `nnnoiseless`
//! bakes both in, they aren't configurable. Everything downstream (VAD,
//! ASR) wants 16kHz, so this stage also resamples 48kHz -> 16kHz once
//! on the way out, via a small windowed-sinc low-pass filter + decimate-
//! by-3 (rather than pulling in a general-purpose resampling crate for
//! one fixed, exact 3:1 ratio).

use nnnoiseless::DenoiseState;

use crate::audio::AudioBuffer;

/// RNNoise's fixed internal rate (see module docs) — the mic capture
/// stage must feed this, not the 16kHz the rest of the pipeline uses.
pub const INPUT_SAMPLE_RATE_HZ: u32 = 48_000;
const OUTPUT_SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
}

impl Denoiser {
    /// No model file to load — `nnnoiseless` bundles RNNoise's trained
    /// weights in the crate itself (see module docs).
    pub fn load() -> Self {
        Self {
            state: DenoiseState::new(),
        }
    }

    /// Denoises `audio` (must be 48kHz mono — [`INPUT_SAMPLE_RATE_HZ`])
    /// and resamples the result to 16kHz for the rest of the pipeline.
    pub fn process(&mut self, audio: AudioBuffer) -> AudioBuffer {
        assert_eq!(
            audio.sample_rate_hz, INPUT_SAMPLE_RATE_HZ,
            "Denoiser expects {INPUT_SAMPLE_RATE_HZ}Hz input (RNNoise's fixed internal rate)"
        );

        // Each PTT cycle is an independent utterance, so reset RNNoise's
        // recurrent state — otherwise the previous utterance's noise
        // profile bleeds into this one. The state stays a field only to
        // reuse the allocation across cycles.
        self.state = DenoiseState::new();

        let mut denoised = Vec::with_capacity(audio.samples.len());
        let mut out_frame = [0.0f32; FRAME_SIZE];
        let mut first_frame = true;

        for chunk in audio.samples.chunks(FRAME_SIZE) {
            let mut in_frame = [0.0f32; FRAME_SIZE];
            in_frame[..chunk.len()].copy_from_slice(chunk);
            self.state.process_frame(&mut out_frame, &in_frame);

            // nnnoiseless has one frame of algorithmic delay; its own
            // documented usage pattern discards the first output frame.
            if first_frame {
                first_frame = false;
            } else {
                denoised.extend_from_slice(&out_frame[..chunk.len()]);
            }
        }

        AudioBuffer::new(resample_48k_to_16k(&denoised), OUTPUT_SAMPLE_RATE_HZ)
    }
}

/// Windowed-sinc low-pass (Hamming window, cutoff below the 16kHz
/// target's 8kHz Nyquist) followed by decimate-by-3. Exact for the
/// 48000/16000 = 3 ratio this stage always uses.
fn resample_48k_to_16k(samples: &[f32]) -> Vec<f32> {
    const DECIMATION_FACTOR: usize = 3;
    const NUM_TAPS: usize = 63; // odd, for a symmetric linear-phase filter
    const CUTOFF_HZ: f32 = 7_500.0;

    let taps = lowpass_taps(NUM_TAPS, CUTOFF_HZ, INPUT_SAMPLE_RATE_HZ as f32);
    let half = taps.len() / 2;

    (0..samples.len())
        .step_by(DECIMATION_FACTOR)
        .map(|center| {
            taps.iter()
                .enumerate()
                .filter_map(|(k, &tap)| {
                    let idx = center as isize + k as isize - half as isize;
                    (idx >= 0 && (idx as usize) < samples.len())
                        .then(|| tap * samples[idx as usize])
                })
                .sum()
        })
        .collect()
}

fn lowpass_taps(num_taps: usize, cutoff_hz: f32, sample_rate_hz: f32) -> Vec<f32> {
    use std::f32::consts::PI;

    let m = (num_taps - 1) as f32;
    let fc = cutoff_hz / sample_rate_hz;

    let mut taps: Vec<f32> = (0..num_taps)
        .map(|n| {
            let x = n as f32 - m / 2.0;
            let sinc = if x == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            let window = 0.54 - 0.46 * (2.0 * PI * n as f32 / m).cos(); // Hamming
            sinc * window
        })
        .collect();

    let dc_gain: f32 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= dc_gain;
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_resamples_to_16khz_at_roughly_a_third_the_length() {
        let mut denoiser = Denoiser::load();
        let one_second = AudioBuffer::new(
            vec![0.0; INPUT_SAMPLE_RATE_HZ as usize],
            INPUT_SAMPLE_RATE_HZ,
        );

        let out = denoiser.process(one_second);

        assert_eq!(out.sample_rate_hz, OUTPUT_SAMPLE_RATE_HZ);
        // One RNNoise frame (10ms) is dropped for algorithmic delay, so
        // expect just under 1/3 of the input length, not exactly 1/3.
        let expected = (INPUT_SAMPLE_RATE_HZ as usize - FRAME_SIZE) / 3;
        assert!(
            out.samples.len().abs_diff(expected) < FRAME_SIZE,
            "got {} samples, expected around {expected}",
            out.samples.len()
        );
    }

    #[test]
    fn lowpass_filter_attenuates_frequencies_above_the_new_nyquist() {
        let sample_rate = INPUT_SAMPLE_RATE_HZ as f32;
        let duration_samples = INPUT_SAMPLE_RATE_HZ as usize;

        let sine_at = |freq_hz: f32| -> Vec<f32> {
            (0..duration_samples)
                .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
                .collect()
        };

        // Below the resampled 8kHz Nyquist: should survive with similar energy.
        let low_tone = resample_48k_to_16k(&sine_at(1_000.0));
        // Above it: should be strongly attenuated by the low-pass filter.
        let high_tone = resample_48k_to_16k(&sine_at(20_000.0));

        let rms = |samples: &[f32]| -> f32 {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
        };

        assert!(
            rms(&low_tone) > 0.5,
            "1kHz tone should pass through mostly intact"
        );
        assert!(
            rms(&high_tone) < 0.1,
            "20kHz tone should be attenuated by the low-pass filter"
        );
    }
}
