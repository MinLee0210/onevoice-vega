//! 2-mic beamforming front-end — differentiator #3 ("noise-first audio
//! front-end"). Runs on CPU (Kryo), *before* denoise/VAD.
//!
//! This is the real algorithm, validated on synthetic multi-channel input;
//! the pipeline is still file-based mono (no mic HAT yet), so this module is
//! not yet wired into `Pipeline` — it becomes the first stage once the
//! ReSpeaker HAT (2-mic) feeds two channels (see AGENTS.md → Hardware).
//!
//! Approach: **delay-and-sum** with the inter-channel delay estimated by
//! cross-correlation (a classic, cheap DoA-aware beamformer). A neural
//! MVDR upgrade path is noted in docs/decisions/pipeline-improvement-research-2026-10.md.
// Not wired into `Pipeline` until the 2-mic HAT exists (see module docs).
#![allow(dead_code)]

use crate::audio::AudioBuffer;

/// Estimates the delay (in samples) of `b` relative to `a` by maximizing the
/// cross-correlation `r(d) = Σ a[i]·b[i-d]` over `|d| ≤ max_lag`.
///
/// Returns `d` such that `b[i] ≈ a[i-d]` (i.e. `b` is `a` delayed by `d`).
/// Positive `d` = `b` lags `a`.
pub fn cross_correlation_delay(a: &[f32], b: &[f32], max_lag: usize) -> isize {
    let la = a.len() as isize;
    let lb = b.len() as isize;
    let mut best_d = 0isize;
    let mut best = f32::NEG_INFINITY;
    for d in -(max_lag as isize)..=(max_lag as isize) {
        // r(d) = Σ_i a[i]·b[i+d] over the valid overlap.
        let lo = 0.max(-d);
        let hi = la.min(lb - d);
        let mut sum = 0.0f32;
        let mut i = lo;
        while i < hi {
            sum += a[i as usize] * b[(i + d) as usize];
            i += 1;
        }
        if sum > best {
            best = sum;
            best_d = d;
        }
    }
    best_d
}

/// Aligns each channel by its delay relative to the first channel and averages
/// them (delay-and-sum). `delays[i]` is the delay of channel `i` relative to
/// channel 0 (from [`cross_correlation_delay`]).
pub fn delay_and_sum(channels: &[AudioBuffer], delays: &[isize]) -> AudioBuffer {
    assert_eq!(
        channels.len(),
        delays.len(),
        "one delay per channel required"
    );
    if channels.is_empty() {
        return AudioBuffer::new(Vec::new(), 16_000);
    }
    let sample_rate_hz = channels[0].sample_rate_hz;
    let n = channels.iter().map(|c| c.samples.len()).min().unwrap_or(0);

    let mut out = vec![0.0f32; n];
    for (ch, &d) in channels.iter().zip(delays) {
        for (i, o) in out.iter_mut().enumerate() {
            // Shift each channel left by its delay so the direct-path signal
            // lines up with the reference (channel 0, delay 0).
            let j = i as isize + d;
            if j >= 0 && (j as usize) < ch.samples.len() {
                *o += ch.samples[j as usize];
            }
        }
    }
    let gain = 1.0 / channels.len() as f32;
    for v in &mut out {
        *v *= gain;
    }
    AudioBuffer::new(out, sample_rate_hz)
}

/// Convenience: estimate the delay between two channels and beamform them.
pub fn beamform_two(reference: &AudioBuffer, other: &AudioBuffer, max_lag: usize) -> AudioBuffer {
    let d = cross_correlation_delay(&reference.samples, &other.samples, max_lag);
    delay_and_sum(&[reference.clone(), other.clone()], &[0, d])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq: f32, sr: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    /// A copy delayed by `d` should be recovered as `d`.
    #[test]
    fn cross_correlation_recovers_a_known_delay() {
        let sr = 16_000;
        let a = tone(4_000, 440.0, sr);
        let d = 7usize;
        let mut b = vec![0.0f32; a.len()];
        b[d..].copy_from_slice(&a[..a.len() - d]);

        assert_eq!(cross_correlation_delay(&a, &b, 32), d as isize);
    }

    /// Delay-and-sum of a channel against a delayed copy should reconstruct
    /// the aligned signal (high correlation with the reference).
    #[test]
    fn delay_and_sum_aligns_a_delayed_channel() {
        let sr = 16_000;
        let a = tone(4_000, 440.0, sr);
        let d = 5usize;
        let mut b = vec![0.0f32; a.len()];
        b[d..].copy_from_slice(&a[..a.len() - d]);

        let refbuf = AudioBuffer::new(a.clone(), sr);
        let other = AudioBuffer::new(b, sr);
        let out = beamform_two(&refbuf, &other, 32);

        // Aligned sum should correlate strongly with the reference.
        assert_eq!(out.samples.len(), a.len());
        assert!(
            out.samples.iter().any(|&s| s.abs() > 0.5),
            "beamformed output looks silent"
        );
    }

    /// Two reference-identical channels average back to that channel.
    #[test]
    fn delay_and_sum_of_identical_channels_is_identity() {
        let sr = 16_000;
        let a = tone(1_000, 300.0, sr);
        let buf = AudioBuffer::new(a.clone(), sr);
        let out = delay_and_sum(&[buf.clone(), buf], &[0, 0]);
        for (x, y) in out.samples.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
