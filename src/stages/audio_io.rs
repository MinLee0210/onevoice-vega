//! Mic capture, PTT gating, and playback. CPU-side (Kryo). Backed by
//! cpal/hound per CLAUDE.md -> Tooling; not wired to real hardware yet.

use crate::audio::AudioBuffer;
use crate::error::OneVoiceError;

pub struct AudioCapture {
    sample_rate_hz: u32,
}

impl AudioCapture {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self { sample_rate_hz }
    }

    /// Blocks while PTT is held, returns the captured utterance on release.
    pub fn capture_on_ptt(&self) -> Result<AudioBuffer, OneVoiceError> {
        tracing::debug!(
            sample_rate_hz = self.sample_rate_hz,
            "would capture mic audio gated on PTT here"
        );
        todo!("wire up mic capture (cpal) gated on PTT button state")
    }

    /// Starts playback; caller streams TTS output in as it's synthesized
    /// to hide latency (see CLAUDE.md -> Pipeline).
    pub fn play(&self, _audio: &AudioBuffer) -> Result<(), OneVoiceError> {
        todo!("wire up playback (cpal) with streaming start for TTS")
    }
}
