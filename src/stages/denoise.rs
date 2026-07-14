//! Denoise stage: RNNoise baseline, DeepFilterNet3 upgrade path. Runs on
//! CPU, during capture (zero added latency — see CLAUDE.md -> Pipeline).

use std::path::PathBuf;

use crate::audio::AudioBuffer;
use crate::error::OneVoiceError;

pub struct Denoiser {
    model_path: PathBuf,
}

impl Denoiser {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_path: model_path.into(),
        })
    }

    pub fn process(&self, audio: AudioBuffer) -> AudioBuffer {
        tracing::debug!(model = %self.model_path.display(), "would denoise audio here");
        let _ = audio;
        todo!("RNNoise/DeepFilterNet3 denoise pass")
    }
}
