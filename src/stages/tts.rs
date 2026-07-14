//! Text-to-speech: Piper (VITS/ONNX). Runs on CPU, real-time — and
//! streams output so playback can start before synthesis completes (see
//! CLAUDE.md -> Pipeline).

use std::path::PathBuf;

use crate::audio::{AudioBuffer, Translation};
use crate::error::OneVoiceError;

pub struct TtsEngine {
    model_path: PathBuf,
}

impl TtsEngine {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_path: model_path.into(),
        })
    }

    pub fn synthesize(&self, translation: &Translation) -> Result<AudioBuffer, OneVoiceError> {
        tracing::debug!(
            model = %self.model_path.display(),
            lang = %translation.lang,
            "would synthesize speech here"
        );
        todo!("Piper VITS/ONNX synthesis with streaming playback start")
    }
}
