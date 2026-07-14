//! ASR encoder/decoder inference via `ort` + the QNN execution provider.
//! Runs on the Hexagon NPU (see CLAUDE.md -> Compute split).

use std::path::PathBuf;

use crate::audio::{AudioBuffer, Utterance};
use crate::error::OneVoiceError;

pub struct AsrEngine {
    model_path: PathBuf,
}

impl AsrEngine {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_path: model_path.into(),
        })
    }

    pub fn transcribe(&self, audio: &AudioBuffer, lang: &str) -> Result<Utterance, OneVoiceError> {
        tracing::debug!(
            model = %self.model_path.display(),
            lang,
            duration_ms = audio.duration_ms(),
            "would run ASR here"
        );
        todo!("run ASR encoder/decoder via ort + QNN execution provider")
    }
}
