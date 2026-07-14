//! Voice activity detection: Silero VAD via the `ort` crate. Runs on CPU,
//! trims silence around the utterance before it reaches ASR.

use std::path::PathBuf;

use crate::audio::AudioBuffer;
use crate::error::OneVoiceError;

pub struct Vad {
    model_path: PathBuf,
}

impl Vad {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_path: model_path.into(),
        })
    }

    pub fn trim(&self, audio: AudioBuffer) -> AudioBuffer {
        tracing::debug!(model = %self.model_path.display(), "would trim silence here");
        let _ = audio;
        todo!("Silero VAD trim via ort")
    }
}
