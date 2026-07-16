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

    pub fn synthesize(&mut self, translation: &Translation) -> Result<AudioBuffer, OneVoiceError> {
        tracing::debug!(
            model = %self.model_path.display(),
            lang = %translation.lang,
            "would synthesize speech here"
        );
        // Deliberately not implemented yet, not just unfinished: Piper's
        // phonemizer (piper-phonemize/espeak-ng, required for Vietnamese)
        // is GPL-3.0, and whether embedding it in this binary GPL-taints
        // the shipped product is an open legal question, not an
        // engineering one -- see docs/decisions/piper-vi-voice-license-trap.md.
        // The pipeline runs everything up through MT and reports this
        // clearly rather than continuing past an unresolved license
        // question.
        Err(OneVoiceError::Tts(
            "not implemented: Piper TTS needs a GPL-3.0 phonemizer for Vietnamese, blocked on legal review \
             before it can be added to this binary -- see docs/decisions/piper-vi-voice-license-trap.md"
                .to_string(),
        ))
    }
}
