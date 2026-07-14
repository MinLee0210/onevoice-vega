//! MT encoder/decoder inference via `ort` + the QNN execution provider.
//! Runs on the Hexagon NPU, greedy/beam=1 decode to hit the latency
//! target (see CLAUDE.md -> Latency target). Glossary enforcement during
//! decode will hook in here once constrained decoding is implemented —
//! see `glossary`.

use std::path::PathBuf;

use crate::audio::{Translation, Utterance};
use crate::error::OneVoiceError;

pub struct MtEngine {
    model_path: PathBuf,
}

impl MtEngine {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_path: model_path.into(),
        })
    }

    pub fn translate(
        &self,
        utterance: &Utterance,
        target_lang: &str,
    ) -> Result<Translation, OneVoiceError> {
        tracing::debug!(
            model = %self.model_path.display(),
            source_lang = %utterance.lang,
            target_lang,
            "would run MT here"
        );
        todo!("run MT encoder/decoder via ort + QNN execution provider, greedy decode")
    }
}
