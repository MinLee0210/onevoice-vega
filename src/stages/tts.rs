//! Text-to-speech. The device is bidirectional, so it synthesizes every
//! target language: VI, EN, ZH (Mandarin) via Piper (VITS/ONNX, espeak-ng
//! front-end) and KO via Meta MMS-TTS (VITS/ONNX, uroman romanization
//! front-end). All four are validated end-to-end in
//! experiments/04_tts.ipynb (synthesis + round-trip TTS→ASR
//! intelligibility) and packaged into `models/tts/<lang>/`.
//!
//! **Rust port still pending — this stage is a stub.** The models and the
//! per-language text front-ends are settled (see the notebook); what's
//! left is the Rust side of VITS inference + phonemization:
//!   - Piper (vi/en/zh): VITS forward pass via `ort` (phoneme_id tensor →
//!     waveform) is straightforward; the real work is the espeak-ng
//!     phonemizer (text → IPA → phoneme ids via the voice's
//!     `phoneme_id_map`). Needs an espeak-ng binding/FFI.
//!   - MMS (ko): VITS forward via `ort` too, but the front-end is uroman
//!     (Hangul → Latin) + the char tokenizer, not espeak — the one
//!     language that doesn't share the Piper path. A Korean-specific
//!     romanizer in Rust is the porting cost here.
//! Tracked in docs/TODO.md. (The espeak-ng GPL-3.0 question is noted in
//! docs/decisions/piper-vi-voice-license-trap.md but is NOT a blocker for
//! the hackathon prototype — it only bites at commercial distribution.)
//!
//! Runs on CPU; QNN/Hexagon is future work like the other ONNX stages.

use std::path::PathBuf;

use crate::audio::{AudioBuffer, Translation};
use crate::error::OneVoiceError;

pub struct TtsEngine {
    /// `models/tts/<target-lang>/` — Piper `.onnx` + `.onnx.json`, or the
    /// MMS `.onnx` + tokenizer for Korean (see config's `tts_model_dir`).
    model_dir: PathBuf,
}

impl TtsEngine {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        Ok(Self {
            model_dir: model_dir.into(),
        })
    }

    pub fn synthesize(&mut self, translation: &Translation) -> Result<AudioBuffer, OneVoiceError> {
        tracing::debug!(
            model_dir = %self.model_dir.display(),
            lang = %translation.lang,
            "would synthesize speech here"
        );
        // Stub: the voice models for all four target languages are
        // validated and packaged (experiments/04_tts.ipynb ->
        // models/tts/<lang>/), but the Rust VITS-inference + phonemizer
        // port hasn't been written yet -- see this module's docs and
        // docs/TODO.md. The pipeline runs everything through MT and
        // reports this clearly rather than emitting silence.
        Err(OneVoiceError::Tts(format!(
            "not implemented: TTS voice for '{}' is validated and packaged (see \
             experiments/04_tts.ipynb and models/tts/{}/), but the Rust synthesis \
             port is not written yet -- see src/stages/tts.rs docs and docs/TODO.md",
            translation.lang, translation.lang
        )))
    }
}
