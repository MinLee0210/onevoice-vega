//! Composes the stages in the fixed order from CLAUDE.md:
//! capture -> denoise -> VAD trim -> ASR -> MT -> TTS -> playback.
//! This is the one place that knows the pipeline order; stage modules
//! don't call each other.

use std::path::Path;

use crate::config::Config;
use crate::error::OneVoiceError;
use crate::stages::asr::AsrEngine;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise::Denoiser;
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtEngine;
use crate::stages::tts::TtsEngine;
use crate::stages::vad::Vad;

const SAMPLE_RATE_HZ: u32 = 16_000;

pub struct Pipeline {
    source_lang: String,
    target_lang: String,
    audio_io: AudioCapture,
    denoiser: Denoiser,
    vad: Vad,
    asr: AsrEngine,
    mt: MtEngine,
    #[allow(dead_code)] // wired into `mt`'s constrained decoding once that lands
    glossary: Glossary,
    tts: TtsEngine,
}

impl Pipeline {
    pub fn new(
        config: &Config,
        models_dir: &Path,
        data_dir: &Path,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<Self, OneVoiceError> {
        let pair = config
            .language_pairs
            .find(source_lang, target_lang)
            .ok_or_else(|| OneVoiceError::LanguagePairNotFound {
                source_lang: source_lang.to_string(),
                target_lang: target_lang.to_string(),
            })?;

        Ok(Self {
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            audio_io: AudioCapture::new(SAMPLE_RATE_HZ),
            denoiser: Denoiser::load(models_dir.join("denoise/denoise.onnx"))?,
            vad: Vad::load(models_dir.join("vad/vad.onnx"))?,
            asr: AsrEngine::load(models_dir.join(&pair.asr_model))?,
            mt: MtEngine::load(models_dir.join(&pair.mt_model))?,
            glossary: Glossary::load(&data_dir.join("glossary"))?,
            tts: TtsEngine::load(models_dir.join(&pair.tts_model))?,
        })
    }

    /// Runs one PTT press-release cycle end to end.
    pub fn run_once(&self) -> Result<(), OneVoiceError> {
        let raw = self.audio_io.capture_on_ptt()?;
        let denoised = self.denoiser.process(raw);
        let trimmed = self.vad.trim(denoised);
        let utterance = self.asr.transcribe(&trimmed, &self.source_lang)?;
        let translation = self.mt.translate(&utterance, &self.target_lang)?;
        let audio_out = self.tts.synthesize(&translation)?;
        self.audio_io.play(&audio_out)
    }
}
