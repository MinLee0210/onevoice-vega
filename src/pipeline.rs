//! Composes the stages in the fixed order from CLAUDE.md:
//! capture -> denoise -> VAD trim -> ASR -> MT -> TTS -> playback.
//! This is the one place that knows the pipeline order; stage modules
//! don't call each other.

use std::path::Path;
use std::time::Instant;

use crate::config::Config;
use crate::error::OneVoiceError;
use crate::stages::asr::AsrEngine;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise::{self, Denoiser};
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtEngine;
use crate::stages::tts::TtsEngine;
use crate::stages::vad::Vad;

pub struct Pipeline {
    source_lang: String,
    target_lang: String,
    audio_io: AudioCapture,
    denoiser: Denoiser,
    vad: Vad,
    asr: AsrEngine,
    mt: MtEngine,
    glossary: Glossary,
    tts: TtsEngine,
}

impl Pipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &Config,
        models_dir: &Path,
        data_dir: &Path,
        source_lang: &str,
        target_lang: &str,
        input_wav_path: &Path,
        output_wav_path: &Path,
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
            // The mic must capture at RNNoise's fixed 48kHz; Denoiser
            // resamples down to 16kHz for VAD/ASR on its way out (see
            // src/stages/denoise.rs module docs).
            audio_io: AudioCapture::new(
                denoise::INPUT_SAMPLE_RATE_HZ,
                input_wav_path,
                output_wav_path,
            ),
            denoiser: Denoiser::load(),
            vad: Vad::load(models_dir.join("vad/model.onnx"))?,
            asr: AsrEngine::load(models_dir.join(&pair.asr_model_dir))?,
            mt: MtEngine::load(models_dir.join(&pair.mt_model_dir))?,
            glossary: Glossary::load(&data_dir.join("glossary"))?,
            tts: TtsEngine::load(models_dir.join(&pair.tts_model_dir))?,
        })
    }

    /// Runs one PTT press-release cycle end to end. Times each stage —
    /// this is the "button-release -> first audio" latency CLAUDE.md
    /// targets at ≤1.2s, *excluding* one-time model loading in `new()`.
    /// Measured on whatever CPU this binary runs on, not the Hexagon
    /// NPU target hardware — see the logged warning every run emits.
    pub fn run_once(&mut self) -> Result<(), OneVoiceError> {
        let cycle_start = Instant::now();

        let t = Instant::now();
        let raw = self.audio_io.capture_on_ptt()?;
        let capture_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let denoised = self.denoiser.process(raw);
        let denoise_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let trimmed = self.vad.trim(denoised)?;
        let vad_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let utterance = self.asr.transcribe(&trimmed, &self.source_lang)?;
        let asr_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(text = %utterance.text, lang = %utterance.lang, "ASR");

        let t = Instant::now();
        let translation = self
            .mt
            .translate(&utterance, &self.target_lang, &self.glossary)?;
        let mt_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(text = %translation.text, lang = %translation.lang, "MT");

        tracing::info!(
            capture_ms = format!("{capture_ms:.1}"),
            denoise_ms = format!("{denoise_ms:.1}"),
            vad_ms = format!("{vad_ms:.1}"),
            asr_ms = format!("{asr_ms:.1}"),
            mt_ms = format!("{mt_ms:.1}"),
            through_mt_ms = format!("{:.1}", cycle_start.elapsed().as_secs_f64() * 1000.0),
            "latency (CPU dev machine, NOT the Hexagon NPU target -- see CLAUDE.md -> Latency target; TTS not included, not yet implemented)"
        );

        let t = Instant::now();
        let audio_out = self.tts.synthesize(&translation)?;
        let tts_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(tts_ms = format!("{tts_ms:.1}"), "TTS");

        self.audio_io.play(&audio_out)
    }
}
