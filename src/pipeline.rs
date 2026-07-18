//! Composes the stages in the fixed order from CLAUDE.md:
//! capture -> denoise -> VAD trim -> ASR -> MT -> TTS -> playback.
//! This is the one place that knows the pipeline order; stage modules
//! don't call each other.

use std::path::Path;
use std::time::Instant;

use crate::config::{Config, LatencyBudgetConfig};
use crate::error::OneVoiceError;
use crate::stages::asr::AsrEngine;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise::{self, Denoiser};
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtEngine;
use crate::stages::tts::TtsEngine;
use crate::stages::vad::Vad;

pub struct PipelineConfig<'a> {
    pub config: &'a Config,
    pub models_dir: &'a Path,
    pub data_dir: &'a Path,
    pub source_lang: &'a str,
    pub target_lang: &'a str,
    pub input_wav_path: &'a Path,
    pub output_wav_path: &'a Path,
}

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
    latency_budget: LatencyBudgetConfig,
}

impl Pipeline {
    pub fn new(cfg: PipelineConfig) -> Result<Self, OneVoiceError> {
        let pair = cfg
            .config
            .language_pairs
            .find(cfg.source_lang, cfg.target_lang)
            .ok_or_else(|| OneVoiceError::LanguagePairNotFound {
                source_lang: cfg.source_lang.to_string(),
                target_lang: cfg.target_lang.to_string(),
            })?;

        Ok(Self {
            source_lang: cfg.source_lang.to_string(),
            target_lang: cfg.target_lang.to_string(),
            // The mic must capture at RNNoise's fixed 48kHz; Denoiser
            // resamples down to 16kHz for VAD/ASR on its way out (see
            // src/stages/denoise.rs module docs).
            audio_io: AudioCapture::new(
                denoise::INPUT_SAMPLE_RATE_HZ,
                cfg.input_wav_path,
                cfg.output_wav_path,
            ),
            denoiser: Denoiser::load(),
            vad: Vad::load(cfg.models_dir.join("vad/model.onnx"))?,
            asr: AsrEngine::load(cfg.models_dir.join(&pair.asr_model_dir))?,
            mt: MtEngine::load(cfg.models_dir.join(&pair.mt_model_dir))?,
            glossary: Glossary::load(&cfg.data_dir.join("glossary"))?,
            // espeak-ng-data is shared across all Piper voices (vi/en/zh);
            // ignored by the MMS (ko) backend -- see src/stages/tts.rs.
            tts: TtsEngine::load(
                cfg.models_dir.join(&pair.tts_model_dir),
                &cfg.models_dir.join("tts"),
            )?,
            latency_budget: cfg.config.latency_budget,
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
            "latency through MT (CPU dev machine, NOT the Hexagon NPU target -- see CLAUDE.md -> Latency target; TTS timed separately below)"
        );

        let t = Instant::now();
        let (audio_out, tts_first_chunk_ms) = self.tts.synthesize(&translation)?;
        let tts_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            tts_first_chunk_ms = format!("{tts_first_chunk_ms:.1}"),
            tts_total_ms = format!("{tts_ms:.1}"),
            through_tts_ms = format!("{:.1}", cycle_start.elapsed().as_secs_f64() * 1000.0),
            "TTS chunked (through_tts_ms is the full button-release -> synthesized-audio latency CLAUDE.md targets at <=1.2s)"
        );

        let t = Instant::now();
        self.audio_io.play(&audio_out)?;
        let play_ms = t.elapsed().as_secs_f64() * 1000.0;

        let io_overhead_ms = capture_ms + play_ms;
        let total_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;

        // Latency budget checks
        let budget = self.latency_budget;
        if vad_ms > budget.stage.vad_trim_ms as f64 {
            tracing::warn!(
                "VAD latency {:.1}ms exceeded stage budget of {}ms",
                vad_ms,
                budget.stage.vad_trim_ms
            );
        }
        if asr_ms > budget.stage.asr_ms as f64 {
            tracing::warn!(
                "ASR latency {:.1}ms exceeded stage budget of {}ms",
                asr_ms,
                budget.stage.asr_ms
            );
        }
        if mt_ms > budget.stage.mt_ms as f64 {
            tracing::warn!(
                "MT latency {:.1}ms exceeded stage budget of {}ms",
                mt_ms,
                budget.stage.mt_ms
            );
        }
        if tts_first_chunk_ms > budget.stage.tts_first_chunk_ms as f64 {
            tracing::warn!(
                "TTS first chunk latency {:.1}ms exceeded stage budget of {}ms",
                tts_first_chunk_ms,
                budget.stage.tts_first_chunk_ms
            );
        }
        if io_overhead_ms > budget.stage.io_overhead_ms as f64 {
            tracing::warn!(
                "I/O overhead latency {:.1}ms exceeded stage budget of {}ms",
                io_overhead_ms,
                budget.stage.io_overhead_ms
            );
        }

        if total_ms > budget.target_total_ms as f64 {
            tracing::warn!(
                "Total latency {:.1}ms exceeded target total budget of {}ms",
                total_ms,
                budget.target_total_ms
            );
        } else if total_ms > budget.stretch_total_ms as f64 {
            tracing::info!(
                "Total latency {:.1}ms met target total budget of {}ms but exceeded stretch budget of {}ms",
                total_ms,
                budget.target_total_ms,
                budget.stretch_total_ms
            );
        } else {
            tracing::info!(
                "Total latency {:.1}ms met stretch budget of {}ms",
                total_ms,
                budget.stretch_total_ms
            );
        }

        Ok(())
    }
}
