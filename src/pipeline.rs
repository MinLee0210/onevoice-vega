//! Composes the stages in the fixed order from AGENTS.md:
//! capture -> denoise -> VAD trim -> ASR -> MT -> TTS -> playback.
//! This is the one place that knows the pipeline order; stage modules
//! don't call each other.

use std::path::Path;
use std::time::Instant;

use crate::config::{Config, LatencyBudgetConfig};
use crate::error::OneVoiceError;
use crate::stages::asr::Asr;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise::{self, Denoiser};
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtChain;
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
    audio_io: AudioCapture,
    denoiser: Denoiser,
    vad: Vad,
    asr: Asr,
    mt: MtChain,
    glossary: Glossary,
    tts: TtsEngine,
    latency_budget: LatencyBudgetConfig,
    min_asr_confidence: f32,
}

/// Conservative default gate on ASR confidence. Whisper mean token log-prob
/// for real speech is typically ≳ −1; noise/hallucination output is lower.
/// Override with `ONEVOICE_MIN_ASR_CONFIDENCE` (the two backends have
/// different scales — calibrate on-device; see `src/stages/asr.rs`).
const DEFAULT_MIN_ASR_CONFIDENCE: f32 = -2.0;

fn min_asr_confidence() -> f32 {
    std::env::var("ONEVOICE_MIN_ASR_CONFIDENCE")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MIN_ASR_CONFIDENCE)
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

        // Direct `src→target` model, or a `src→pivot→target` chain when the
        // pair config declares a pivot (e.g. VI→KO via EN).
        let mt = match &pair.pivot {
            Some(p) => MtChain::pivot(
                cfg.models_dir.join(&pair.mt_model_dir),
                &p.lang,
                cfg.models_dir.join(&p.mt_model_dir),
                &pair.target,
            )?,
            None => MtChain::direct(cfg.models_dir.join(&pair.mt_model_dir), &pair.target)?,
        };

        Ok(Self {
            source_lang: cfg.source_lang.to_string(),
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
            asr: Asr::load(cfg.models_dir.join(&pair.asr_model_dir))?,
            mt,
            glossary: Glossary::load(&cfg.data_dir.join("glossary"))?,
            // espeak-ng-data is shared across all Piper voices (vi/en/zh);
            // ignored by the MMS (ko) backend -- see src/stages/tts.rs.
            tts: TtsEngine::load(
                cfg.models_dir.join(&pair.tts_model_dir),
                &cfg.models_dir.join("tts"),
            )?,
            latency_budget: cfg.config.latency_budget,
            min_asr_confidence: min_asr_confidence(),
        })
    }

    /// Runs one PTT press-release cycle end to end. Times each stage —
    /// this is the "button-release -> first audio" latency AGENTS.md
    /// targets at ≤1.2s, *excluding* one-time model loading in `new()`
    /// and the pre-release capture (see the `release_start` comment
    /// below). Measured on whatever CPU this binary runs on, not the
    /// Hexagon NPU target hardware — see the logged warning every run
    /// emits.
    pub fn run_once(&mut self) -> Result<(), OneVoiceError> {
        // Capture happens *before* the button is released, so it is not
        // part of the "button-release -> first audio" target (AGENTS.md ->
        // Latency target). It's still timed for reporting, but every
        // latency number below is anchored at `release_start` -- the moment
        // capture returns -- because on real PTT hardware capture blocks
        // for the whole utterance (seconds), which would otherwise dwarf
        // the actual post-release budget.
        let t = Instant::now();
        let raw = self.audio_io.capture_on_ptt()?;
        let capture_ms = t.elapsed().as_secs_f64() * 1000.0;

        let release_start = Instant::now();

        let t = Instant::now();
        let denoised = self.denoiser.process(raw);
        let denoise_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let trimmed = self.vad.trim(denoised)?;
        let vad_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Silero found no speech — don't hand ASR a 30s-padded silence
        // buffer (it would transcribe noise into garbage, then MT/TTS
        // would happily translate it). On a real device this is the LED
        // "no-speech" signal; here it's a clean no-op cycle.
        if trimmed.samples.is_empty() {
            tracing::info!(
                vad_ms = format!("{vad_ms:.1}"),
                "no speech detected after VAD trim — skipping ASR/MT/TTS"
            );
            return Ok(());
        }

        let t = Instant::now();
        let (utterance, asr_confidence) = self
            .asr
            .transcribe_with_confidence(&trimmed, &self.source_lang)?;
        let asr_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            text = %utterance.text,
            lang = %utterance.lang,
            confidence = format!("{asr_confidence:.3}"),
            "ASR"
        );

        // Whisper happily produces fluent text on noise-only input that VAD
        // lets through. If the decoder is very unsure, treat it like "no
        // speech" rather than translating a confident-sounding fabrication.
        // On a real device this is the "didn't catch that" LED/haptic state.
        if asr_confidence < self.min_asr_confidence {
            tracing::warn!(
                confidence = format!("{asr_confidence:.3}"),
                threshold = self.min_asr_confidence,
                "ASR confidence below threshold — skipping MT/TTS (likely noise)"
            );
            return Ok(());
        }

        let t = Instant::now();
        let translation = self.mt.translate(&utterance, &self.glossary)?;
        let mt_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(text = %translation.text, lang = %translation.lang, "MT");

        tracing::info!(
            capture_ms = format!("{capture_ms:.1}"),
            denoise_ms = format!("{denoise_ms:.1}"),
            vad_ms = format!("{vad_ms:.1}"),
            asr_ms = format!("{asr_ms:.1}"),
            mt_ms = format!("{mt_ms:.1}"),
            through_mt_ms = format!("{:.1}", release_start.elapsed().as_secs_f64() * 1000.0),
            "latency through MT (CPU dev machine, NOT the Hexagon NPU target -- see AGENTS.md -> Latency target; TTS timed separately below)"
        );

        // Streaming playback: open the output on the FIRST synthesized chunk
        // and write as each sentence completes, so audio starts before the
        // whole utterance is synthesized. `ttfa_ms` (release -> first audio)
        // is the metric AGENTS.md targets at ≤1.2s.
        let audio_io = &self.audio_io;
        let mut playback: Option<crate::stages::audio_io::Playback> = None;
        let first_audio_at: std::cell::Cell<Option<Instant>> = std::cell::Cell::new(None);
        let t = Instant::now();
        let tts_first_chunk_ms = self.tts.synthesize_streaming(&translation, &mut |chunk| {
            if playback.is_none() {
                playback = Some(audio_io.start_playback(chunk.sample_rate_hz)?);
                first_audio_at.set(Some(Instant::now()));
            }
            playback.as_mut().expect("just set").write(&chunk.samples)
        })?;
        let tts_ms = t.elapsed().as_secs_f64() * 1000.0;

        let play_ms = {
            let t = Instant::now();
            if let Some(p) = playback.take() {
                p.finalize()?;
            }
            t.elapsed().as_secs_f64() * 1000.0
        };

        let ttfa_ms = first_audio_at
            .get()
            .map(|t0| (t0 - release_start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        tracing::info!(
            tts_first_chunk_ms = format!("{tts_first_chunk_ms:.1}"),
            tts_total_ms = format!("{tts_ms:.1}"),
            ttfa_ms = format!("{ttfa_ms:.1}"),
            through_tts_ms = format!("{:.1}", release_start.elapsed().as_secs_f64() * 1000.0),
            "TTS streamed (ttfa_ms = button-release -> first audio; AGENTS.md targets <=1.2s)"
        );

        // Device-I/O duration, informational only. With the file-based
        // stand-in this is file read + write; on real hardware `capture_ms`
        // is the pre-release utterance and playback is streamed, neither of
        // which is post-release latency, so this budget check is a proxy.
        let io_overhead_ms = capture_ms + play_ms;
        let total_ms = release_start.elapsed().as_secs_f64() * 1000.0;

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
