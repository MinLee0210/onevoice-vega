//! Vega Hand — file-based one-shot: capture a WAV, run one PT Tcycle through
//! [`Engine`], write the translated audio, log per-stage telemetry and check
//! the latency budget. The stage order lives in `engine.rs`; this module is
//! the CLI's file I/O + reporting wrapper (the demo UI / server use `Engine`
//! directly).

use std::path::Path;
use std::time::Instant;

use crate::config::{Config, LatencyBudgetConfig};
use crate::engine::{Engine, EngineConfig};
use crate::error::OneVoiceError;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise;

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
    audio_io: AudioCapture,
    engine: Engine,
    latency_budget: LatencyBudgetConfig,
}

impl Pipeline {
    pub fn new(cfg: PipelineConfig) -> Result<Self, OneVoiceError> {
        let engine = Engine::new(EngineConfig {
            config: cfg.config,
            models_dir: cfg.models_dir,
            data_dir: cfg.data_dir,
            source_lang: cfg.source_lang,
            target_lang: cfg.target_lang,
        })?;

        Ok(Self {
            // The mic must capture at RNNoise's fixed 48kHz; `Denoiser`
            // (inside `Engine`) resamples down to 16kHz for VAD/ASR.
            audio_io: AudioCapture::new(
                denoise::INPUT_SAMPLE_RATE_HZ,
                cfg.input_wav_path,
                cfg.output_wav_path,
            ),
            engine,
            latency_budget: cfg.config.latency_budget,
        })
    }

    /// Runs one PTT press-release cycle end to end and reports the
    /// "button-release → first audio" latency AGENTS.md targets at ≤1.2s
    /// (excluding one-time model loading and the pre-release capture).
    pub fn run_once(&mut self) -> Result<(), OneVoiceError> {
        let t = Instant::now();
        let raw = self.audio_io.capture_on_ptt()?;
        let capture_ms = t.elapsed().as_secs_f64() * 1000.0;

        let release_start = Instant::now();
        let outcome = self.engine.process(raw)?;
        let te = outcome.telemetry;

        tracing::info!(
            text = %outcome.transcript,
            lang = %outcome.source_lang,
            confidence = format!("{:.3}", outcome.confidence),
            "ASR"
        );
        if outcome.no_speech {
            tracing::info!("no speech detected after VAD trim — skipping MT/TTS");
            return Ok(());
        }
        if outcome.low_confidence {
            tracing::warn!(
                confidence = format!("{:.3}", outcome.confidence),
                "ASR confidence below threshold — skipping MT/TTS (likely noise)"
            );
            return Ok(());
        }
        tracing::info!(text = %outcome.translation, lang = %outcome.target_lang, "MT");

        tracing::info!(
            capture_ms = format!("{capture_ms:.1}"),
            denoise_ms = format!("{:.1}", te.denoise_ms),
            vad_ms = format!("{:.1}", te.vad_ms),
            asr_ms = format!("{:.1}", te.asr_ms),
            mt_ms = format!("{:.1}", te.mt_ms),
            through_mt_ms = format!("{:.1}", te.denoise_ms + te.vad_ms + te.asr_ms + te.mt_ms),
            "latency through MT (CPU dev machine, NOT the Hexagon NPU target — see AGENTS.md → Latency target)"
        );

        let t = Instant::now();
        self.audio_io.play(&outcome.audio)?;
        let play_ms = t.elapsed().as_secs_f64() * 1000.0;

        let ttfa_ms = te.to_first_audio_ms;
        tracing::info!(
            tts_first_chunk_ms = format!("{:.1}", te.tts_first_chunk_ms),
            tts_total_ms = format!("{:.1}", te.tts_total_ms),
            ttfa_ms = format!("{ttfa_ms:.1}"),
            through_tts_ms = format!("{:.1}", release_start.elapsed().as_secs_f64() * 1000.0),
            "TTS streamed (ttfa_ms = button-release → first audio; AGENTS.md targets ≤1.2s)"
        );

        // Device-I/O duration, informational only (file read + write here;
        // on device capture is pre-release and playback is streamed).
        let io_overhead_ms = capture_ms + play_ms;
        let total_ms = release_start.elapsed().as_secs_f64() * 1000.0;
        self.check_budget(&te, io_overhead_ms, total_ms);
        Ok(())
    }

    fn check_budget(&self, te: &crate::engine::Telemetry, io_overhead_ms: f64, total_ms: f64) {
        let budget = self.latency_budget;
        let stage = [
            ("VAD", te.vad_ms, budget.stage.vad_trim_ms as f64),
            ("ASR", te.asr_ms, budget.stage.asr_ms as f64),
            ("MT", te.mt_ms, budget.stage.mt_ms as f64),
            (
                "TTS first chunk",
                te.tts_first_chunk_ms,
                budget.stage.tts_first_chunk_ms as f64,
            ),
            (
                "I/O overhead",
                io_overhead_ms,
                budget.stage.io_overhead_ms as f64,
            ),
        ];
        for (name, got, cap) in stage {
            if got > cap {
                tracing::warn!("{name} latency {got:.1}ms exceeded stage budget of {cap:.0}ms");
            }
        }

        if total_ms > budget.target_total_ms as f64 {
            tracing::warn!(
                "Total latency {total_ms:.1}ms exceeded target total budget of {}ms",
                budget.target_total_ms
            );
        } else if total_ms > budget.stretch_total_ms as f64 {
            tracing::info!(
                "Total latency {total_ms:.1}ms met target total budget of {}ms but exceeded stretch budget of {}ms",
                budget.target_total_ms,
                budget.stretch_total_ms
            );
        } else {
            tracing::info!(
                "Total latency {total_ms:.1}ms met stretch budget of {}ms",
                budget.stretch_total_ms
            );
        }
    }
}
