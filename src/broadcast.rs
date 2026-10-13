//! Vega Air CLI wrapper: file capture in, one WAV per target out. The stage
//! orchestration lives in [`crate::engine::BroadcastEngine`]; this module is
//! file I/O + reporting (the demo UI calls the engine directly).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::Config;
use crate::engine::{BroadcastEngine, BroadcastEngineConfig};
use crate::error::OneVoiceError;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise;

pub struct BroadcastConfig<'a> {
    pub config: &'a Config,
    pub models_dir: &'a Path,
    pub data_dir: &'a Path,
    pub source_lang: &'a str,
    pub target_langs: &'a [String],
    pub input_wav: &'a Path,
    /// Output template; `out.wav` -> `out.<lang>.wav` per target.
    pub output_wav: &'a Path,
}

pub struct Broadcast {
    input_wav: PathBuf,
    output_wav: PathBuf,
    engine: BroadcastEngine,
}

/// `out.wav` -> `out.<lang>.wav`.
fn target_path(template: &Path, lang: &str) -> PathBuf {
    let stem = template
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");
    let ext = template
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("wav");
    let parent = template.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{stem}.{lang}.{ext}"))
}

impl Broadcast {
    pub fn new(cfg: BroadcastConfig) -> Result<Self, OneVoiceError> {
        let engine = BroadcastEngine::new(BroadcastEngineConfig {
            config: cfg.config,
            models_dir: cfg.models_dir,
            data_dir: cfg.data_dir,
            source_lang: cfg.source_lang,
            target_langs: cfg.target_langs,
        })?;
        Ok(Self {
            input_wav: cfg.input_wav.to_path_buf(),
            output_wav: cfg.output_wav.to_path_buf(),
            engine,
        })
    }

    /// One broadcast: capture, ASR once, then MT+TTS per target, each to its
    /// own WAV.
    pub fn run_once(&mut self) -> Result<(), OneVoiceError> {
        let cycle_start = Instant::now();

        let mut capture_io = AudioCapture::new(
            denoise::INPUT_SAMPLE_RATE_HZ,
            &self.input_wav,
            Path::new("/tmp/onevoice_broadcast_scratch.wav"),
        );
        let raw = capture_io.capture_on_ptt()?;
        let outcome = self.engine.process(raw)?;

        tracing::info!(text = %outcome.transcript, confidence = format!("{:.3}", outcome.confidence), "broadcast ASR");
        if outcome.no_speech {
            tracing::info!("no speech detected — broadcast aborted");
            return Ok(());
        }
        if outcome.low_confidence {
            tracing::warn!(
                confidence = format!("{:.3}", outcome.confidence),
                "low ASR confidence — broadcast aborted"
            );
            return Ok(());
        }

        for leg in &outcome.legs {
            let out = target_path(&self.output_wav, &leg.lang);
            let mut out_io =
                AudioCapture::new(denoise::INPUT_SAMPLE_RATE_HZ, &self.input_wav, &out);
            out_io.play(&leg.audio)?;
            tracing::info!(target = %leg.lang, text = %leg.translation, out = %out.display(), "broadcast leg");
        }

        tracing::info!(
            through_broadcast_ms = format!("{:.1}", cycle_start.elapsed().as_secs_f64() * 1000.0),
            targets = outcome.legs.len(),
            "broadcast complete (ASR once, then MT+TTS per language)"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_path_inserts_language_before_extension() {
        assert_eq!(
            target_path(Path::new("/tmp/out.wav"), "en"),
            PathBuf::from("/tmp/out.en.wav")
        );
        assert_eq!(
            target_path(Path::new("dist/bundle"), "zh"),
            PathBuf::from("dist/bundle.zh.wav")
        );
    }
}
