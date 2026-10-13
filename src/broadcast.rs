//! Vega Air — one-to-many multilingual broadcast. One spoken instruction
//! (VI) is transcribed once, translated to each target language, and
//! synthesized sequentially, each to its own output WAV. This is the second
//! carrier's execution shape (see AGENTS.md → Vega Air); it reuses the exact
//! same stages and model set as Vega Hand (`Pipeline`), just composed
//! one-to-many and without a per-target ASR pass.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::audio::Translation;
use crate::config::Config;
use crate::error::OneVoiceError;
use crate::stages::asr::Asr;
use crate::stages::audio_io::AudioCapture;
use crate::stages::denoise::{self, Denoiser};
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtChain;
use crate::stages::tts::TtsEngine;
use crate::stages::vad::Vad;

/// Broadcast-tuned TTS rate: >1 spells each sentence out a little slower for
/// outdoor intelligibility (AGENTS.md → Vega Air). Applies to Piper voices
/// (EN/ZH); the MMS (KO) backend ignores it.
const BROADCAST_LENGTH_SCALE: f32 = 1.15;

struct Target {
    lang: String,
    mt: MtChain,
    tts: TtsEngine,
    output_wav: PathBuf,
}

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
    source_lang: String,
    input_wav: PathBuf,
    denoiser: Denoiser,
    vad: Vad,
    asr: Asr,
    glossary: Glossary,
    targets: Vec<Target>,
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
        // ASR/VAD: reuse any pair whose source matches (ASR is source-side).
        let asr_pair = cfg
            .config
            .language_pairs
            .pairs
            .iter()
            .find(|p| p.source == cfg.source_lang)
            .ok_or_else(|| OneVoiceError::LanguagePairNotFound {
                source_lang: cfg.source_lang.to_string(),
                target_lang: "<any>".to_string(),
            })?;

        let mut targets = Vec::new();
        for target_lang in cfg.target_langs {
            let pair = cfg
                .config
                .language_pairs
                .find(cfg.source_lang, target_lang)
                .ok_or_else(|| OneVoiceError::LanguagePairNotFound {
                    source_lang: cfg.source_lang.to_string(),
                    target_lang: target_lang.clone(),
                })?;
            let mut tts = TtsEngine::load(
                cfg.models_dir.join(&pair.tts_model_dir),
                &cfg.models_dir.join("tts"),
            )?;
            tts.set_length_scale(BROADCAST_LENGTH_SCALE);
            targets.push(Target {
                lang: target_lang.clone(),
                mt: crate::engine::build_mt_chain(pair, cfg.models_dir)?,
                tts,
                output_wav: target_path(cfg.output_wav, target_lang),
            });
        }

        Ok(Self {
            source_lang: cfg.source_lang.to_string(),
            input_wav: cfg.input_wav.to_path_buf(),
            denoiser: Denoiser::load(),
            vad: Vad::load(cfg.models_dir.join("vad/model.onnx"))?,
            asr: Asr::load(cfg.models_dir.join(&asr_pair.asr_model_dir))?,
            glossary: Glossary::load(&cfg.data_dir.join("glossary"))?,
            targets,
        })
    }

    /// One broadcast: capture, ASR once, then MT+TTS per target sequentially.
    pub fn run_once(&mut self) -> Result<(), OneVoiceError> {
        let cycle_start = Instant::now();

        let mut capture_io = AudioCapture::new(
            denoise::INPUT_SAMPLE_RATE_HZ,
            &self.input_wav,
            Path::new("/tmp/onevoice_broadcast_scratch.wav"),
        );
        let raw = capture_io.capture_on_ptt()?;
        let denoised = self.denoiser.process(raw);
        let trimmed = self.vad.trim(denoised)?;
        if trimmed.samples.is_empty() {
            tracing::info!("no speech detected — broadcast aborted");
            return Ok(());
        }

        let (utterance, confidence) = self
            .asr
            .transcribe_with_confidence(&trimmed, &self.source_lang)?;
        tracing::info!(text = %utterance.text, confidence = format!("{confidence:.3}"), "broadcast ASR");

        for target in &mut self.targets {
            let t = Instant::now();
            let translation: Translation = target.mt.translate(&utterance, &self.glossary)?;
            let (audio, _ttfa) = target.tts.synthesize(&translation)?;

            let mut out_io = AudioCapture::new(
                denoise::INPUT_SAMPLE_RATE_HZ,
                &self.input_wav,
                &target.output_wav,
            );
            out_io.play(&audio)?;

            tracing::info!(
                target = %target.lang,
                text = %translation.text,
                out = %target.output_wav.display(),
                ms = format!("{:.1}", t.elapsed().as_secs_f64() * 1000.0),
                "broadcast leg"
            );
        }

        tracing::info!(
            through_broadcast_ms = format!("{:.1}", cycle_start.elapsed().as_secs_f64() * 1000.0),
            targets = self.targets.len(),
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
