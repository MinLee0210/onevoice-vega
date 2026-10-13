//! In-memory orchestrator — the canonical stage order from AGENTS.md:
//! denoise → VAD trim → ASR → MT → TTS. Unlike [`pipeline::Pipeline`]
//! (file I/O), this takes an [`AudioBuffer`] and returns one, so a demo UI or
//! an IPC server can feed captured PCM and play back the translation. Both
//! `pipeline` and `broadcast` build on these stages.

use std::path::Path;
use std::time::Instant;

use crate::audio::AudioBuffer;
use crate::config::{Config, LanguagePair};
use crate::error::OneVoiceError;
use crate::stages::asr::Asr;
use crate::stages::denoise::Denoiser;
use crate::stages::glossary::Glossary;
use crate::stages::mt::MtChain;
use crate::stages::tts::TtsEngine;
use crate::stages::vad::Vad;

/// Conservative default gate on ASR confidence. Whisper mean token log-prob
/// for real speech is typically ≳ −1; noise/hallucination output is lower.
/// Override with `ONEVOICE_MIN_ASR_CONFIDENCE`.
pub const DEFAULT_MIN_ASR_CONFIDENCE: f32 = -2.0;
const OUTPUT_FALLBACK_RATE_HZ: u32 = 16_000;

fn min_asr_confidence() -> f32 {
    std::env::var("ONEVOICE_MIN_ASR_CONFIDENCE")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MIN_ASR_CONFIDENCE)
}

/// `ONEVOICE_SKIP_DENOISE=1` runs the no-denoise ablation arm (RNNoise off,
/// resample kept) — denoising can hurt ASR, so this must be measurable.
fn skip_denoise() -> bool {
    std::env::var("ONEVOICE_SKIP_DENOISE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub struct EngineConfig<'a> {
    pub config: &'a Config,
    pub models_dir: &'a Path,
    pub data_dir: &'a Path,
    pub source_lang: &'a str,
    pub target_lang: &'a str,
}

/// Per-stage timings (milliseconds) for one cycle.
#[derive(Debug, Default, Clone, Copy)]
pub struct Telemetry {
    pub denoise_ms: f64,
    pub vad_ms: f64,
    pub asr_ms: f64,
    pub mt_ms: f64,
    pub tts_first_chunk_ms: f64,
    pub tts_total_ms: f64,
    /// Engine start → first synthesized chunk ready (the "release → first
    /// audio" proxy the demo shows).
    pub to_first_audio_ms: f64,
    pub total_ms: f64,
}

/// Result of one translation cycle. `audio` is empty when `no_speech` /
/// `low_confidence` short-circuited the cycle.
#[derive(Debug)]
pub struct CycleOutcome {
    pub audio: AudioBuffer,
    pub transcript: String,
    pub translation: String,
    pub source_lang: String,
    pub target_lang: String,
    pub confidence: f32,
    pub no_speech: bool,
    pub low_confidence: bool,
    pub telemetry: Telemetry,
}

/// Builds an [`MtChain`] for a pair (direct or EN-pivot).
pub(crate) fn build_mt_chain(
    pair: &LanguagePair,
    models_dir: &Path,
) -> Result<MtChain, OneVoiceError> {
    match &pair.pivot {
        Some(p) => MtChain::pivot(
            models_dir.join(&pair.mt_model_dir),
            &p.lang,
            models_dir.join(&p.mt_model_dir),
            &pair.target,
        ),
        None => MtChain::direct(models_dir.join(&pair.mt_model_dir), &pair.target),
    }
}

pub struct Engine {
    source_lang: String,
    target_lang: String,
    denoiser: Denoiser,
    vad: Vad,
    asr: Asr,
    mt: MtChain,
    glossary: Glossary,
    tts: TtsEngine,
    min_asr_confidence: f32,
}

impl Engine {
    /// Loads all stages for one language pair (one-time; reuse the `Engine`
    /// across cycles).
    pub fn new(cfg: EngineConfig) -> Result<Self, OneVoiceError> {
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
            denoiser: if skip_denoise() {
                Denoiser::bypassed()
            } else {
                Denoiser::load()
            },
            vad: Vad::load(cfg.models_dir.join("vad/model.onnx"))?,
            asr: Asr::load(cfg.models_dir.join(&pair.asr_model_dir))?,
            mt: build_mt_chain(pair, cfg.models_dir)?,
            glossary: Glossary::load(&cfg.data_dir.join("glossary"))?,
            tts: TtsEngine::load(
                cfg.models_dir.join(&pair.tts_model_dir),
                &cfg.models_dir.join("tts"),
            )?,
            min_asr_confidence: min_asr_confidence(),
        })
    }

    /// Runs one full cycle on captured audio (48 kHz mono, RNNoise's rate).
    pub fn process(&mut self, raw: AudioBuffer) -> Result<CycleOutcome, OneVoiceError> {
        let start = Instant::now();
        let mut telem = Telemetry::default();

        let t = Instant::now();
        let denoised = self.denoiser.process(raw);
        telem.denoise_ms = ms(t);

        let t = Instant::now();
        let trimmed = self.vad.trim(denoised)?;
        telem.vad_ms = ms(t);

        if trimmed.samples.is_empty() {
            telem.total_ms = ms(start);
            return Ok(self.outcome(
                AudioBuffer::new(Vec::new(), OUTPUT_FALLBACK_RATE_HZ),
                String::new(),
                String::new(),
                0.0,
                true,
                false,
                telem,
            ));
        }

        let t = Instant::now();
        let (utterance, confidence) = self
            .asr
            .transcribe_with_confidence(&trimmed, &self.source_lang)?;
        telem.asr_ms = ms(t);

        if confidence < self.min_asr_confidence {
            telem.total_ms = ms(start);
            return Ok(self.outcome(
                AudioBuffer::new(Vec::new(), OUTPUT_FALLBACK_RATE_HZ),
                utterance.text,
                String::new(),
                confidence,
                false,
                true,
                telem,
            ));
        }

        let t = Instant::now();
        let translation = self.mt.translate(&utterance, &self.glossary)?;
        telem.mt_ms = ms(t);

        // Stream TTS chunks into one buffer; record engine-start → first chunk.
        let mut samples = Vec::new();
        let mut sample_rate_hz = 0u32;
        let mut first_audio_ms = 0.0f64;
        let t = Instant::now();
        telem.tts_first_chunk_ms = self.tts.synthesize_streaming(&translation, &mut |chunk| {
            if first_audio_ms == 0.0 {
                first_audio_ms = ms(start);
            }
            if sample_rate_hz == 0 {
                sample_rate_hz = chunk.sample_rate_hz;
            }
            samples.extend_from_slice(&chunk.samples);
            Ok(())
        })?;
        telem.tts_total_ms = ms(t);
        telem.to_first_audio_ms = first_audio_ms;
        telem.total_ms = ms(start);

        if sample_rate_hz == 0 {
            sample_rate_hz = OUTPUT_FALLBACK_RATE_HZ;
        }
        Ok(self.outcome(
            AudioBuffer::new(samples, sample_rate_hz),
            utterance.text,
            translation.text,
            confidence,
            false,
            false,
            telem,
        ))
    }

    /// Convenience: decode a WAV (48 kHz mono) and run one cycle.
    pub fn process_wav(&mut self, wav_bytes: &[u8]) -> Result<CycleOutcome, OneVoiceError> {
        let audio = wav_to_audio(wav_bytes)?;
        self.process(audio)
    }

    #[allow(clippy::too_many_arguments)]
    fn outcome(
        &self,
        audio: AudioBuffer,
        transcript: String,
        translation: String,
        confidence: f32,
        no_speech: bool,
        low_confidence: bool,
        telemetry: Telemetry,
    ) -> CycleOutcome {
        CycleOutcome {
            audio,
            transcript,
            translation,
            source_lang: self.source_lang.clone(),
            target_lang: self.target_lang.clone(),
            confidence,
            no_speech,
            low_confidence,
            telemetry,
        }
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Encodes a mono f32 buffer as 16-bit PCM WAV bytes (no file I/O).
pub fn audio_to_wav(audio: &AudioBuffer) -> Vec<u8> {
    let data_len = (audio.samples.len() * 2) as u32;
    let sr = audio.sample_rate_hz;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVE");
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Decodes WAV bytes into an [`AudioBuffer`] (any rate/channels → downmixed
/// mono; the caller is responsible for the 48 kHz RNNoise input rate).
pub fn wav_to_audio(bytes: &[u8]) -> Result<AudioBuffer, OneVoiceError> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|e| OneVoiceError::AudioIo(format!("failed to read WAV: {e}")))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>(),
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
    }
    .map_err(|e| OneVoiceError::AudioIo(format!("failed to read samples: {e}")))?;

    let samples = if spec.channels > 1 {
        let ch = spec.channels as usize;
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };
    Ok(AudioBuffer::new(samples, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_through_bytes() {
        let audio = AudioBuffer::new(vec![0.0, 0.5, -0.5, 1.0, -1.0], 16_000);
        let bytes = audio_to_wav(&audio);
        assert_eq!(&bytes[0..4], b"RIFF");
        let back = wav_to_audio(&bytes).expect("decode");
        assert_eq!(back.sample_rate_hz, 16_000);
        assert_eq!(back.samples.len(), audio.samples.len());
        for (a, b) in audio.samples.iter().zip(back.samples.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    /// End-to-end in-memory API: WAV bytes in → translated WAV bytes + the
    /// transcript/translation + telemetry. Requires `models/`/`configs/`/`data/`
    /// and must run from the repo root (`cargo test -- --ignored`).
    #[test]
    #[ignore = "requires models/, configs/, data/ (repo root)"]
    fn engine_processes_a_wav_in_memory() {
        let config = crate::config::Config::load(std::path::Path::new("configs")).expect("config");
        let mut engine = Engine::new(EngineConfig {
            config: &config,
            models_dir: std::path::Path::new("models"),
            data_dir: std::path::Path::new("data"),
            source_lang: "vi",
            target_lang: "en",
        })
        .expect("failed to build Engine");

        let wav = std::fs::read("data/eval/ptt_sample_vi_48k.wav").expect("fixture wav");
        let outcome = engine.process_wav(&wav).expect("process failed");

        assert!(!outcome.no_speech);
        assert!(!outcome.transcript.is_empty(), "empty transcript");
        assert!(!outcome.translation.is_empty(), "empty translation");
        assert!(!outcome.audio.samples.is_empty(), "empty output audio");
        assert!(outcome.telemetry.total_ms > 0.0);
        // Output WAV re-encodes cleanly (the demo plays exactly this).
        let out = audio_to_wav(&outcome.audio);
        assert_eq!(&out[0..4], b"RIFF");
    }

    #[test]
    fn wav_decode_downmixes_stereo() {
        // 2-channel 16-bit WAV built inline (L=+0.5,R=-0.5 -> 0.0).
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16_000u32.to_le_bytes());
        b.extend_from_slice(&64_000u32.to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&(i16::MAX / 2).to_le_bytes());
        b.extend_from_slice(&(-(i16::MAX / 2)).to_le_bytes());
        let audio = wav_to_audio(&b).expect("decode");
        assert_eq!(audio.samples.len(), 1);
        assert!(audio.samples[0].abs() < 1e-3);
    }
}
