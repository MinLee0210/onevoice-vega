//! CTC ASR (Wav2Vec2-family): a single forward pass over the raw waveform
//! followed by greedy CTC decode — **no autoregressive decoder loop**, which
//! is the whole latency argument for CTC over Whisper (see
//! `docs/reports/asr_ab.md`: ~5× faster but higher WER on the VI model tested).
//!
//! Loads `model.onnx` (`input_values [1, N]` → `logits [1, T, V]`),
//! `vocab.json` (token → id) and `preprocessor_config.json` (`do_normalize`)
//! from the model dir — see `scripts/edge/export_ctc_asr.py`.
//!
//! `Asr::load` (in `asr.rs`) picks this engine when the model dir contains
//! `model.onnx`, and the Whisper engine otherwise, so both backends are
//! selectable purely by which artifact is in `configs/language-pairs.toml`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::audio::{AudioBuffer, Utterance};
use crate::error::OneVoiceError;
use crate::stages::session;

const SAMPLE_RATE_HZ: u32 = 16_000;
/// Wav2Vec2 feature-extractor variance floor.
const NORM_EPS: f32 = 1e-7;

pub struct CtcAsrEngine {
    session: Session,
    /// id → token, indexed by id.
    id_to_token: Vec<String>,
    blank_id: usize,
    do_normalize: bool,
}

impl CtcAsrEngine {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        let model_dir = model_dir.into();

        let session = session::commit_from_file(model_dir.join("model.onnx"))
            .map_err(|s| OneVoiceError::Asr(format!("failed to load model.onnx: {s}")))?;

        let vocab_raw = std::fs::read_to_string(model_dir.join("vocab.json"))
            .map_err(|s| OneVoiceError::Asr(format!("failed to read vocab.json: {s}")))?;
        let vocab: HashMap<String, i64> = serde_json::from_str(&vocab_raw)
            .map_err(|s| OneVoiceError::Asr(format!("failed to parse vocab.json: {s}")))?;

        let size = vocab
            .values()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        let mut id_to_token = vec![String::new(); size];
        for (tok, id) in &vocab {
            if let Some(slot) = id_to_token.get_mut(*id as usize) {
                *slot = tok.clone();
            }
        }

        let blank_id = vocab
            .get("<pad>")
            .map(|&v| v as usize)
            .ok_or_else(|| OneVoiceError::Asr("vocab.json has no <pad> (CTC blank)".into()))?;

        let do_normalize = read_do_normalize(&model_dir.join("preprocessor_config.json"));
        let tokenizer_has_blank = id_to_token.get(blank_id).is_some();
        if !tokenizer_has_blank {
            return Err(OneVoiceError::Asr(format!(
                "blank id {blank_id} out of vocab range ({size})"
            )));
        }

        Ok(Self {
            session,
            id_to_token,
            blank_id,
            do_normalize,
        })
    }

    /// `lang` is ignored (a CTC model is single-language) but kept for a
    /// uniform signature with `AsrEngine`. Returns the transcript plus the
    /// mean per-frame max-softmax probability (0..1) as a confidence signal.
    pub fn transcribe_with_confidence(
        &mut self,
        audio: &AudioBuffer,
        lang: &str,
    ) -> Result<(Utterance, f32), OneVoiceError> {
        if audio.sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(OneVoiceError::Asr(format!(
                "CTC ASR expects {SAMPLE_RATE_HZ}Hz input, got {}Hz",
                audio.sample_rate_hz
            )));
        }

        let samples = if self.do_normalize {
            normalize(&audio.samples)
        } else {
            audio.samples.clone()
        };
        let n = samples.len();

        let input = Tensor::from_array(([1usize, n], samples))
            .map_err(|s| OneVoiceError::Asr(format!("failed to build input tensor: {s}")))?;
        let outputs = self
            .session
            .run(ort::inputs!["input_values" => input])
            .map_err(|s| OneVoiceError::Asr(format!("CTC inference failed: {s}")))?;

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|s| OneVoiceError::Asr(format!("failed to read logits: {s}")))?;
        let vocab = *shape.last().expect("logits is never rank-0") as usize;
        let frames = logits.len() / vocab.max(1);

        let mut best = Vec::with_capacity(frames);
        let mut conf_sum = 0.0f64;
        for t in 0..frames {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let mut argmax = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    argmax = i;
                }
            }
            // max-softmax prob for this frame = 1 / sum_j exp(x_j - max)
            let sum_exp: f32 = row.iter().map(|&v| (v - best_val).exp()).sum();
            if sum_exp.is_finite() && sum_exp > 0.0 {
                conf_sum += (1.0 / sum_exp) as f64;
            }
            best.push(argmax);
        }
        let confidence = if frames > 0 {
            (conf_sum / frames as f64) as f32
        } else {
            0.0
        };

        Ok((
            Utterance {
                text: ctc_greedy_decode(&best, &self.id_to_token, self.blank_id),
                lang: lang.to_string(),
            },
            confidence,
        ))
    }
}

/// Greedy CTC collapse: drop blanks and consecutive duplicates, map ids to
/// tokens, turn the `|` word delimiter into spaces. Pure — unit-tested.
pub fn ctc_greedy_decode(best: &[usize], id_to_token: &[String], blank_id: usize) -> String {
    let mut out = String::new();
    let mut prev = usize::MAX;
    for &id in best {
        if id != prev && id != blank_id {
            if let Some(tok) = id_to_token.get(id) {
                if tok == "|" {
                    out.push(' ');
                } else if !tok.starts_with('<') {
                    out.push_str(tok); // skip <s>/</s>/<unk> if they ever win
                }
            }
        }
        prev = id;
    }
    out.trim().to_string()
}

/// Wav2Vec2 utterance-level zero-mean unit-variance normalization.
fn normalize(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().max(1) as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let var = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
    let denom = (var + NORM_EPS).sqrt();
    samples.iter().map(|s| (s - mean) / denom).collect()
}

fn read_do_normalize(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v.get("do_normalize").and_then(|b| b.as_bool()))
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<String> {
        // id 0 = blank, 1 = 'a', 2 = 'b', 3 = '|', 4 = <unk>
        vec![
            "<pad>".into(),
            "a".into(),
            "b".into(),
            "|".into(),
            "<unk>".into(),
        ]
    }

    #[test]
    fn greedy_decode_collapses_repeats_and_drops_blanks() {
        // "a a _ a | b b" -> "aa b"
        let ids = [1, 1, 0, 1, 3, 2, 2];
        assert_eq!(ctc_greedy_decode(&ids, &vocab(), 0), "aa b");
    }

    #[test]
    fn greedy_decode_handles_leading_and_trailing_blanks() {
        let ids = [0, 0, 2, 0, 1, 0];
        assert_eq!(ctc_greedy_decode(&ids, &vocab(), 0), "ba");
    }

    #[test]
    fn greedy_decode_maps_word_delimiter_to_space() {
        let ids = [1, 3, 2];
        assert_eq!(ctc_greedy_decode(&ids, &vocab(), 0), "a b");
    }

    /// End-to-end against the real exported ONNX + WAV fixture. CTC accuracy
    /// is weaker than Whisper (see docs/reports/asr_ab.md), so this only
    /// checks the engine runs and emits characters — not an exact transcript.
    #[test]
    #[ignore = "requires models/asr/vi-ctc and data/eval/vad_sample_vi.wav"]
    fn transcribes_real_speech_through_the_onnx_model() {
        let mut asr = CtcAsrEngine::load("models/asr/vi-ctc").expect("failed to load CtcAsrEngine");
        let mut reader =
            hound::WavReader::open("data/eval/vad_sample_vi.wav").expect("failed to open wav");
        let sr = reader.spec().sample_rate;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect();
        let audio = AudioBuffer::new(samples, sr);

        let (utterance, confidence) = asr
            .transcribe_with_confidence(&audio, "vi")
            .expect("transcribe failed");

        assert!(!utterance.text.trim().is_empty(), "transcript was empty");
        assert_eq!(utterance.lang, "vi");
        assert!((0.0..=1.0).contains(&confidence), "confidence {confidence}");
    }

    #[test]
    fn normalize_is_zero_mean_unit_variance() {
        let out = normalize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = out.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }
}
