//! ASR encoder/decoder inference via `ort` (ONNX Runtime), greedy
//! (beam=1) decode — same merged-decoder KV-cache pattern as `mt`, see
//! that module's doc comment for the two ONNX Runtime gotchas that
//! apply here too (cross-attention KV only valid on step 0; execution
//! provider matters for zero-length tensors).
//!
//! Unlike MT's tokenizer, Whisper needs a real signal-processing step
//! before the ONNX encoder: a log-mel spectrogram, computed here with
//! the `mel_spec` crate. That crate's `BatchLogMelSpectrogram::compute`
//! does NOT reproduce Whisper's exact preprocessing out of the box —
//! two differences found and corrected, both validated numerically
//! against `transformers.WhisperFeatureExtractor` and end-to-end against
//! the real ONNX model (mean abs diff ~0.0003 on the normalized
//! features; identical transcription on the reference test sentence):
//!
//! 1. `compute()` returns natural-log energies (`.ln()`), not `log10()`
//!    — must divide by `ln(10)` to get log10, which is what Whisper's
//!    normalization formula expects.
//! 2. Its `center: true` mode zero-pads the signal edges; PyTorch's
//!    `torch.stft(center=True)` (what Whisper actually uses) reflects
//!    the signal at the edges instead. Reflect-pad the audio ourselves
//!    and use `center: false` so the crate just consumes what we hand
//!    it. Also: Whisper explicitly drops the last of the resulting
//!    N+1 centered-STFT frames (`stft[..., :-1]` in openai/whisper) —
//!    trim to `N_FRAMES` here to match.
//!
//! The encoder's exported ONNX graph declares a dynamic sequence-length
//! input dim, but actually requires exactly `N_FRAMES` (30s padded) —
//! its positional embeddings were baked in as a fixed-size constant at
//! export time. Confirmed by testing variable-length input directly: it
//! hard-errors. So every call costs a fixed ~30s-worth of encoder
//! compute regardless of actual utterance length, which is a real
//! latency concern against CLAUDE.md's ≤1.2s target — not yet measured
//! on real hardware, see docs/TODO.md.

use ndarray::{Array2, ArrayD};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use mel_spec::prelude::{BatchLogMelConfig, BatchLogMelSpectrogram};

use crate::audio::{AudioBuffer, Utterance};
use crate::error::OneVoiceError;

const SAMPLE_RATE_HZ: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const N_SAMPLES: usize = 480_000; // 30s @ 16kHz -- see module docs on why this is fixed
const N_FRAMES: usize = 3_000;

const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const DECODER_START_TOKEN_ID: i64 = 50258;
const TASK_TRANSCRIBE_TOKEN_ID: i64 = 50359;
const NO_TIMESTAMPS_TOKEN_ID: i64 = 50363;
const EOS_TOKEN_ID: i64 = 50257;
const MAX_NEW_TOKENS: usize = 64;

/// Never generated (generation_config's `suppress_tokens`) — mostly
/// non-linguistic/other-task special tokens.
const SUPPRESS_TOKEN_IDS: &[i64] = &[
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873,
    893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585,
    6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553,
    16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
    47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362,
];
/// Only suppressed at the first generated position (generation_config's
/// `begin_suppress_tokens`) — stops the model emitting a leading space
/// or an immediate end-of-transcript.
const BEGIN_SUPPRESS_TOKEN_IDS: &[i64] = &[220, 50257];

fn language_token_id(lang: &str) -> Result<i64, OneVoiceError> {
    match lang {
        "vi" => Ok(50278),
        "en" => Ok(50259),
        "zh" => Ok(50260),
        "ko" => Ok(50264),
        other => Err(OneVoiceError::Asr(format!("unsupported ASR language {other:?}"))),
    }
}

struct LayerCache {
    decoder_key: ArrayD<f32>,
    decoder_value: ArrayD<f32>,
    encoder_key: ArrayD<f32>,
    encoder_value: ArrayD<f32>,
}

impl LayerCache {
    fn empty() -> Self {
        let empty = || ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, NUM_HEADS, 0, HEAD_DIM]));
        Self {
            decoder_key: empty(),
            decoder_value: empty(),
            encoder_key: empty(),
            encoder_value: empty(),
        }
    }
}

pub struct AsrEngine {
    tokenizer: Tokenizer,
    encoder: Session,
    decoder: Session,
    mel: BatchLogMelSpectrogram,
}

impl AsrEngine {
    /// `model_dir` is a directory containing `tokenizer.json`,
    /// `encoder_model.onnx`, and `decoder_model_merged.onnx` (see
    /// models/README.md and models/asr/vi/MANIFEST.md).
    pub fn load(model_dir: impl Into<std::path::PathBuf>) -> Result<Self, OneVoiceError> {
        let model_dir = model_dir.into();

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|source| OneVoiceError::Asr(format!("failed to load tokenizer.json: {source}")))?;

        let encoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_dir.join("encoder_model.onnx")))
            .map_err(|source| OneVoiceError::Asr(format!("failed to load encoder_model.onnx: {source}")))?;

        let decoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_dir.join("decoder_model_merged.onnx")))
            .map_err(|source| OneVoiceError::Asr(format!("failed to load decoder_model_merged.onnx: {source}")))?;

        let mel_config = BatchLogMelConfig {
            sample_rate: SAMPLE_RATE_HZ as usize,
            n_fft: N_FFT,
            win_length: N_FFT,
            hop_length: HOP_LENGTH,
            n_mels: N_MELS,
            f_min: 0.0,
            f_max: None,
            htk: false,
            norm: true,
            preemphasis: 0.0,
            center: false, // we reflect-pad ourselves, see module docs
            log_zero_guard: 1e-10,
            pad_to: 0,
            normalize_per_feature: false,
        };
        let mel = BatchLogMelSpectrogram::new(mel_config)
            .map_err(|source| OneVoiceError::Asr(format!("failed to build mel spectrogram: {source}")))?;

        Ok(Self { tokenizer, encoder, decoder, mel })
    }

    pub fn transcribe(&mut self, audio: &AudioBuffer, lang: &str) -> Result<Utterance, OneVoiceError> {
        tracing::debug!(lang, duration_ms = audio.duration_ms(), "running ASR");

        assert_eq!(
            audio.sample_rate_hz, SAMPLE_RATE_HZ,
            "AsrEngine expects {SAMPLE_RATE_HZ}Hz input"
        );
        let lang_token = language_token_id(lang)?;

        let t = std::time::Instant::now();
        let features = self.log_mel_spectrogram(&audio.samples)?;
        let mel_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = std::time::Instant::now();
        let encoder_hidden_states = self.run_encoder(&features)?;
        let encoder_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = std::time::Instant::now();
        let generated = self.run_decoder_greedy(&encoder_hidden_states, lang_token)?;
        let decoder_ms = t.elapsed().as_secs_f64() * 1000.0;

        tracing::info!(
            mel_ms = format!("{mel_ms:.1}"),
            encoder_ms = format!("{encoder_ms:.1}"),
            decoder_ms = format!("{decoder_ms:.1}"),
            generated_tokens = generated.len(),
            "ASR breakdown (encoder always processes the fixed 30s-padded input, see module docs)"
        );

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|source| OneVoiceError::Asr(format!("tokenizer decode failed: {source}")))?;

        Ok(Utterance { text, lang: lang.to_string() })
    }

    /// See module docs for why this isn't just `self.mel.compute(samples)`.
    fn log_mel_spectrogram(&self, samples: &[f32]) -> Result<Array2<f32>, OneVoiceError> {
        let mut padded = samples.to_vec();
        padded.resize(N_SAMPLES, 0.0); // pads short audio, truncates long audio to exactly 30s
        let reflect_padded = reflect_pad(&padded, N_FFT / 2);

        let raw = self
            .mel
            .compute(&reflect_padded)
            .map_err(|source| OneVoiceError::Asr(format!("mel spectrogram computation failed: {source}")))?;

        let mut log10_trimmed = Array2::<f32>::zeros((N_MELS, N_FRAMES));
        for r in 0..N_MELS {
            for c in 0..N_FRAMES {
                log10_trimmed[[r, c]] = raw[[r, c]] / std::f32::consts::LN_10;
            }
        }

        let max_val = log10_trimmed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let floor = max_val - 8.0;
        log10_trimmed.mapv_inplace(|v| (v.max(floor) + 4.0) / 4.0);
        Ok(log10_trimmed)
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<ArrayD<f32>, OneVoiceError> {
        let input = features.clone().into_shape_with_order((1, N_MELS, N_FRAMES)).map_err(|source| {
            OneVoiceError::Asr(format!("failed to reshape mel features: {source}"))
        })?;
        let input_tensor = Tensor::from_array(input)
            .map_err(|source| OneVoiceError::Asr(format!("failed to build input_features tensor: {source}")))?;

        let outputs = self
            .encoder
            .run(ort::inputs!["input_features" => input_tensor])
            .map_err(|source| OneVoiceError::Asr(format!("encoder inference failed: {source}")))?;

        outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|source| OneVoiceError::Asr(format!("failed to read encoder output: {source}")))
            .map(|view| view.to_owned())
    }

    fn run_decoder_greedy(
        &mut self,
        encoder_hidden_states: &ArrayD<f32>,
        lang_token: i64,
    ) -> Result<Vec<u32>, OneVoiceError> {
        let mut cache: Vec<LayerCache> = (0..NUM_LAYERS).map(|_| LayerCache::empty()).collect();
        let mut current_input_ids =
            vec![DECODER_START_TOKEN_ID, lang_token, TASK_TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID];
        let mut generated = Vec::new();

        for step in 0..MAX_NEW_TOKENS {
            let feeds = self.build_decoder_feeds(&current_input_ids, step > 0, encoder_hidden_states, &cache)?;

            let outputs = self
                .decoder
                .run(feeds)
                .map_err(|source| OneVoiceError::Asr(format!("decoder inference failed at step {step}: {source}")))?;

            let (shape, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|source| OneVoiceError::Asr(format!("failed to read decoder logits: {source}")))?;
            let vocab_size = *shape.last().expect("logits tensor is never rank-0") as usize;
            let mut last_step_logits = logits[logits.len() - vocab_size..].to_vec();

            for &t in SUPPRESS_TOKEN_IDS {
                last_step_logits[t as usize] = f32::NEG_INFINITY;
            }
            if step == 0 {
                for &t in BEGIN_SUPPRESS_TOKEN_IDS {
                    last_step_logits[t as usize] = f32::NEG_INFINITY;
                }
            }
            let next_id = argmax(&last_step_logits) as i64;

            for (i, layer) in cache.iter_mut().enumerate() {
                layer.decoder_key = extract_owned(&outputs, &format!("present.{i}.decoder.key"))?;
                layer.decoder_value = extract_owned(&outputs, &format!("present.{i}.decoder.value"))?;
            }
            if step == 0 {
                // Cross-attention KV is only correctly populated on the
                // first (no-cache) call -- see the `mt` module doc comment.
                for (i, layer) in cache.iter_mut().enumerate() {
                    layer.encoder_key = extract_owned(&outputs, &format!("present.{i}.encoder.key"))?;
                    layer.encoder_value = extract_owned(&outputs, &format!("present.{i}.encoder.value"))?;
                }
            }

            if next_id == EOS_TOKEN_ID {
                break;
            }
            generated.push(next_id as u32);
            current_input_ids = vec![next_id];
        }

        Ok(generated)
    }

    fn build_decoder_feeds<'a>(
        &self,
        input_ids: &[i64],
        use_cache: bool,
        encoder_hidden_states: &ArrayD<f32>,
        cache: &[LayerCache],
    ) -> Result<Vec<(String, SessionInputValue<'a>)>, OneVoiceError> {
        let mut feeds = Vec::with_capacity(3 + NUM_LAYERS * 4);

        feeds.push(("input_ids".to_string(), tensor_input(([1usize, input_ids.len()], input_ids.to_vec()))?));
        feeds.push(("encoder_hidden_states".to_string(), tensor_input(encoder_hidden_states.clone())?));
        feeds.push(("use_cache_branch".to_string(), tensor_input(([1usize], vec![use_cache]))?));

        for (i, layer) in cache.iter().enumerate() {
            feeds.push((format!("past_key_values.{i}.decoder.key"), tensor_input(layer.decoder_key.clone())?));
            feeds.push((format!("past_key_values.{i}.decoder.value"), tensor_input(layer.decoder_value.clone())?));
            feeds.push((format!("past_key_values.{i}.encoder.key"), tensor_input(layer.encoder_key.clone())?));
            feeds.push((format!("past_key_values.{i}.encoder.value"), tensor_input(layer.encoder_value.clone())?));
        }

        Ok(feeds)
    }
}

/// PyTorch's `torch.stft(center=True)` (what Whisper actually uses) pads
/// by reflecting `pad` samples off each edge -- see module docs.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    out.extend(samples[1..=pad].iter().rev().copied());
    out.extend_from_slice(samples);
    let n = samples.len();
    out.extend(samples[n - pad - 1..n - 1].iter().rev().copied());
    out
}

fn tensor_input<'a, T, D>(data: D) -> Result<SessionInputValue<'a>, OneVoiceError>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
    D: ort::value::OwnedTensorArrayData<T>,
{
    Tensor::from_array(data)
        .map(SessionInputValue::from)
        .map_err(|source| OneVoiceError::Asr(format!("failed to build input tensor: {source}")))
}

fn extract_owned(outputs: &ort::session::SessionOutputs, name: &str) -> Result<ArrayD<f32>, OneVoiceError> {
    outputs[name]
        .try_extract_array::<f32>()
        .map_err(|source| OneVoiceError::Asr(format!("failed to read decoder output {name}: {source}")))
        .map(|view| view.to_owned())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("logits is never empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_mirrors_without_duplicating_the_edge_sample() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&samples, 2);
        // left: reverse of samples[1..=2] = [2,3] -> [3,2]; right: reverse of samples[2..4] = [3,4] -> [4,3]
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    /// Requires `models/asr/vi/` on disk (gitignored; see
    /// experiments/02_asr.ipynb) and `data/eval/vad_sample_vi.wav`. Run
    /// with `cargo test -- --ignored`. This exact recipe (reflect-pad,
    /// ln->log10, frame trim) was validated numerically against
    /// `transformers.WhisperFeatureExtractor` before being ported here —
    /// see the module doc comment.
    #[test]
    #[ignore = "requires models/asr/vi/ and data/eval/vad_sample_vi.wav"]
    fn transcribes_vi_speech_matching_the_known_reference() {
        let mut asr = AsrEngine::load("models/asr/vi").expect("failed to load AsrEngine");

        let mut reader = hound::WavReader::open("data/eval/vad_sample_vi.wav").expect("failed to open wav fixture");
        let sample_rate = reader.spec().sample_rate;
        let samples: Vec<f32> =
            reader.samples::<i16>().map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32).collect();
        let audio = AudioBuffer::new(samples, sample_rate);

        let utterance = asr.transcribe(&audio, "vi").expect("transcribe failed");

        assert_eq!(utterance.text.trim(), "nửa vòng trái đất hơn bảy năm.");
        assert_eq!(utterance.lang, "vi");
    }
}
