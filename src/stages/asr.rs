//! ASR encoder/decoder inference via `ort` (ONNX Runtime), greedy
//! (beam=1) decode — same KV-cache pattern as `mt`, see that module's doc
//! comment for the two ONNX Runtime gotchas that apply here too
//! (cross-attention KV only valid on step 0; execution provider matters for
//! zero-length tensors).
//!
//! `Decoder` supports two interchangeable decoder exports: the merged graph
//! (`decoder_model_merged.onnx`) and the split pair (`decoder_model.onnx` +
//! `decoder_with_past_model.onnx`). Split is preferred when present because
//! it compiles to a Hexagon QNN context binary on QCS6490 while the merged
//! graph's `use_cache_branch` If-node does not (see
//! docs/reports/qcs6490-deployment-evaluation-2026-10.md); both produce the
//! same transcription on CPU.
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
//! latency concern against AGENTS.md's ≤1.2s target — not yet measured
//! on real hardware, see docs/TODO.md.

use std::path::{Path, PathBuf};

use ndarray::{Array2, ArrayD};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use mel_spec::prelude::{BatchLogMelConfig, BatchLogMelSpectrogram};

use crate::audio::{AudioBuffer, Utterance};
use crate::error::OneVoiceError;
use crate::stages::seq2seq::{self, DecoderSpec, LayerCache};
use crate::stages::session;

const SAMPLE_RATE_HZ: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const N_SAMPLES: usize = 480_000; // 30s @ 16kHz -- see module docs on why this is fixed
const N_FRAMES: usize = 3_000;

const DECODER_START_TOKEN_ID: i64 = 50258;
const TASK_TRANSCRIBE_TOKEN_ID: i64 = 50359;
const NO_TIMESTAMPS_TOKEN_ID: i64 = 50363;
const EOS_TOKEN_ID: i64 = 50257;
const MAX_NEW_TOKENS: usize = 64;

/// Never generated (generation_config's `suppress_tokens`) — mostly
/// non-linguistic/other-task special tokens.
const SUPPRESS_TOKEN_IDS: &[i64] = &[
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359,
    503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268,
    3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331,
    12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
    26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254,
    50258, 50358, 50359, 50360, 50361, 50362,
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
        other => Err(OneVoiceError::Asr(format!(
            "unsupported ASR language {other:?}"
        ))),
    }
}

/// The decoder graph(s). Two interchangeable exports of the same weights:
///
/// * `Merged` — one graph with a `use_cache_branch` input (the current
///   default; what `setup_models.sh` ships). Runs ORT/CPU fine.
/// * `Split` — `decoder_model.onnx` (first step) + `decoder_with_past_model.onnx`
///   (cached steps). This is the export that compiles to a QNN context binary
///   for the Hexagon NPU — the merged graph's `use_cache_branch` If-node does
///   not (see docs/reports/qcs6490-deployment-evaluation-2026-10.md).
///
/// `AsrEngine::load` picks `Split` when both split files are present in the
/// model dir, otherwise falls back to `Merged`, so both artifacts work.
enum Decoder {
    Merged(Session),
    Split { first: Session, past: Session },
}

impl Decoder {
    fn from_model_dir(model_dir: &Path) -> Result<Self, OneVoiceError> {
        let first_path = model_dir.join("decoder_model.onnx");
        let past_path = model_dir.join("decoder_with_past_model.onnx");
        if first_path.exists() && past_path.exists() {
            Ok(Decoder::Split {
                first: session::commit_from_file(&first_path).map_err(|source| {
                    OneVoiceError::Asr(format!("failed to load {}: {source}", first_path.display()))
                })?,
                past: session::commit_from_file(&past_path).map_err(|source| {
                    OneVoiceError::Asr(format!("failed to load {}: {source}", past_path.display()))
                })?,
            })
        } else {
            let merged_path = model_dir.join("decoder_model_merged.onnx");
            Ok(Decoder::Merged(
                session::commit_from_file(&merged_path).map_err(|source| {
                    OneVoiceError::Asr(format!(
                        "failed to load {}: {source}",
                        merged_path.display()
                    ))
                })?,
            ))
        }
    }

    fn spec(&self) -> Result<DecoderSpec, String> {
        match self {
            Decoder::Merged(session) | Decoder::Split { first: session, .. } => {
                DecoderSpec::from_session(session)
            }
        }
    }
}

pub struct AsrEngine {
    tokenizer: Tokenizer,
    encoder: Session,
    decoder: Decoder,
    mel: BatchLogMelSpectrogram,
    spec: DecoderSpec,
}

impl AsrEngine {
    /// `model_dir` is a directory containing `tokenizer.json`,
    /// `encoder_model.onnx`, and either the merged decoder
    /// (`decoder_model_merged.onnx`) or the split pair
    /// (`decoder_model.onnx` + `decoder_with_past_model.onnx`) — see
    /// models/README.md and models/asr/vi/MANIFEST.md.
    pub fn load(model_dir: impl Into<std::path::PathBuf>) -> Result<Self, OneVoiceError> {
        let model_dir = model_dir.into();

        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|source| {
                OneVoiceError::Asr(format!("failed to load tokenizer.json: {source}"))
            })?;

        let encoder =
            session::commit_from_file(model_dir.join("encoder_model.onnx")).map_err(|source| {
                OneVoiceError::Asr(format!("failed to load encoder_model.onnx: {source}"))
            })?;

        let decoder = Decoder::from_model_dir(&model_dir)?;

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
        let mel = BatchLogMelSpectrogram::new(mel_config).map_err(|source| {
            OneVoiceError::Asr(format!("failed to build mel spectrogram: {source}"))
        })?;

        // Derived from the decoder graph, not hardcoded: PhoWhisper-base is
        // 6 layers / 8 heads, Whisper-small is 12 layers / 12 heads (see
        // `seq2seq::DecoderSpec`).
        let spec = decoder.spec().map_err(OneVoiceError::Asr)?;

        Ok(Self {
            tokenizer,
            encoder,
            decoder,
            mel,
            spec,
        })
    }

    /// Transcribes, returning the text plus a confidence score: the mean
    /// log-probability of the generated tokens. The pipeline uses it to gate
    /// hallucinated transcriptions on noise-only input (see `pipeline.rs`).
    /// Closer to 0 = more confident.
    pub fn transcribe_with_confidence(
        &mut self,
        audio: &AudioBuffer,
        lang: &str,
    ) -> Result<(Utterance, f32), OneVoiceError> {
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
        let (generated, confidence) =
            self.run_decoder_greedy(&encoder_hidden_states, lang_token)?;
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

        Ok((
            Utterance {
                text,
                lang: lang.to_string(),
            },
            confidence,
        ))
    }

    /// See module docs for why this isn't just `self.mel.compute(samples)`.
    fn log_mel_spectrogram(&self, samples: &[f32]) -> Result<Array2<f32>, OneVoiceError> {
        let mut padded = samples.to_vec();
        padded.resize(N_SAMPLES, 0.0); // pads short audio, truncates long audio to exactly 30s
        let reflect_padded = reflect_pad(&padded, N_FFT / 2);

        let raw = self.mel.compute(&reflect_padded).map_err(|source| {
            OneVoiceError::Asr(format!("mel spectrogram computation failed: {source}"))
        })?;

        let mut log10_trimmed = Array2::<f32>::zeros((N_MELS, N_FRAMES));
        for r in 0..N_MELS {
            for c in 0..N_FRAMES {
                log10_trimmed[[r, c]] = raw[[r, c]] / std::f32::consts::LN_10;
            }
        }

        let max_val = log10_trimmed
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let floor = max_val - 8.0;
        log10_trimmed.mapv_inplace(|v| (v.max(floor) + 4.0) / 4.0);
        Ok(log10_trimmed)
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<ArrayD<f32>, OneVoiceError> {
        let input = features
            .clone()
            .into_shape_with_order((1, N_MELS, N_FRAMES))
            .map_err(|source| {
                OneVoiceError::Asr(format!("failed to reshape mel features: {source}"))
            })?;
        let input_tensor = Tensor::from_array(input).map_err(|source| {
            OneVoiceError::Asr(format!("failed to build input_features tensor: {source}"))
        })?;

        let outputs = self
            .encoder
            .run(ort::inputs!["input_features" => input_tensor])
            .map_err(|source| OneVoiceError::Asr(format!("encoder inference failed: {source}")))?;

        outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|source| {
                OneVoiceError::Asr(format!("failed to read encoder output: {source}"))
            })
            .map(|view| view.to_owned())
    }

    /// Returns the generated token ids and the mean per-token log-probability
    /// (the confidence signal used by the pipeline's hallucination gate).
    fn run_decoder_greedy(
        &mut self,
        encoder_hidden_states: &ArrayD<f32>,
        lang_token: i64,
    ) -> Result<(Vec<u32>, f32), OneVoiceError> {
        let num_layers = self.spec.num_layers;
        let mut cache: Vec<LayerCache> = (0..num_layers)
            .map(|_| LayerCache::empty(&self.spec))
            .collect();
        let mut current_input_ids = vec![
            DECODER_START_TOKEN_ID,
            lang_token,
            TASK_TRANSCRIBE_TOKEN_ID,
            NO_TIMESTAMPS_TOKEN_ID,
        ];
        let mut generated = Vec::new();
        let mut logprob_sum = 0.0f64;
        let mut logprob_n = 0usize;

        for step in 0..MAX_NEW_TOKENS {
            // Merged: one graph, `use_cache_branch` flips after step 0.
            // Split: step 0 through the no-cache `first` graph (the whole
            // prompt); later steps through `past` with the KV cache. Both
            // feed `current_input_ids`, which is the full prompt on step 0
            // and the last generated token thereafter.
            let outputs = match &mut self.decoder {
                Decoder::Merged(session) => {
                    let feeds = build_merged_feeds(
                        num_layers,
                        &current_input_ids,
                        step > 0,
                        encoder_hidden_states,
                        &cache,
                    )?;
                    session.run(feeds).map_err(|source| {
                        OneVoiceError::Asr(format!(
                            "decoder inference failed at step {step}: {source}"
                        ))
                    })?
                }
                Decoder::Split { first, past } => {
                    if step == 0 {
                        let feeds = build_first_feeds(&current_input_ids, encoder_hidden_states)?;
                        first.run(feeds).map_err(|source| {
                            OneVoiceError::Asr(format!(
                                "decoder(first) inference failed at step {step}: {source}"
                            ))
                        })?
                    } else {
                        let feeds = build_past_feeds(current_input_ids[0], &cache)?;
                        past.run(feeds).map_err(|source| {
                            OneVoiceError::Asr(format!(
                                "decoder(past) inference failed at step {step}: {source}"
                            ))
                        })?
                    }
                }
            };

            let (shape, logits) =
                outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|source| {
                        OneVoiceError::Asr(format!("failed to read decoder logits: {source}"))
                    })?;
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
            let next_id = seq2seq::argmax(&last_step_logits) as i64;
            logprob_sum += log_softmax_at(&last_step_logits, next_id as usize) as f64;
            logprob_n += 1;

            seq2seq::extract_present_caches(&outputs, &mut cache, step == 0)
                .map_err(OneVoiceError::Asr)?;

            if next_id == EOS_TOKEN_ID {
                break;
            }
            generated.push(next_id as u32);
            current_input_ids = vec![next_id];
        }

        // Hitting the cap means the loop never saw EOS: a long utterance was
        // silently cut off. Warn so it shows up in telemetry rather than as
        // a mysteriously short transcript.
        if generated.len() == MAX_NEW_TOKENS {
            tracing::warn!(
                max_new_tokens = MAX_NEW_TOKENS,
                "ASR decode hit the max-new-tokens cap — transcript may be truncated"
            );
        }

        let confidence = if logprob_n > 0 {
            (logprob_sum / logprob_n as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        Ok((generated, confidence))
    }
}

/// Log-softmax evaluated at `idx` (logprob of the chosen token). Handles the
/// `-inf` entries the suppress-token masking writes.
fn log_softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum_exp: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    logits[idx] - (max + sum_exp.ln())
}

/// Feeds for the merged decoder: `use_cache_branch` selects no-cache (step 0)
/// vs cached (later steps) inside the one graph.
fn build_merged_feeds<'a>(
    num_layers: usize,
    input_ids: &[i64],
    use_cache: bool,
    encoder_hidden_states: &ArrayD<f32>,
    cache: &[LayerCache],
) -> Result<Vec<(String, SessionInputValue<'a>)>, OneVoiceError> {
    let mut feeds = Vec::with_capacity(3 + num_layers * 4);
    feeds.push((
        "input_ids".to_string(),
        tensor_input(([1usize, input_ids.len()], input_ids.to_vec()))?,
    ));
    feeds.push((
        "encoder_hidden_states".to_string(),
        tensor_input(encoder_hidden_states.clone())?,
    ));
    feeds.push((
        "use_cache_branch".to_string(),
        tensor_input(([1usize], vec![use_cache]))?,
    ));
    push_past_feeds(&mut feeds, cache)?;
    Ok(feeds)
}

/// Feeds for the split `first` decoder (step 0): the prompt plus the encoder
/// hidden states; no cache inputs.
fn build_first_feeds<'a>(
    input_ids: &[i64],
    encoder_hidden_states: &ArrayD<f32>,
) -> Result<Vec<(String, SessionInputValue<'a>)>, OneVoiceError> {
    let feeds = vec![
        (
            "input_ids".to_string(),
            tensor_input(([1usize, input_ids.len()], input_ids.to_vec()))?,
        ),
        (
            "encoder_hidden_states".to_string(),
            tensor_input(encoder_hidden_states.clone())?,
        ),
    ];
    Ok(feeds)
}

/// Feeds for the split `past` decoder (steps > 0): one token plus the full
/// KV cache (self- and cross-attention; the cross-attention KV captured on
/// step 0 is fed back, not recomputed).
fn build_past_feeds<'a>(
    next_id: i64,
    cache: &[LayerCache],
) -> Result<Vec<(String, SessionInputValue<'a>)>, OneVoiceError> {
    let mut feeds = Vec::with_capacity(1 + cache.len() * 4);
    feeds.push((
        "input_ids".to_string(),
        tensor_input(([1usize, 1usize], vec![next_id]))?,
    ));
    push_past_feeds(&mut feeds, cache)?;
    Ok(feeds)
}

fn push_past_feeds<'a>(
    feeds: &mut Vec<(String, SessionInputValue<'a>)>,
    cache: &[LayerCache],
) -> Result<(), OneVoiceError> {
    for (i, layer) in cache.iter().enumerate() {
        feeds.push((
            format!("past_key_values.{i}.decoder.key"),
            tensor_input(layer.decoder_key.clone())?,
        ));
        feeds.push((
            format!("past_key_values.{i}.decoder.value"),
            tensor_input(layer.decoder_value.clone())?,
        ));
        feeds.push((
            format!("past_key_values.{i}.encoder.key"),
            tensor_input(layer.encoder_key.clone())?,
        ));
        feeds.push((
            format!("past_key_values.{i}.encoder.value"),
            tensor_input(layer.encoder_value.clone())?,
        ));
    }
    Ok(())
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

/// Thin wrapper mapping the shared `seq2seq::tensor_input` string error
/// onto this stage's `OneVoiceError::Asr` variant.
fn tensor_input<'a, T, D>(data: D) -> Result<SessionInputValue<'a>, OneVoiceError>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
    D: ort::value::OwnedTensorArrayData<T>,
{
    seq2seq::tensor_input(data).map_err(OneVoiceError::Asr)
}

/// ASR backend selector. Whisper (seq2seq, KV-cache autoregressive decode) or
/// CTC (single forward pass + greedy decode — see `ctc.rs`). Chosen by which
/// artifact is in the model dir: `model.onnx` → CTC; otherwise the Whisper
/// `encoder_model.onnx` + decoder pair. So the backend is selected purely by
/// `configs/language-pairs.toml`'s `asr_model_dir`.
// The two engines differ in size (Whisper carries more state); one instance
// lives for the whole process, so the enum-size lint is noise here.
#[allow(clippy::large_enum_variant)]
pub enum Asr {
    Whisper(AsrEngine),
    Ctc(crate::stages::ctc::CtcAsrEngine),
}

impl Asr {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        let dir = model_dir.into();
        if dir.join("model.onnx").exists() {
            Ok(Asr::Ctc(crate::stages::ctc::CtcAsrEngine::load(dir)?))
        } else {
            Ok(Asr::Whisper(AsrEngine::load(dir)?))
        }
    }

    /// Returns the transcript plus a confidence score (Whisper: mean token
    /// log-prob; CTC: mean frame max-softmax prob). The two scales differ —
    /// the pipeline's gate threshold is calibrated per backend.
    pub fn transcribe_with_confidence(
        &mut self,
        audio: &AudioBuffer,
        lang: &str,
    ) -> Result<(Utterance, f32), OneVoiceError> {
        match self {
            Asr::Whisper(engine) => engine.transcribe_with_confidence(audio, lang),
            Asr::Ctc(engine) => engine.transcribe_with_confidence(audio, lang),
        }
    }
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

        let mut reader = hound::WavReader::open("data/eval/vad_sample_vi.wav")
            .expect("failed to open wav fixture");
        let sample_rate = reader.spec().sample_rate;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
            .collect();
        let audio = AudioBuffer::new(samples, sample_rate);

        let (utterance, confidence) = asr
            .transcribe_with_confidence(&audio, "vi")
            .expect("transcribe failed");

        assert_eq!(utterance.text.trim(), "nửa vòng trái đất hơn bảy năm.");
        assert_eq!(utterance.lang, "vi");
        assert!(confidence.is_finite(), "confidence {confidence}");
    }
}
