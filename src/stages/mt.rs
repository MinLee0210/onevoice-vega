//! MT encoder/decoder inference via `ort` (ONNX Runtime), greedy
//! (beam=1) decode to hit the latency target (see AGENTS.md -> Latency
//! target). Implemented for the vi-en pair against the Opus-MT export in
//! experiments/03_mt.ipynb — see that notebook (sections 2-6) for how
//! the ONNX export, the tokenizer.json conversion, and the merged-decoder
//! KV-cache wiring were derived and validated against the PyTorch
//! reference before being ported here. In particular, two gotchas found
//! there and worth remembering if this breaks again:
//!
//! 1. `decoder_model_merged.onnx`'s `present.*.encoder.*` (cross-attention
//!    KV) outputs are only valid on the first decode step
//!    (`use_cache_branch = false`); they must be captured once and held
//!    fixed, not re-read every step.
//! 2. On the dev machine used for that notebook, `onnxruntime`'s Python
//!    wheel defaults to `CoreMLExecutionProvider`, which mishandles this
//!    graph's zero-length KV-cache tensors. Not an issue here since `ort`
//!    only compiles in CPU unless an execution-provider Cargo feature
//!    (e.g. `coreml`, `qnn`) is explicitly enabled — worth remembering if
//!    one gets added later.
//!
//! ## Glossary enforcement (decoder logit bias)
//!
//! `translate()` runs a cheap unbiased greedy decode first. If every
//! glossary target term relevant to this sentence (via
//! `Glossary::find_matches`) already appears in that output, it's
//! returned as-is. Only if something's missing does a second decode
//! pass run with a logit bias toward the missing term's tokens — this
//! keeps the 5/10 sentences that don't need correction (in the
//! validation set below) at full unbiased quality, and only pays the
//! extra decode pass on the sentences that actually need it.
//!
//! The bias itself tracks one "next expected token" position per
//! glossary target sequence (`progress` in `run_decoder_greedy`), not
//! independent per-position checks — an earlier prototype that biased
//! every matching position independently produced runaway repetition
//! loops ("fork fork fork fork...") once a sequence was partway through,
//! because both "restart the sequence" and "continue it" got biased on
//! the same step and competed. Validated in
//! experiments/03_mt.ipynb section 7 (7a-7d) against this exact
//! model/tokenizer before being ported here: went from 50% to 100% term
//! accuracy on a 10-sentence glossary carrier set, with three real bugs
//! caught and fixed along the way (see that notebook for the blow-by-blow
//! and docs/decisions/glossary-runtime-enforcement-2026-07.md for the
//! summary). Known limitation: biased output is grammatically rougher
//! than unbiased ("forklift approaching sector three." drops "The"/"is")
//! — acceptable for a safety-critical term to be present and
//! intelligible, not yet polished. LoRA fine-tuning (AGENTS.md's
//! original plan) would fix this more thoroughly but is a much bigger
//! task; this is the cheap-to-ship stopgap.
//!
//! Runs on CPU here; the QNN execution provider for the Hexagon NPU
//! (AGENTS.md -> Compute split) is future work — `ort` supports it via
//! the `qnn` Cargo feature, not yet enabled.

use std::path::PathBuf;

use ndarray::ArrayD;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::audio::{Translation, Utterance};
use crate::error::OneVoiceError;
use crate::stages::glossary::Glossary;
use crate::stages::seq2seq::{self, DecoderSpec, LayerCache};

const MAX_NEW_TOKENS: usize = 64;
/// Logit bias added to a glossary target sequence's next expected token.
/// Validated in experiments/03_mt.ipynb section 7b: the real logit gap
/// between "_lift" and "_fork" (the known "forklift" mistranslation) was
/// 9.46 at the deciding decode step, so anything below ~9.5 provably
/// can't flip it; 12.0 gives margin without being tuned so high that it
/// starts winning over strongly-preferred alternatives elsewhere.
const GLOSSARY_BIAS: f32 = 12.0;

pub struct MtEngine {
    tokenizer: Tokenizer,
    encoder: Session,
    decoder: Session,
    pad_token_id: i64,
    eos_token_id: i64,
    spec: DecoderSpec,
}

impl MtEngine {
    /// `model_dir` is a directory containing `tokenizer.json`,
    /// `encoder_model.onnx`, and `decoder_model_merged.onnx` (see
    /// models/README.md and models/mt/vi-en/MANIFEST.md).
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        let model_dir = model_dir.into();

        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|source| {
                OneVoiceError::Mt(format!("failed to load tokenizer.json: {source}"))
            })?;

        let encoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_dir.join("encoder_model.onnx")))
            .map_err(|source| {
                OneVoiceError::Mt(format!("failed to load encoder_model.onnx: {source}"))
            })?;

        let decoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_dir.join("decoder_model_merged.onnx")))
            .map_err(|source| {
                OneVoiceError::Mt(format!(
                    "failed to load decoder_model_merged.onnx: {source}"
                ))
            })?;

        let pad_token_id = tokenizer
            .token_to_id("<pad>")
            .ok_or_else(|| OneVoiceError::Mt("vocab has no <pad> token".to_string()))?
            as i64;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .ok_or_else(|| OneVoiceError::Mt("vocab has no </s> token".to_string()))?
            as i64;

        let spec = DecoderSpec::from_session(&decoder).map_err(OneVoiceError::Mt)?;

        Ok(Self {
            tokenizer,
            encoder,
            decoder,
            pad_token_id,
            eos_token_id,
            spec,
        })
    }

    pub fn translate(
        &mut self,
        utterance: &Utterance,
        target_lang: &str,
        glossary: &Glossary,
    ) -> Result<Translation, OneVoiceError> {
        tracing::debug!(source_lang = %utterance.lang, target_lang, "running MT");

        let encoding = self
            .tokenizer
            .encode(utterance.text.as_str(), false)
            .map_err(|source| OneVoiceError::Mt(format!("tokenizer encode failed: {source}")))?;
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        input_ids.push(self.eos_token_id); // tokenizer.json has no post-processor to add this automatically
        let seq_len = input_ids.len();
        let attention_mask = vec![1i64; seq_len];

        let encoder_hidden_states = self.run_encoder(&input_ids, &attention_mask, seq_len)?;

        let baseline_generated =
            self.run_decoder_greedy(&encoder_hidden_states, &attention_mask, seq_len, &[])?;
        let baseline_text = self.decode_ids(&baseline_generated)?;

        let matches = glossary.find_matches(&utterance.lang, target_lang, &utterance.text);
        let baseline_lower = baseline_text.to_lowercase();
        let missing_terms: Vec<(&str, &str)> = matches
            .into_iter()
            .filter(|(_, target_term)| !baseline_lower.contains(&target_term.to_lowercase()))
            .collect();

        if missing_terms.is_empty() {
            return Ok(Translation {
                text: baseline_text,
                lang: target_lang.to_string(),
            });
        }

        tracing::debug!(
            missing_terms = ?missing_terms.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            "glossary term(s) missing from unbiased translation, running biased decode pass"
        );
        let bias_seqs: Vec<Vec<i64>> = missing_terms
            .iter()
            .map(|(_, target_term)| -> Result<Vec<i64>, OneVoiceError> {
                let encoding = self
                    .tokenizer
                    .encode(*target_term, false)
                    .map_err(|source| {
                        OneVoiceError::Mt(format!(
                            "tokenizer encode failed for glossary term {target_term:?}: {source}"
                        ))
                    })?;
                Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
            })
            .collect::<Result<_, _>>()?;

        let biased_generated =
            self.run_decoder_greedy(&encoder_hidden_states, &attention_mask, seq_len, &bias_seqs)?;
        let text = self.decode_ids(&biased_generated)?;

        Ok(Translation {
            text,
            lang: target_lang.to_string(),
        })
    }

    fn decode_ids(&self, ids: &[i64]) -> Result<String, OneVoiceError> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&ids_u32, true)
            .map_err(|source| OneVoiceError::Mt(format!("tokenizer decode failed: {source}")))
    }

    fn run_encoder(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
        seq_len: usize,
    ) -> Result<ArrayD<f32>, OneVoiceError> {
        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids.to_vec()))
            .map_err(|source| {
                OneVoiceError::Mt(format!("failed to build input_ids tensor: {source}"))
            })?;
        let attention_mask_tensor =
            Tensor::from_array(([1usize, seq_len], attention_mask.to_vec())).map_err(|source| {
                OneVoiceError::Mt(format!("failed to build attention_mask tensor: {source}"))
            })?;

        let outputs = self
            .encoder
            .run(ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor])
            .map_err(|source| OneVoiceError::Mt(format!("encoder inference failed: {source}")))?;

        outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|source| OneVoiceError::Mt(format!("failed to read encoder output: {source}")))
            .map(|view| view.to_owned())
    }

    /// `bias_seqs`: glossary target term token sequences to bias the
    /// decode toward — see the module docs' "Glossary enforcement"
    /// section. Empty for a plain unbiased decode.
    fn run_decoder_greedy(
        &mut self,
        encoder_hidden_states: &ArrayD<f32>,
        encoder_attention_mask: &[i64],
        encoder_seq_len: usize,
        bias_seqs: &[Vec<i64>],
    ) -> Result<Vec<i64>, OneVoiceError> {
        let mut cache: Vec<LayerCache> = (0..self.spec.num_layers)
            .map(|_| LayerCache::empty(&self.spec))
            .collect();
        let mut next_id = self.pad_token_id;
        let mut generated: Vec<i64> = Vec::new();
        // Index of the next expected token in each bias_seqs entry --
        // exactly one biased position per sequence at any time, not
        // every position independently (see module docs for why that
        // distinction matters).
        let mut progress = vec![0usize; bias_seqs.len()];
        let mut satisfied = vec![false; bias_seqs.len()];

        for step in 0..MAX_NEW_TOKENS {
            let feeds = self.build_decoder_feeds(
                next_id,
                step > 0,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_seq_len,
                &cache,
            )?;

            let outputs = self.decoder.run(feeds).map_err(|source| {
                OneVoiceError::Mt(format!("decoder inference failed at step {step}: {source}"))
            })?;

            let (shape, logits) =
                outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|source| {
                        OneVoiceError::Mt(format!("failed to read decoder logits: {source}"))
                    })?;
            let vocab_size = *shape.last().expect("logits tensor is never rank-0") as usize;
            let last_step_logits = &logits[logits.len() - vocab_size..];

            let mut biased_logits;
            let step_logits: &[f32] = if bias_seqs.is_empty() {
                last_step_logits
            } else {
                biased_logits = last_step_logits.to_vec();
                for (i, seq) in bias_seqs.iter().enumerate() {
                    if !satisfied[i] {
                        biased_logits[seq[progress[i]] as usize] += GLOSSARY_BIAS;
                    }
                }
                &biased_logits
            };

            next_id = seq2seq::argmax_excluding(step_logits, self.pad_token_id as usize) as i64;
            if next_id == self.eos_token_id {
                break;
            }
            generated.push(next_id);

            for (i, seq) in bias_seqs.iter().enumerate() {
                if satisfied[i] {
                    continue;
                }
                if next_id == seq[progress[i]] {
                    progress[i] += 1;
                    if progress[i] == seq.len() {
                        satisfied[i] = true;
                    }
                } else {
                    // Didn't continue the match -- restart it if this
                    // token happens to also be the sequence's first
                    // token, otherwise back to square one.
                    progress[i] = if seq.len() > 1 && seq[0] == next_id {
                        1
                    } else {
                        0
                    };
                }
            }

            seq2seq::extract_present_caches(&outputs, &mut cache, step == 0)
                .map_err(OneVoiceError::Mt)?;
        }

        Ok(generated)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decoder_feeds<'a>(
        &self,
        next_id: i64,
        use_cache: bool,
        encoder_hidden_states: &ArrayD<f32>,
        encoder_attention_mask: &[i64],
        encoder_seq_len: usize,
        cache: &[LayerCache],
    ) -> Result<Vec<(String, SessionInputValue<'a>)>, OneVoiceError> {
        let mut feeds = Vec::with_capacity(4 + self.spec.num_layers * 4);

        feeds.push((
            "input_ids".to_string(),
            tensor_input(([1usize, 1usize], vec![next_id]))?,
        ));
        feeds.push((
            "encoder_attention_mask".to_string(),
            tensor_input(([1usize, encoder_seq_len], encoder_attention_mask.to_vec()))?,
        ));
        feeds.push((
            "encoder_hidden_states".to_string(),
            tensor_input(encoder_hidden_states.clone())?,
        ));
        feeds.push((
            "use_cache_branch".to_string(),
            tensor_input(([1usize], vec![use_cache]))?,
        ));

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

        Ok(feeds)
    }
}

/// Thin wrapper mapping the shared `seq2seq::tensor_input` string error
/// onto this stage's `OneVoiceError::Mt` variant.
fn tensor_input<'a, T, D>(data: D) -> Result<SessionInputValue<'a>, OneVoiceError>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
    D: ort::value::OwnedTensorArrayData<T>,
{
    seq2seq::tensor_input(data).map_err(OneVoiceError::Mt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires `models/mt/vi-en/` on disk (gitignored; built by running
    /// experiments/03_mt.ipynb) and must run from the repo root so the
    /// relative path resolves, e.g. `cargo test -- --ignored`. Checks
    /// against the same reference translation validated in that notebook.
    #[test]
    #[ignore = "requires models/mt/vi-en/, see experiments/03_mt.ipynb"]
    fn translates_vi_to_en_matching_python_reference() {
        let mut engine = MtEngine::load("models/mt/vi-en").expect("failed to load MtEngine");
        let utterance = Utterance {
            text: "Xin chào, bạn khỏe không?".to_string(),
            lang: "vi".to_string(),
        };

        let translation = engine
            .translate(&utterance, "en", &Glossary::new())
            .expect("translate failed");

        assert_eq!(translation.text, "Hello, how are you?");
        assert_eq!(translation.lang, "en");
    }

    #[test]
    #[ignore = "requires models/mt/en-vi/"]
    fn translates_en_to_vi_matching_python_reference() {
        let mut engine = MtEngine::load("models/mt/en-vi").expect("failed to load MtEngine");
        let utterance = Utterance {
            text: "Hello, how are you?".to_string(),
            lang: "en".to_string(),
        };

        let translation = engine
            .translate(&utterance, "vi", &Glossary::new())
            .expect("translate failed");

        assert_eq!(translation.text, "Xin chào, anh khỏe không?");
        assert_eq!(translation.lang, "vi");
    }

    /// The plain (no-glossary) baseline behavior from before glossary
    /// enforcement existed — kept as a regression check that an empty
    /// glossary is a true no-op.
    #[test]
    #[ignore = "requires models/mt/vi-en/, see experiments/03_mt.ipynb"]
    fn translates_industrial_sentence_matching_python_reference_without_glossary() {
        let mut engine = MtEngine::load("models/mt/vi-en").expect("failed to load MtEngine");
        let utterance = Utterance {
            text: "Xe nâng đang đến gần khu vực số ba.".to_string(),
            lang: "vi".to_string(),
        };

        let translation = engine
            .translate(&utterance, "en", &Glossary::new())
            .expect("translate failed");

        assert_eq!(translation.text, "The lift is approaching sector three.");
    }

    /// The known failure case this whole mechanism exists for: the
    /// generic model mistranslates "xe nâng" (forklift) as "lift". With
    /// the real termbase entry loaded, the biased decode pass should
    /// kick in and correct it. Exact string validated in
    /// experiments/03_mt.ipynb section 7d against this same model.
    #[test]
    #[ignore = "requires models/mt/vi-en/, see experiments/03_mt.ipynb"]
    fn glossary_corrects_forklift_mistranslation() {
        let mut engine = MtEngine::load("models/mt/vi-en").expect("failed to load MtEngine");
        let mut glossary = Glossary::new();
        glossary.insert("vi", "en", "xe nâng", "forklift");
        let utterance = Utterance {
            text: "Xe nâng đang đến gần khu vực số ba.".to_string(),
            lang: "vi".to_string(),
        };

        let translation = engine
            .translate(&utterance, "en", &glossary)
            .expect("translate failed");

        assert_eq!(translation.text, "forklift approaching sector three.");
    }

    /// When the unbiased pass already contains the glossary target term,
    /// the biased pass must not run -- otherwise every glossary-flavored
    /// sentence pays for a fluency-degrading second decode it doesn't
    /// need (see the module docs' section 7d rationale).
    #[test]
    #[ignore = "requires models/mt/vi-en/, see experiments/03_mt.ipynb"]
    fn glossary_leaves_an_already_correct_translation_unchanged() {
        let mut engine = MtEngine::load("models/mt/vi-en").expect("failed to load MtEngine");
        let mut glossary = Glossary::new();
        glossary.insert("vi", "en", "kính bảo hộ", "safety glasses");
        let utterance = Utterance {
            text: "Vui lòng đeo kính bảo hộ trước khi vận hành máy.".to_string(),
            lang: "vi".to_string(),
        };

        let translation = engine
            .translate(&utterance, "en", &glossary)
            .expect("translate failed");

        assert_eq!(
            translation.text,
            "Please wear your safety glasses before the engine works."
        );
    }
}
