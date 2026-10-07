//! Shared autoregressive seq2seq decode plumbing for the ASR and MT
//! stages. Both the Whisper-family ASR and Opus-MT use the same
//! merged-decoder KV-cache ONNX shape (see
//! docs/decisions/onnx-runtime-merged-decoder-gotchas.md), so the
//! per-layer KV caches, tensor construction, argmax, and present-state
//! extraction are identical between `asr.rs` and `mt.rs` and live here.
//!
//! The stage-specific decode *loops* (ASR's suppress-token masking,
//! MT's glossary logit bias) stay in their own modules; they call back
//! into this module for the shared bits.

use std::fmt::Debug;

use ndarray::ArrayD;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{Tensor, ValueType};

/// Static decoder geometry, needed to size the KV caches.
///
/// Derived from the loaded decoder ONNX graph (`DecoderSpec::from_session`)
/// rather than hardcoded, because the ASR stage points at two different
/// Whisper sizes: PhoWhisper-base (`asr/vi`, 6 layers / 8 heads) and
/// Whisper-small (`asr/en-zh-ko`, 12 layers / 12 heads). Hardcoding one
/// size silently breaks the other — this is exactly the bug that
/// prompted this module.
#[derive(Debug, Clone, Copy)]
pub struct DecoderSpec {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl DecoderSpec {
    /// Counts the `present.*.decoder.key` outputs for the layer count and
    /// reads head count / head dim from the static dimensions of
    /// `present.0.decoder.key` (`[batch, heads, seq, head_dim]`; `batch`
    /// and `seq` are dynamic `-1`, `heads` and `head_dim` are static).
    ///
    /// Returns a `String` error so each caller can wrap it in its own
    /// `OneVoiceError` variant.
    pub fn from_session(decoder: &Session) -> Result<Self, String> {
        let outputs = decoder.outputs();

        let num_layers = outputs
            .iter()
            .filter(|o| {
                let name = o.name();
                name.starts_with("present.") && name.ends_with(".decoder.key")
            })
            .count();
        if num_layers == 0 {
            return Err("decoder graph has no `present.*.decoder.key` outputs".to_string());
        }

        let first = outputs
            .iter()
            .find(|o| o.name() == "present.0.decoder.key")
            .ok_or_else(|| "decoder graph is missing `present.0.decoder.key`".to_string())?;

        let (num_heads, head_dim) = match first.dtype() {
            ValueType::Tensor { shape, .. } => {
                let dims: &[i64] = shape;
                if dims.len() != 4 {
                    return Err(format!(
                        "expected 4-D KV-cache tensor, got {}-D",
                        dims.len()
                    ));
                }
                let num_heads = dims[1];
                let head_dim = dims[3];
                if num_heads <= 0 || head_dim <= 0 {
                    return Err(format!(
                        "KV-cache heads/head_dim are dynamic, not static: heads={num_heads}, head_dim={head_dim}"
                    ));
                }
                (num_heads as usize, head_dim as usize)
            }
            other => {
                return Err(format!(
                    "`present.0.decoder.key` is not a tensor: {other:?}"
                ));
            }
        };

        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
        })
    }
}

/// Per-layer self-attention (decoder) and cross-attention (encoder) KV
/// cache, kept as dynamic-rank arrays so a zero-length first step
/// (`[1, heads, 0, head_dim]`) is just a normal empty array rather than
/// a special case.
pub struct LayerCache {
    pub decoder_key: ArrayD<f32>,
    pub decoder_value: ArrayD<f32>,
    pub encoder_key: ArrayD<f32>,
    pub encoder_value: ArrayD<f32>,
}

impl LayerCache {
    pub fn empty(spec: &DecoderSpec) -> Self {
        let empty =
            || ArrayD::<f32>::zeros(ndarray::IxDyn(&[1usize, spec.num_heads, 0, spec.head_dim]));
        Self {
            decoder_key: empty(),
            decoder_value: empty(),
            encoder_key: empty(),
            encoder_value: empty(),
        }
    }
}

/// Pulls the `present.*` KV-cache outputs out of a decode step and stores
/// them in `cache`. Cross-attention (encoder) KV is only valid on the
/// first (no-cache) step — see `src/stages/mt.rs` module docs — so it's
/// captured once (`first_step`) and held fixed thereafter.
pub fn extract_present_caches(
    outputs: &SessionOutputs,
    cache: &mut [LayerCache],
    first_step: bool,
) -> Result<(), String> {
    for (i, layer) in cache.iter_mut().enumerate() {
        layer.decoder_key = extract_owned(outputs, &format!("present.{i}.decoder.key"))?;
        layer.decoder_value = extract_owned(outputs, &format!("present.{i}.decoder.value"))?;
    }
    if first_step {
        for (i, layer) in cache.iter_mut().enumerate() {
            layer.encoder_key = extract_owned(outputs, &format!("present.{i}.encoder.key"))?;
            layer.encoder_value = extract_owned(outputs, &format!("present.{i}.encoder.value"))?;
        }
    }
    Ok(())
}

pub fn extract_owned(outputs: &SessionOutputs, name: &str) -> Result<ArrayD<f32>, String> {
    outputs[name]
        .try_extract_array::<f32>()
        .map(|view| view.to_owned())
        .map_err(|e| format!("failed to read decoder output {name}: {e}"))
}

pub fn tensor_input<'a, T, D>(data: D) -> Result<SessionInputValue<'a>, String>
where
    T: ort::value::PrimitiveTensorElementType + Debug + Clone + 'static,
    D: ort::value::OwnedTensorArrayData<T>,
{
    Tensor::from_array(data)
        .map(SessionInputValue::from)
        .map_err(|e| format!("failed to build input tensor: {e}"))
}

pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("logits is never empty")
}

/// argmax over `logits`, treating `excluded` as if its score were -inf
/// (matches generation_config's `bad_words_ids` masking out the pad token).
pub fn argmax_excluding(logits: &[f32], excluded: usize) -> usize {
    logits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != excluded)
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("logits is never empty")
}
