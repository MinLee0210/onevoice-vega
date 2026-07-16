//! Voice activity detection: Silero VAD via `ort`. Runs on CPU, trims
//! leading/trailing silence around the utterance before it reaches ASR.
//!
//! Silero VAD is stateful — a small recurrent hidden state carried
//! between calls — and processes fixed 512-sample chunks (32ms at
//! 16kHz, its documented window size). See models/vad/MANIFEST.md for
//! the exact ONNX I/O signature this was built against.

use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

use crate::audio::AudioBuffer;
use crate::error::OneVoiceError;

const CHUNK_SIZE: usize = 512;
const STATE_SHAPE: [usize; 3] = [2, 1, 128];
const SPEECH_THRESHOLD: f32 = 0.5;

pub struct Vad {
    session: Session,
}

impl Vad {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, OneVoiceError> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path.into()))
            .map_err(|source| OneVoiceError::Vad(format!("failed to load VAD model: {source}")))?;
        Ok(Self { session })
    }

    /// Trims leading/trailing silence based on per-chunk speech
    /// probability. Interior pauses (e.g. mid-sentence) are left alone —
    /// only the edges are trimmed.
    pub fn trim(&mut self, audio: AudioBuffer) -> Result<AudioBuffer, OneVoiceError> {
        let probs = self.speech_probabilities(&audio)?;

        let Some(first_speech) = probs.iter().position(|&p| p > SPEECH_THRESHOLD) else {
            // No speech detected — return empty rather than the raw
            // audio, so ASR doesn't waste a call transcribing noise.
            return Ok(AudioBuffer::new(Vec::new(), audio.sample_rate_hz));
        };
        let last_speech = probs.iter().rposition(|&p| p > SPEECH_THRESHOLD).unwrap();

        let start = first_speech * CHUNK_SIZE;
        let end = ((last_speech + 1) * CHUNK_SIZE).min(audio.samples.len());
        Ok(AudioBuffer::new(
            audio.samples[start..end].to_vec(),
            audio.sample_rate_hz,
        ))
    }

    fn speech_probabilities(&mut self, audio: &AudioBuffer) -> Result<Vec<f32>, OneVoiceError> {
        let mut state = ArrayD::<f32>::zeros(IxDyn(&STATE_SHAPE));
        let sr = audio.sample_rate_hz as i64;
        let mut probs = Vec::with_capacity(audio.samples.len() / CHUNK_SIZE);

        for chunk in audio.samples.chunks(CHUNK_SIZE) {
            if chunk.len() < CHUNK_SIZE {
                break; // drop a trailing partial chunk shorter than Silero's window
            }

            let input =
                Tensor::from_array(([1usize, CHUNK_SIZE], chunk.to_vec())).map_err(|source| {
                    OneVoiceError::Vad(format!("failed to build input tensor: {source}"))
                })?;
            let sr_tensor = Tensor::from_array(((), vec![sr])).map_err(|source| {
                OneVoiceError::Vad(format!("failed to build sr tensor: {source}"))
            })?;
            let state_tensor = Tensor::from_array(state.clone()).map_err(|source| {
                OneVoiceError::Vad(format!("failed to build state tensor: {source}"))
            })?;

            let outputs = self
                .session
                .run(ort::inputs!["input" => input, "state" => state_tensor, "sr" => sr_tensor])
                .map_err(|source| OneVoiceError::Vad(format!("VAD inference failed: {source}")))?;

            let (_, prob) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|source| {
                    OneVoiceError::Vad(format!("failed to read VAD output: {source}"))
                })?;
            probs.push(prob[0]);

            state = outputs["stateN"]
                .try_extract_array::<f32>()
                .map_err(|source| {
                    OneVoiceError::Vad(format!("failed to read VAD state: {source}"))
                })?
                .to_owned();
        }

        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires `models/vad/model.onnx` on disk (gitignored; see
    /// models/vad/MANIFEST.md) and `data/eval/vad_sample_vi.wav` (a real
    /// ~2.4s VIVOS utterance, committed as a small test fixture — see
    /// data/README.md). Run with `cargo test -- --ignored`.
    ///
    /// Uses real speech rather than a synthetic tone: Silero VAD is a
    /// trained speech/non-speech classifier, not an energy detector — a
    /// loud pure sine tone does *not* register as speech to it, so a
    /// synthetic signal doesn't exercise the real behavior.
    #[test]
    #[ignore = "requires models/vad/model.onnx and data/eval/vad_sample_vi.wav"]
    fn trims_silence_around_real_speech() {
        let mut vad = Vad::load("models/vad/model.onnx").expect("failed to load Vad");

        let mut reader = hound::WavReader::open("data/eval/vad_sample_vi.wav")
            .expect("failed to open wav fixture");
        let sample_rate = reader.spec().sample_rate;
        let speech: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
            .collect();

        let silence = vec![0.0f32; sample_rate as usize]; // 1s
        let mut samples = silence.clone();
        samples.extend_from_slice(&speech);
        samples.extend_from_slice(&silence);
        let audio = AudioBuffer::new(samples, sample_rate);
        let original_len = audio.samples.len();

        let trimmed = vad.trim(audio).expect("trim failed");

        assert!(
            trimmed.samples.len() < original_len,
            "expected silence to be trimmed off"
        );
        // Allow some slack: VAD trims to chunk boundaries and speech
        // onset/offset isn't perfectly sharp, but it should land close
        // to the real speech length, not the padded total.
        assert!(
            trimmed.samples.len() < speech.len() + CHUNK_SIZE * 4,
            "trimmed length {} should be close to the speech length {}",
            trimmed.samples.len(),
            speech.len()
        );
        assert!(
            !trimmed.samples.is_empty(),
            "expected the speech to be kept"
        );
    }
}
