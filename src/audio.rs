//! Types passed between pipeline stages. Kept crate-wide (rather than
//! private to one stage) since every stage boundary in the pipeline
//! consumes or produces one of these.

/// Mono PCM audio buffer.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate_hz: u32) -> Self {
        Self {
            samples,
            sample_rate_hz,
        }
    }

    pub fn duration_ms(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate_hz as f32 * 1000.0
    }
}

/// ASR output: recognized text tagged with the language it was spoken in.
#[derive(Debug, Clone)]
pub struct Utterance {
    pub text: String,
    pub lang: String,
}

/// MT output: translated text tagged with the target language, ready for TTS.
#[derive(Debug, Clone)]
pub struct Translation {
    pub text: String,
    pub lang: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_ms_matches_sample_rate() {
        let one_second = AudioBuffer::new(vec![0.0; 16_000], 16_000);
        assert_eq!(one_second.duration_ms(), 1000.0);
    }
}
