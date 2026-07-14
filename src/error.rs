use std::path::PathBuf;

use thiserror::Error;

/// Errors surfaced while building or running the pipeline. Stage-specific
/// variants (denoise/VAD/ASR/MT/TTS failures) get added here as those
/// stages move from `todo!()` stubs to real model inference.
#[derive(Debug, Error)]
pub enum OneVoiceError {
    #[error("failed to read config file {path}: {source}")]
    ConfigIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse config: {0}")]
    Config(#[from] toml::de::Error),

    #[error("no config found for language pair {source_lang} -> {target_lang}")]
    LanguagePairNotFound {
        source_lang: String,
        target_lang: String,
    },
}
