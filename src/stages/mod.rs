//! One module per pipeline stage, in the fixed order from CLAUDE.md:
//! audio_io -> denoise -> vad -> asr -> mt -> (glossary, used by mt) -> tts.
//! `pipeline.rs` composes these; nothing here calls another stage module.

pub mod asr;
pub mod audio_io;
pub mod denoise;
pub mod glossary;
pub mod mt;
pub mod tts;
pub mod vad;
