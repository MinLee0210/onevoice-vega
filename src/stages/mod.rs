//! One module per pipeline stage, in the fixed order from AGENTS.md:
//! audio_io -> denoise -> vad -> asr -> mt -> (glossary, used by mt) -> tts.
//! `pipeline.rs` composes these; nothing here calls another stage module.

pub mod asr;
pub mod audio_io;
pub mod ctc;
pub mod denoise;
pub mod glossary;
pub mod mt;
pub mod seq2seq;
pub mod session;
pub mod tts;
pub mod vad;
