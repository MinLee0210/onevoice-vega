//! OneVoice on-device speech-translation engine.
//!
//! Usable two ways:
//! - **Library** — [`engine::Engine`] runs the pipeline in-memory (audio bytes
//!   in → translated audio bytes + per-stage telemetry out). This is the API
//!   the demo front-end / LAN server drives (see
//!   `docs/decisions/demo-frontend-design-2026-10.md`).
//! - **Binary** — `onevoice` (`main.rs`) is the CLI: file-based [`pipeline::Pipeline`]
//!   (Vega Hand, one PTT cycle) and [`broadcast::Broadcast`] (Vega Air, one-to-many).
//!
//! Both paths share the same stages; [`engine`] is the canonical stage order.

pub mod audio;
pub mod broadcast;
pub mod config;
pub mod engine;
pub mod error;
pub mod pipeline;
pub mod stages;

pub use engine::{
    audio_to_wav, wav_to_audio, BroadcastEngine, BroadcastEngineConfig, BroadcastLeg,
    BroadcastOutcome, CycleOutcome, Engine, EngineConfig, Telemetry,
};
