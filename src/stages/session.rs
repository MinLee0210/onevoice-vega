//! Shared ONNX Runtime session construction for every ML-backed stage
//! (`asr`, `mt`, `vad`, `tts`).
//!
//! Centralizes `Session::builder().commit_from_file(...)` so all stages get
//! the same thread configuration. ONNX Runtime otherwise sizes its intra-op
//! thread pool to the machine's physical cores; on the target QCS6490
//! (8 Kryo cores) several resident sessions each claiming every core can
//! oversubscribe while the CPU-side denoise/VAD/TTS work is also runnable.
//! `ONEVOICE_INTRA_THREADS` sets an explicit cap (unset = ORT's default, so
//! behavior is unchanged unless the env var is set — a device-tuning knob).

use std::path::Path;

use ort::session::Session;

/// Builds a session from an ONNX file, applying [`intra_threads`] if set.
pub fn commit_from_file(path: impl AsRef<Path>) -> Result<Session, ort::Error> {
    let mut builder = Session::builder()?;
    if let Some(n) = intra_threads() {
        builder = builder.with_intra_threads(n)?;
    }
    builder.commit_from_file(path.as_ref())
}

/// Explicit intra-op thread cap from `ONEVOICE_INTRA_THREADS`, if set to a
/// positive integer. `None` lets ONNX Runtime pick its default.
fn intra_threads() -> Option<usize> {
    std::env::var("ONEVOICE_INTRA_THREADS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&n| n > 0)
}
