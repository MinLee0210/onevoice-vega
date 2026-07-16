//! Entry point: parse CLI args, load config, build the pipeline for one
//! language pair, then run one PTT press-release cycle. Everything below
//! "wire it up" is delegated to `pipeline`; this file has no pipeline
//! logic itself.

// ASR/TTS are still `todo!()` stubs, the glossary isn't wired into MT's
// decode yet, and the latency budget isn't read by anything yet either
// -- so plenty of types/fields/methods aren't exercised by any live code
// path (only by unit tests). Remove this once those land and the normal
// dead_code lint is meaningful again.
#![allow(dead_code)]

mod audio;
mod config;
mod error;
mod pipeline;
mod stages;

use std::path::PathBuf;

use clap::Parser;

use config::Config;
use pipeline::Pipeline;

/// OneVoice: on-device VI<->EN/ZH/KO speech translation for noisy
/// industrial environments. See CLAUDE.md for the full project spec.
///
/// Runs one push-to-talk cycle: reads `input_wav`, denoises, trims
/// silence, transcribes, translates, and (once TTS lands) synthesizes
/// and writes `output_wav`. Live mic/speaker capture isn't wired up yet
/// (see src/stages/audio_io.rs) -- both args are WAV file paths for now.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Input WAV file to translate (mono, 48kHz -- RNNoise's fixed input
    /// rate, see src/stages/denoise.rs)
    input_wav: PathBuf,

    /// Output WAV file to write the synthesized translation to
    output_wav: PathBuf,

    /// Source language: vi, en, zh, or ko
    #[arg(short, long, default_value = "vi")]
    source: String,

    /// Target language: vi, en, zh, or ko
    #[arg(short, long, default_value = "en")]
    target: String,

    /// Directory containing configs/*.toml (see configs/README.md)
    #[arg(long, default_value = "configs")]
    configs_dir: PathBuf,

    /// Directory containing model artifacts (see models/README.md)
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Directory containing glossary/eval data (see data/README.md)
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let config = Config::load(&cli.configs_dir).unwrap_or_else(|e| {
        tracing::error!("failed to load config: {e}");
        std::process::exit(1);
    });

    let mut pipeline = Pipeline::new(
        &config,
        &cli.models_dir,
        &cli.data_dir,
        &cli.source,
        &cli.target,
        &cli.input_wav,
        &cli.output_wav,
    )
    .unwrap_or_else(|e| {
        tracing::error!("failed to build pipeline: {e}");
        std::process::exit(1);
    });

    // TODO: loop on PTT press-release cycles once audio_io is backed by
    // live hardware capture instead of a fixed input file (see
    // src/stages/audio_io.rs) -- one file in, one file out for now.
    if let Err(e) = pipeline.run_once() {
        tracing::error!("pipeline run failed: {e}");
        std::process::exit(1);
    }
}
