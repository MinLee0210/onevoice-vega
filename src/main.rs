//! Entry point: parse CLI args, load config, build the pipeline for one
//! language pair, then run one PTT press-release cycle. Everything below
//! "wire it up" is delegated to `pipeline`; this file has no pipeline
//! logic itself.

// The latency budget config (configs/latency-budget.toml ->
// LatencyBudgetConfig) is loaded and checked at the end of each run.
// No extra dead_code warnings should trigger.

use std::path::PathBuf;

use clap::Parser;

use onevoice::broadcast::{Broadcast, BroadcastConfig};
use onevoice::config::Config;
use onevoice::pipeline::{Pipeline, PipelineConfig};

/// OneVoice: on-device VI<->EN/ZH/KO speech translation for noisy
/// industrial environments. See AGENTS.md for the full project spec.
///
/// Runs one push-to-talk cycle: reads `input_wav`, denoises, trims
/// silence, transcribes, translates, synthesizes, and writes
/// `output_wav`. Live mic/speaker capture isn't wired up yet
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

    /// Vega Air broadcast mode: comma-separated target languages (e.g.
    /// "en,zh,ko"). Transcribes once, then translates+synthesizes each
    /// language sequentially, writing `output_wav` as a template
    /// (`out.wav` -> `out.<lang>.wav`).
    #[arg(long, value_delimiter = ',')]
    broadcast: Vec<String>,
}

fn main() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("onevoice=info,ort=warn"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();

    let config = Config::load(&cli.configs_dir).unwrap_or_else(|e| {
        tracing::error!("failed to load config: {e}");
        std::process::exit(1);
    });

    if cli.broadcast.is_empty() {
        let pipeline_config = PipelineConfig {
            config: &config,
            models_dir: &cli.models_dir,
            data_dir: &cli.data_dir,
            source_lang: &cli.source,
            target_lang: &cli.target,
            input_wav_path: &cli.input_wav,
            output_wav_path: &cli.output_wav,
        };

        let mut pipeline = Pipeline::new(pipeline_config).unwrap_or_else(|e| {
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
    } else {
        let broadcast_config = BroadcastConfig {
            config: &config,
            models_dir: &cli.models_dir,
            data_dir: &cli.data_dir,
            source_lang: &cli.source,
            target_langs: &cli.broadcast,
            input_wav: &cli.input_wav,
            output_wav: &cli.output_wav,
        };

        let mut bc = Broadcast::new(broadcast_config).unwrap_or_else(|e| {
            tracing::error!("failed to build broadcast: {e}");
            std::process::exit(1);
        });

        if let Err(e) = bc.run_once() {
            tracing::error!("broadcast run failed: {e}");
            std::process::exit(1);
        }
    }
}
