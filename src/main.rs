//! Entry point: load config, build the pipeline for one language pair,
//! then loop on PTT press-release cycles. Everything below "wire it up"
//! is delegated to `pipeline`; this file has no pipeline logic itself.

// Every stage body is a `todo!()` stub pending real model inference, so
// plenty of the types/fields/methods around them aren't exercised by any
// live code path yet (only by unit tests). Remove this once stages stop
// being stubs and the normal dead_code lint is meaningful again.
#![allow(dead_code)]

mod audio;
mod config;
mod error;
mod pipeline;
mod stages;

use std::path::Path;

use config::Config;
use pipeline::Pipeline;

fn main() {
    tracing_subscriber::fmt::init();

    let config = Config::load(Path::new("configs")).unwrap_or_else(|e| {
        tracing::error!("failed to load config: {e}");
        std::process::exit(1);
    });

    // TODO: source/target language selected via PTT UI (e.g. button combo
    // or paired device state), not hardcoded — placeholder for now.
    let pipeline = Pipeline::new(
        &config,
        Path::new("models"),
        Path::new("data"),
        "vi",
        "en",
    )
    .unwrap_or_else(|e| {
        tracing::error!("failed to build pipeline: {e}");
        std::process::exit(1);
    });

    loop {
        if let Err(e) = pipeline.run_once() {
            tracing::error!("pipeline run failed: {e}");
        }
    }
}
