//! Tauri command layer over the on-device `onevoice` engine.
//!
//! The front-end captures mic PCM, encodes a 48 kHz mono WAV and calls
//! `process_utterance`; Rust runs the real pipeline (denoise → VAD → ASR → MT
//! → TTS) via `onevoice::Engine` and returns the translated WAV + telemetry.
//! See docs/decisions/demo-frontend-design-2026-10.md.

use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;
use tauri::State;

use onevoice::config::Config;
use onevoice::engine::{audio_to_wav, CycleOutcome, Engine, EngineConfig};

/// Where the model/config/data tree lives. Overridable with `ONEVOICE_ROOT`;
/// otherwise the repo root relative to this crate (dev).
#[derive(Clone)]
struct AppPaths {
    configs: PathBuf,
    models: PathBuf,
    data: PathBuf,
}

impl AppPaths {
    fn resolve() -> Self {
        let root = std::env::var("ONEVOICE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
            });
        Self {
            configs: root.join("configs"),
            models: root.join("models"),
            data: root.join("data"),
        }
    }
}

/// Serialized per-stage timings for the latency HUD.
#[derive(Serialize, Clone)]
struct TelemetryDto {
    denoise_ms: f64,
    vad_ms: f64,
    asr_ms: f64,
    mt_ms: f64,
    tts_first_chunk_ms: f64,
    tts_total_ms: f64,
    to_first_audio_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct CycleResult {
    /// Translated audio as 16-bit PCM WAV bytes.
    wav: Vec<u8>,
    transcript: String,
    translation: String,
    source_lang: String,
    target_lang: String,
    confidence: f32,
    no_speech: bool,
    low_confidence: bool,
    telemetry: TelemetryDto,
}

fn to_result(outcome: CycleOutcome) -> CycleResult {
    let te = outcome.telemetry;
    CycleResult {
        wav: audio_to_wav(&outcome.audio),
        transcript: outcome.transcript,
        translation: outcome.translation,
        source_lang: outcome.source_lang,
        target_lang: outcome.target_lang,
        confidence: outcome.confidence,
        no_speech: outcome.no_speech,
        low_confidence: outcome.low_confidence,
        telemetry: TelemetryDto {
            denoise_ms: te.denoise_ms,
            vad_ms: te.vad_ms,
            asr_ms: te.asr_ms,
            mt_ms: te.mt_ms,
            tts_first_chunk_ms: te.tts_first_chunk_ms,
            tts_total_ms: te.tts_total_ms,
            to_first_audio_ms: te.to_first_audio_ms,
            total_ms: te.total_ms,
        },
    }
}

/// One resident engine per active pair (models are expensive to load).
#[derive(Default)]
struct EngineState {
    engine: Option<Engine>,
    pair: (String, String),
}

/// Is the model bundle present so the engine can load?
#[tauri::command]
fn ready(paths: State<AppPaths>) -> bool {
    paths.models.join("vad/model.onnx").exists()
        && paths.configs.join("language-pairs.toml").exists()
        && paths.models.join("asr/vi").exists()
}

/// Runs one translation cycle on captured WAV bytes (48 kHz mono). Loads or
/// swaps the engine when the language pair changes.
#[tauri::command]
fn process_utterance(
    pcm_wav: Vec<u8>,
    source: String,
    target: String,
    state: State<'_, Mutex<EngineState>>,
    paths: State<'_, AppPaths>,
) -> Result<CycleResult, String> {
    let mut st = state.lock().map_err(|e| e.to_string())?;

    if st.engine.is_none() || st.pair != (source.clone(), target.clone()) {
        let config = Config::load(&paths.configs).map_err(|e| e.to_string())?;
        let engine = Engine::new(EngineConfig {
            config: &config,
            models_dir: &paths.models,
            data_dir: &paths.data,
            source_lang: &source,
            target_lang: &target,
        })
        .map_err(|e| e.to_string())?;
        st.engine = Some(engine);
        st.pair = (source, target);
    }

    let engine = st.engine.as_mut().expect("engine set just above");
    let outcome = engine.process_wav(&pcm_wav).map_err(|e| e.to_string())?;
    Ok(to_result(outcome))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppPaths::resolve())
        .manage(Mutex::new(EngineState::default()))
        .invoke_handler(tauri::generate_handler![ready, process_utterance])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
