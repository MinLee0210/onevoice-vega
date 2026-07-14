//! Loads `configs/language-pairs.toml` and `configs/latency-budget.toml`
//! (see configs/README.md). Kept as plain deserialize targets — the TOML
//! files are the source of truth, this module just gives them types.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::OneVoiceError;

#[derive(Debug, Deserialize)]
pub struct LanguagePair {
    pub source: String,
    pub target: String,
    pub asr_model: PathBuf,
    pub mt_model: PathBuf,
    pub tts_model: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct LanguagePairsConfig {
    #[serde(rename = "pair")]
    pub pairs: Vec<LanguagePair>,
}

impl LanguagePairsConfig {
    pub fn find(&self, source: &str, target: &str) -> Option<&LanguagePair> {
        self.pairs
            .iter()
            .find(|p| p.source == source && p.target == target)
    }
}

#[derive(Debug, Deserialize)]
pub struct LatencyStageBudget {
    pub vad_trim_ms: u32,
    pub asr_ms: u32,
    pub mt_ms: u32,
    pub tts_first_chunk_ms: u32,
    pub io_overhead_ms: u32,
}

#[derive(Debug, Deserialize)]
pub struct LatencyBudgetConfig {
    pub target_total_ms: u32,
    pub stretch_total_ms: u32,
    pub stage: LatencyStageBudget,
}

#[derive(Debug)]
pub struct Config {
    pub language_pairs: LanguagePairsConfig,
    pub latency_budget: LatencyBudgetConfig,
}

impl Config {
    pub fn load(configs_dir: &Path) -> Result<Self, OneVoiceError> {
        Ok(Self {
            language_pairs: load_toml(&configs_dir.join("language-pairs.toml"))?,
            latency_budget: load_toml(&configs_dir.join("latency-budget.toml"))?,
        })
    }
}

fn load_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, OneVoiceError> {
    let raw = std::fs::read_to_string(path).map_err(|source| OneVoiceError::ConfigIo {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(toml::from_str(&raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_language_pairs() {
        let raw = r#"
            [[pair]]
            source = "vi"
            target = "en"
            asr_model = "asr/vi/asr-vi.onnx"
            mt_model = "mt/vi-en/mt-vi-en.onnx"
            tts_model = "tts/en/tts-en.onnx"
        "#;
        let cfg: LanguagePairsConfig = toml::from_str(raw).unwrap();
        assert!(cfg.find("vi", "en").is_some());
        assert!(cfg.find("vi", "zh").is_none());
    }

    #[test]
    fn parses_latency_budget() {
        let raw = r#"
            target_total_ms = 1200
            stretch_total_ms = 800

            [stage]
            vad_trim_ms = 50
            asr_ms = 450
            mt_ms = 150
            tts_first_chunk_ms = 250
            io_overhead_ms = 100
        "#;
        let cfg: LatencyBudgetConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.target_total_ms, 1200);
        assert_eq!(cfg.stage.asr_ms, 450);
    }

    #[test]
    fn missing_file_is_config_io_error() {
        let err = Config::load(Path::new("/nonexistent/dir")).unwrap_err();
        assert!(matches!(err, OneVoiceError::ConfigIo { .. }));
    }
}
