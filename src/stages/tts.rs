//! Text-to-speech. The device is bidirectional, so it synthesizes every
//! target language: VI, EN, ZH (Mandarin) via Piper (VITS/ONNX, espeak-ng
//! front-end) and KO via Meta MMS-TTS (VITS/ONNX, uroman romanization
//! front-end — Piper has no Korean voice). All four validated end-to-end
//! (synthesis + round-trip TTS→ASR intelligibility) in
//! experiments/04_tts.ipynb, then again here against the real Rust
//! inference path before trusting it — see the tests below.
//!
//! ## Backend dispatch
//!
//! Which backend a `models/tts/<lang>/` directory uses is detected from
//! its *contents*, not the language code: a Piper voice has a
//! `<name>.onnx.json` config sibling next to its `.onnx`; MMS doesn't (it
//! ships `vocab.json` + tokenizer config instead). This means adding a
//! 5th language later needs no code change here as long as it's one of
//! these two families.
//!
//! ## Piper (vi/en/zh): the `piper-rs` crate
//!
//! `piper-rs` (<https://github.com/thewh1teagle/piper-rs>, MIT) does the
//! full job — espeak-ng phonemization (via its `espeak-rs` /
//! `espeak-rs-sys` dependencies) and VITS inference via `ort`, the same
//! crate this project already depends on. One real build gotcha: the
//! default `compile-espeak-intonations` feature fails on the crates.io
//! package (`Error processing file '.../phsource/intonation': No such
//! file or directory` — a broken relative path in the vendored
//! CMakeLists, not an actually-missing file; the source file is present
//! in the package). Building with `default-features = false` (see
//! Cargo.toml) skips it and compiles cleanly — the cost is losing
//! synthesized prosody *intonation curves*, not phonemization itself.
//!
//! That fix has a side effect, though: without `compile-espeak-intonations`,
//! the crate's CMake build also doesn't generate espeak-ng's core
//! `phontab`/`phondata`/`phonindex` runtime data files (they're compiled
//! from `phsource/` by the same disabled step) — only the plain `lang`/
//! `voices` files ship as-is. Those three files are required at runtime
//! (espeak-ng aborts on `espeak_Initialize` without them) and aren't
//! produced by our build at all now, so they're sourced once from a real
//! espeak-ng install (the `piper-tts` PyPI package bundles a complete
//! `espeak-ng-data/`, already a project dependency) and shipped as a
//! model-like artifact at `models/tts/espeak-ng-data/` (see
//! `scripts/setup_models.sh`) — pure data tables, not compiled code, so
//! identical across OS/arch. `TtsEngine::load` points `espeak-rs` at it
//! via the `PIPER_ESPEAKNG_DATA_DIRECTORY` env var before the first
//! phonemization call (espeak-ng initializes lazily, once per process).
//!
//! ## MMS (ko): hand-rolled — no crate covers this path
//!
//! No Rust crate wraps MMS-TTS, so this is a direct port of
//! `transformers.VitsTokenizer.prepare_for_tokenization` / `_tokenize`
//! (the exact algorithm this model's config selects: `normalize=true`,
//! `is_uroman=true`, `phonemize=false`, `add_blank=true`) plus raw `ort`
//! inference against `models/tts/ko/mms_tts_kor.onnx`:
//! 1. Lowercase, then romanize via the `uroman` crate — a from-scratch
//!    Rust reimplementation of the same `uroman` Python package the HF
//!    tokenizer uses; verified to produce byte-identical romanized
//!    output (see `mms_tokenize` docs and the tests below).
//! 2. Drop any character not in the model's `vocab.json`, trim.
//! 3. Intersperse the blank token (id 0) between every remaining
//!    character (and at both ends) — VITS/MMS's char-level tokenization.
//!
//! Ported and verified to produce byte-identical `input_ids` to the
//! Python tokenizer on both of `experiments/04_tts.ipynb`'s KO probe
//! sentences before being trusted here.
//!
//! Runs on CPU; QNN/Hexagon is future work like the other ONNX stages.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use uroman::{rom_format, Uroman};

use crate::audio::{AudioBuffer, Translation};
use crate::error::OneVoiceError;
use crate::stages::session;

/// Env var `espeak-rs` (pulled in by `piper-rs`) reads to find its
/// `espeak-ng-data` directory — see that crate's `locate_espeak_data()`.
/// Must be set before the first phonemization call; espeak-ng
/// initializes lazily, once per process, guarded by a `OnceLock` inside
/// `espeak-rs`.
const ESPEAK_DATA_ENV: &str = "PIPER_ESPEAKNG_DATA_DIRECTORY";
/// MMS-TTS's fixed output rate (`VitsModel.config.sampling_rate` — see
/// models/tts/ko/MANIFEST.md). Piper voices carry their own sample rate
/// in their `.onnx.json` and don't need this constant.
const MMS_SAMPLE_RATE_HZ: u32 = 16_000;
/// MMS/VITS's char-level blank/separator token id, interspersed between
/// every real character (see `mms_tokenize`). Always 0 for this model
/// family, independent of what character its vocab happens to assign to
/// id 0 (see the module docs' MMS section).
const MMS_BLANK_TOKEN_ID: i64 = 0;

enum Backend {
    /// VI/EN/ZH: Piper VITS + espeak-ng phonemization, handled internally
    /// by `piper_rs::Piper`.
    Piper(piper_rs::Piper),
    /// KO: Meta MMS-TTS VITS, hand-rolled front-end (see module docs).
    Mms {
        session: Session,
        vocab: HashMap<char, i64>,
        uroman: Uroman,
    },
}

pub struct TtsEngine {
    backend: Backend,
    /// Piper VITS length scale (>1 = slower, more intelligible speech).
    /// `None` = the voice's default. Used by Vega Air for broadcast-tuned
    /// synthesis (outdoor SPL). Ignored by the MMS (KO) backend.
    length_scale: Option<f32>,
}

impl TtsEngine {
    /// `model_dir` is `models/tts/<target-lang>/` (see config's
    /// `tts_model_dir`). `espeak_data_dir` must be the *parent* of an
    /// `espeak-ng-data/` directory (i.e. pass `models_dir.join("tts")`
    /// for the layout `models/tts/espeak-ng-data/`) — ignored for the
    /// MMS backend, required for Piper.
    pub fn load(
        model_dir: impl Into<PathBuf>,
        espeak_data_dir: &Path,
    ) -> Result<Self, OneVoiceError> {
        let model_dir = model_dir.into();
        let onnx_path = find_onnx_file(&model_dir)?;
        let piper_config_path = PathBuf::from(format!("{}.json", onnx_path.display()));

        let backend = if piper_config_path.exists() {
            std::env::set_var(ESPEAK_DATA_ENV, espeak_data_dir);
            let piper = piper_rs::Piper::new(&onnx_path, &piper_config_path).map_err(|source| {
                OneVoiceError::Tts(format!(
                    "failed to load Piper voice {}: {source}",
                    onnx_path.display()
                ))
            })?;
            Backend::Piper(piper)
        } else {
            let vocab_path = model_dir.join("vocab.json");
            let vocab_json = std::fs::read_to_string(&vocab_path).map_err(|source| {
                OneVoiceError::Tts(format!("failed to read {}: {source}", vocab_path.display()))
            })?;
            let vocab = parse_char_vocab(&vocab_json).map_err(|source| {
                OneVoiceError::Tts(format!(
                    "failed to parse {}: {source}",
                    vocab_path.display()
                ))
            })?;

            let session = session::commit_from_file(&onnx_path).map_err(|source| {
                OneVoiceError::Tts(format!("failed to load {}: {source}", onnx_path.display()))
            })?;

            Backend::Mms {
                session,
                vocab,
                uroman: Uroman::new(),
            }
        };

        Ok(Self {
            backend,
            length_scale: None,
        })
    }

    /// Sets a Piper length scale (>1 = slower, more intelligible) — the
    /// broadcast-tuned rate for Vega Air. No effect on the MMS (KO) backend.
    pub fn set_length_scale(&mut self, scale: f32) {
        self.length_scale = Some(scale);
    }

    /// Synthesizes sentence-by-sentence and hands each chunk to `on_chunk` as
    /// soon as it is ready (streaming playback: the caller can start writing
    /// audio before the whole utterance is synthesized). Returns the
    /// time-to-first-chunk in milliseconds.
    pub fn synthesize_streaming(
        &mut self,
        translation: &Translation,
        on_chunk: &mut dyn FnMut(&AudioBuffer) -> Result<(), OneVoiceError>,
    ) -> Result<f64, OneVoiceError> {
        tracing::debug!(lang = %translation.lang, "running TTS");

        let sentences = split_sentences(&translation.text);
        if sentences.is_empty() {
            return Ok(0.0);
        }

        let mut first_chunk_ms = 0.0;
        let length_scale = self.length_scale;

        match &mut self.backend {
            Backend::Piper(piper) => {
                for (i, sentence) in sentences.iter().enumerate() {
                    let t = std::time::Instant::now();
                    let (samples, rate) = piper
                        .create(sentence, false, None, length_scale, None, None)
                        .map_err(|source| {
                            OneVoiceError::Tts(format!("Piper synthesis failed: {source}"))
                        })?;
                    if i == 0 {
                        first_chunk_ms = t.elapsed().as_secs_f64() * 1000.0;
                    }
                    on_chunk(&AudioBuffer::new(samples, rate))?;
                }
            }
            Backend::Mms {
                session,
                vocab,
                uroman,
            } => {
                for (i, sentence) in sentences.iter().enumerate() {
                    let t = std::time::Instant::now();
                    let ids = mms_tokenize(sentence, vocab, uroman);
                    let seq_len = ids.len();
                    let input = Tensor::from_array(([1usize, seq_len], ids)).map_err(|source| {
                        OneVoiceError::Tts(format!("failed to build input tensor: {source}"))
                    })?;

                    let outputs =
                        session
                            .run(ort::inputs!["input_ids" => input])
                            .map_err(|source| {
                                OneVoiceError::Tts(format!("MMS inference failed: {source}"))
                            })?;
                    let (_shape, waveform) = outputs["waveform"]
                        .try_extract_tensor::<f32>()
                        .map_err(|source| {
                            OneVoiceError::Tts(format!("failed to read waveform output: {source}"))
                        })?;
                    if i == 0 {
                        first_chunk_ms = t.elapsed().as_secs_f64() * 1000.0;
                    }
                    on_chunk(&AudioBuffer::new(waveform.to_vec(), MMS_SAMPLE_RATE_HZ))?;
                }
            }
        }

        Ok(first_chunk_ms)
    }

    /// Batch wrapper: synthesizes the whole utterance into one buffer. Used
    /// by the tests; the pipeline streams via `synthesize_streaming`.
    #[allow(dead_code)]
    pub fn synthesize(
        &mut self,
        translation: &Translation,
    ) -> Result<(AudioBuffer, f64), OneVoiceError> {
        let mut all_samples = Vec::new();
        let mut sample_rate_hz = 0u32;
        let first_chunk_ms = self.synthesize_streaming(translation, &mut |chunk| {
            if sample_rate_hz == 0 {
                sample_rate_hz = chunk.sample_rate_hz;
            }
            all_samples.extend_from_slice(&chunk.samples);
            Ok(())
        })?;
        if sample_rate_hz == 0 {
            sample_rate_hz = 16_000;
        }
        Ok((
            AudioBuffer::new(all_samples, sample_rate_hz),
            first_chunk_ms,
        ))
    }
}

/// Splits input text into sentences by looking for punctuation marks
/// (. ? ! ; \n and their full-width Chinese equivalents).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if c == '.'
            || c == '?'
            || c == '!'
            || c == '\n'
            || c == ';'
            || c == '。'
            || c == '？'
            || c == '！'
            || c == '；'
        {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    if sentences.is_empty() && !text.is_empty() {
        sentences.push(text.to_string());
    }
    sentences
}

/// Finds the single `.onnx` file directly inside `model_dir`. Every
/// `models/tts/<lang>/` directory has exactly one (see MANIFEST.md in
/// each).
fn find_onnx_file(model_dir: &Path) -> Result<PathBuf, OneVoiceError> {
    std::fs::read_dir(model_dir)
        .map_err(|source| {
            OneVoiceError::Tts(format!("failed to read {}: {source}", model_dir.display()))
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "onnx"))
        .ok_or_else(|| {
            OneVoiceError::Tts(format!("no .onnx file found in {}", model_dir.display()))
        })
}

/// Parses a VITS/MMS `vocab.json` (`{"a": 7, "b": 13, ...}`) into a
/// single-character lookup table, silently skipping any multi-character
/// entries (e.g. `"<unk>"`) — this tokenizer only ever needs to look up
/// individual characters from already-romanized text.
fn parse_char_vocab(vocab_json: &str) -> Result<HashMap<char, i64>, serde_json::Error> {
    let raw: HashMap<String, i64> = serde_json::from_str(vocab_json)?;
    Ok(raw
        .into_iter()
        .filter_map(|(token, id)| {
            let mut chars = token.chars();
            let c = chars.next()?;
            if chars.next().is_some() {
                return None; // multi-char special token, e.g. "<unk>" -- not looked up by char
            }
            Some((c, id))
        })
        .collect())
}

/// Ports `transformers.VitsTokenizer.prepare_for_tokenization` +
/// `_tokenize` for this model's config (`normalize=true`,
/// `is_uroman=true`, `phonemize=false`, `add_blank=true`): lowercase,
/// romanize, drop any character the model's vocab doesn't have, trim,
/// then intersperse the blank token between every character (and at
/// both ends) — e.g. `"ab"` with blank id 0 becomes `[0, a, 0, b, 0]`.
/// Any character dropped by the vocab filter (typically punctuation
/// the model was never trained on) is silently omitted, matching the
/// Python tokenizer's own behavior exactly, not approximated.
fn mms_tokenize(text: &str, vocab: &HashMap<char, i64>, uroman: &Uroman) -> Vec<i64> {
    let lower = text.to_lowercase();
    let romanized = uroman
        .romanize_string::<rom_format::Str>(&lower, None)
        .to_string();
    let filtered: String = romanized
        .chars()
        .filter(|c| vocab.contains_key(c))
        .collect();
    let filtered = filtered.trim();

    let mut ids = Vec::with_capacity(filtered.chars().count() * 2 + 1);
    ids.push(MMS_BLANK_TOKEN_ID);
    for c in filtered.chars() {
        ids.push(vocab[&c]);
        ids.push(MMS_BLANK_TOKEN_ID);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ko_vocab() -> HashMap<char, i64> {
        parse_char_vocab(
            r#"{" ": 21, "'": 15, "-": 11, "_": 1, "a": 7, "b": 13, "c": 14, "d": 18, "e": 22,
                "g": 19, "h": 8, "i": 9, "j": 10, "k": 12, "l": 17, "m": 23, "n": 16, "o": 5,
                "p": 24, "r": 20, "s": 4, "t": 2, "u": 0, "w": 3, "y": 6}"#,
        )
        .unwrap()
    }

    /// Exact `input_ids` reference captured from the real
    /// `facebook/mms-tts-kor` HF tokenizer (`tok(text)["input_ids"]`) in
    /// experiments/04_tts.ipynb's environment — not hand-derived, so this
    /// catches any divergence between `uroman`-the-crate and
    /// `uroman`-the-Python-package, not just a logic bug in this port.
    #[test]
    fn mms_tokenize_matches_the_python_tokenizer_exactly() {
        let vocab = ko_vocab();
        let uroman = Uroman::new();

        let ids = mms_tokenize("경보가 울리면 즉시 기계를 멈추세요.", &vocab, &uroman);
        let expected = vec![
            0, 19, 0, 6, 0, 22, 0, 5, 0, 16, 0, 19, 0, 13, 0, 5, 0, 19, 0, 7, 0, 21, 0, 0, 0, 17,
            0, 20, 0, 9, 0, 23, 0, 6, 0, 22, 0, 5, 0, 16, 0, 21, 0, 10, 0, 22, 0, 0, 0, 19, 0, 4,
            0, 9, 0, 21, 0, 19, 0, 9, 0, 19, 0, 6, 0, 22, 0, 20, 0, 22, 0, 0, 0, 17, 0, 21, 0, 23,
            0, 22, 0, 5, 0, 23, 0, 14, 0, 0, 0, 4, 0, 22, 0, 6, 0, 5, 0,
        ];
        assert_eq!(ids, expected);

        let ids2 = mms_tokenize("안녕하세요, 잘 지내세요?", &vocab, &uroman);
        let expected2 = vec![
            0, 7, 0, 16, 0, 16, 0, 6, 0, 22, 0, 5, 0, 16, 0, 19, 0, 8, 0, 7, 0, 4, 0, 22, 0, 6, 0,
            5, 0, 21, 0, 10, 0, 7, 0, 17, 0, 21, 0, 10, 0, 9, 0, 16, 0, 7, 0, 22, 0, 4, 0, 22, 0,
            6, 0, 5, 0,
        ];
        assert_eq!(ids2, expected2);
    }

    #[test]
    fn parse_char_vocab_skips_multi_char_tokens() {
        let vocab = parse_char_vocab(r#"{"a": 0, "<unk>": 25}"#).unwrap();
        assert_eq!(vocab.get(&'a'), Some(&0));
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn find_onnx_file_locates_the_only_onnx_in_a_directory() {
        let dir =
            std::env::temp_dir().join(format!("onevoice-tts-find-onnx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("voice.onnx"), b"").unwrap();
        std::fs::write(dir.join("voice.onnx.json"), b"{}").unwrap();

        let found = find_onnx_file(&dir).expect("should find the .onnx file");
        assert_eq!(found, dir.join("voice.onnx"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Requires `models/tts/vi/` and `models/tts/espeak-ng-data/` on disk
    /// (gitignored; populate via `scripts/setup_models.sh tts`) and must
    /// run from the repo root, e.g. `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires models/tts/vi/ and models/tts/espeak-ng-data/, see scripts/setup_models.sh"]
    fn synthesizes_vi_speech_via_piper() {
        let mut engine = TtsEngine::load("models/tts/vi", Path::new("models/tts"))
            .expect("failed to load TtsEngine");
        let translation = Translation {
            text: "Dừng máy ngay lập tức nếu nghe thấy báo động.".to_string(),
            lang: "vi".to_string(),
        };

        let (audio, _first_chunk_ms) = engine.synthesize(&translation).expect("synthesize failed");

        assert_eq!(audio.sample_rate_hz, 22_050);
        assert!(!audio.samples.is_empty());
        // Real speech, several seconds -- not silence, not a truncated stub.
        assert!(
            audio.duration_ms() > 1000.0,
            "duration was {}ms",
            audio.duration_ms()
        );
        assert!(
            audio.samples.iter().any(|&s| s.abs() > 0.01),
            "audio is silent -- likely a phonemization/inference failure"
        );
    }

    /// Requires `models/tts/ko/` on disk (gitignored; populate via
    /// `scripts/setup_models.sh tts`) and must run from the repo root.
    /// MMS-TTS's ONNX export was made deterministic (`noise_scale=0`,
    /// see models/tts/ko/MANIFEST.md), so unlike Piper's stochastic
    /// VITS, this checks an *exact* sample count against the real model
    /// -- validated once against experiments/04_tts.ipynb section 5.
    #[test]
    #[ignore = "requires models/tts/ko/, see scripts/setup_models.sh"]
    fn synthesizes_ko_speech_via_mms_matching_expected_sample_count() {
        let mut engine = TtsEngine::load("models/tts/ko", Path::new("models/tts"))
            .expect("failed to load TtsEngine");
        let translation = Translation {
            text: "경보가 울리면 즉시 기계를 멈추세요.".to_string(),
            lang: "ko".to_string(),
        };

        let (audio, _first_chunk_ms) = engine.synthesize(&translation).expect("synthesize failed");

        assert_eq!(audio.sample_rate_hz, 16_000);
        assert_eq!(audio.samples.len(), 57_856);
    }
}
