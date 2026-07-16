# src

On-device runtime, in Rust. Single crate, one module per pipeline stage
from CLAUDE.md — not a Cargo workspace. A workspace of many small crates
was tried first and cut for being more ceremony than the project (still
pre-prototype, Phase 2 spec) needs; revisit only if compile times or
real team-parallelism actually demand crate boundaries.

```
Cargo.toml
src/
├── main.rs           entry point: load config, build Pipeline, run one
│                     PTT cycle from a WAV file (see "Run" below)
├── config.rs         Config/LanguagePair/LatencyBudget — deserializes
│                     configs/*.toml (see configs/README.md)
├── error.rs          OneVoiceError (thiserror)
├── audio.rs          shared types passed between stages: AudioBuffer,
│                     Utterance (ASR output), Translation (MT output)
├── pipeline.rs        Pipeline: owns one instance of each stage below
│                     and runs them in order — the only place that
│                     knows the pipeline order
└── stages/
    ├── audio_io.rs    reads/writes WAV files (stands in for live mic
    │                  capture/playback until real hardware exists)
    ├── denoise.rs      RNNoise via `nnnoiseless` (CPU) — real, tested
    ├── vad.rs          Silero VAD via `ort` (CPU) — real, tested
    ├── asr.rs          Whisper-family ASR via `ort` (CPU today, Hexagon
    │                  NPU via QNN once on real hardware) — real, tested
    ├── mt.rs           Opus-MT via `ort` (same CPU/QNN split as asr) —
    │                  real, tested, includes glossary-term decoder-bias
    │                  enforcement (see glossary.rs and
    │                  docs/decisions/glossary-runtime-enforcement-2026-07.md)
    ├── glossary.rs     industrial termbase lookup (the "≥95% glossary
    │                  accuracy" differentiator) — real, tested, and
    │                  consulted by `mt`'s decode (see mt.rs below)
    └── tts.rs          Piper TTS — interface only; deliberately not
                       implemented, not just unfinished: Piper's
                       Vietnamese phonemizer is GPL-3.0 and needs a
                       legal read before it goes into this binary, see
                       docs/decisions/piper-vi-voice-license-trap.md
```

Every ML-backed stage (`denoise`, `vad`, `asr`, `mt`) follows the same
shape: `load(model_path) -> Result<Self, OneVoiceError>`, then one
processing method — all real, all validated against a Python reference
before being trusted (see docs/decisions/onnx-runtime-merged-decoder-gotchas.md,
docs/decisions/mel-spec-crate-whisper-compat-bugs.md, and
docs/decisions/glossary-runtime-enforcement-2026-07.md for the real bugs
that validation caught in each). `glossary` is wired into `mt`'s decode
(decoder-logit-bias, vi-en only so far). `tts` is the one genuine stub, and `audio_io` is
file-based rather than live-hardware-based — see the tree above for why.

## Build / test / run

```sh
cargo build
cargo test                 # always-on unit tests
cargo test -- --ignored    # +integration tests against real ONNX
                            # models in models/ (gitignored, regenerate
                            # via scripts/setup_models.sh) and WAV
                            # fixtures in data/eval/ (committed, small)

cargo run -- --help
```

For actually running the built binary as a CLI app, use
`scripts/run_pipeline.sh` (see `scripts/README.md`) rather than typing
`cargo run --release -- ...` by hand — it builds `--release`, checks
`models/` is populated first, and defaults to the committed eval
fixture:

```sh
scripts/setup_models.sh     # first time only -- populates models/
scripts/run_pipeline.sh     # builds --release, runs against
                             # data/eval/ptt_sample_vi_48k.wav, RUST_LOG=info
scripts/run_pipeline.sh in.wav out.wav -- --source vi --target en
```

This runs the real `Pipeline` end to end: capture -> denoise -> VAD
trim -> ASR -> MT, logs the transcription/translation and per-stage
timing at INFO level, then exits with a clear error at the TTS stage
(see `stages/tts.rs` above). Input WAV must be mono, 48kHz (RNNoise's
fixed input rate).

Models are loaded from `models/<stage>/` — see `models/README.md` for
what's there and `scripts/setup_models.sh` for how to (re)populate it.
`asr` and `mt` need the QNN execution provider (`libQnnHtp.so`)
available on-device to run on the Hexagon NPU; see CLAUDE.md → Hexagon
/ QNN glossary and → Hardware toolchain. Nothing in this crate has been
run against QNN yet — only CPU, so far.
