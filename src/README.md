# src

On-device runtime, in Rust. Single crate, one module per pipeline stage
from CLAUDE.md — not a Cargo workspace. A workspace of many small crates
was tried first and cut for being more ceremony than the project (still
pre-prototype, Phase 2 spec) needs; revisit only if compile times or
real team-parallelism actually demand crate boundaries.

```
src/
├── Cargo.toml
└── src/
    ├── main.rs           entry point: load config, build Pipeline, run PTT loop
    ├── config.rs         Config/LanguagePair/LatencyBudget — deserializes
    │                     configs/*.toml (see configs/README.md)
    ├── error.rs          OneVoiceError (thiserror)
    ├── audio.rs          shared types passed between stages: AudioBuffer,
    │                     Utterance (ASR output), Translation (MT output)
    ├── pipeline.rs        Pipeline: owns one instance of each stage below
    │                     and runs them in order — the only place that
    │                     knows the pipeline order
    └── stages/
        ├── audio_io.rs    mic capture, PTT gating, playback (CPU)
        ├── denoise.rs      RNNoise / DeepFilterNet3 (CPU)
        ├── vad.rs          Silero VAD via `ort` (CPU)
        ├── asr.rs          ASR via `ort` + QNN execution provider (Hexagon NPU)
        ├── mt.rs           MT via `ort` + QNN execution provider (Hexagon NPU)
        ├── glossary.rs     industrial termbase lookup (the "≥95% glossary
        │                  accuracy" differentiator) — real, tested logic;
        │                  not a stub
        └── tts.rs          Piper TTS, streaming synthesis (CPU)
```

Every ML-backed stage (`denoise`, `vad`, `asr`, `mt`, `tts`) follows the
same shape: `load(model_path) -> Result<Self, OneVoiceError>`, then one
processing method that currently `todo!()`s — the interface is real, the
inference isn't wired up yet. `glossary` is the exception: it's plain
data lookup, so it's actually implemented and unit-tested now.

Build: `cd src && cargo build`.
Run: `cargo run` (will panic at the first `todo!()` until a stage is
implemented — expected while stages are stubs).
Test: `cargo test`.

Models are loaded from `models/<stage>/` — see `models/README.md` for
the artifact naming convention. `asr` and `mt` need the QNN execution
provider (`libQnnHtp.so`) available on-device; see CLAUDE.md → Hexagon /
QNN glossary and → Hardware toolchain.

Per CLAUDE.md's Rust risk note: smoke-test `ort` + QNN loading one
quantized model on real hardware early, before implementing the rest of
the stage bodies.
