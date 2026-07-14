# OneVoice_2026

Participating in the Saigon AI Hub × Qualcomm "OneVoice AI Challenge":
an on-device (no cloud) speech translation device for noisy industrial
environments. See [CLAUDE.md](CLAUDE.md) for the full project spec
(hardware target, model stack, pipeline, performance targets, timeline).

## Layout
```
docs/       notes, TODOs, decisions, phase-submission drafts
notebook/   Python — model eval, fine-tuning, quantization, ONNX export
src/        Rust — on-device runtime
models/     exported .onnx / QNN artifacts (Python -> Rust handoff)
data/       glossary termbase, eval sets, raw corpora
configs/    language-pair configs, latency budgets
scripts/    glue scripts (AI Hub export, QNN convert, device deploy)
```
Each directory has its own `README.md`.

## Quickstart
```sh
# Python side (notebook/)
pip install -r notebook/requirements.txt

# Rust side (src/)
cd src && cargo build
```
