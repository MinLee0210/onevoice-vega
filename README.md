# OneVoice_2026

Participating in the Saigon AI Hub × Qualcomm "OneVoice AI Challenge":
an on-device (no cloud) speech translation device for noisy industrial
environments. See [CLAUDE.md](CLAUDE.md) for the full project spec
(hardware target, model stack, pipeline, performance targets, timeline).

## Layout
```
docs/       notes, TODOs, decisions, phase-submission drafts
experiments/ Python — model eval, fine-tuning, quantization, ONNX export
src/        Rust — on-device runtime
models/     exported .onnx / QNN artifacts (Python -> Rust handoff)
data/       glossary termbase, eval sets, raw corpora
configs/    language-pair configs, latency budgets
scripts/    glue scripts (AI Hub export, QNN convert, device deploy)
```
Each directory has its own `README.md`.

## Quickstart
```sh
# Python side (experiments/) — use the `audio-lab` conda env, see experiments/README.md
conda activate audio-lab
pip install -r experiments/requirements.txt

# Rust side (src/)
cargo build
```
