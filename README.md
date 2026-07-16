# OneVoice_2026

Participating in the Saigon AI Hub × Qualcomm "OneVoice AI Challenge":
an on-device (no cloud) push-to-talk speech translator for noisy
industrial environments (factories, construction, logistics,
hospitals), VI↔EN/ZH/KO. See [CLAUDE.md](CLAUDE.md) for the full
project spec (hardware target, model stack, pipeline, performance
targets, timeline). Currently Phase 2 (technical spec submission).

## Status

The pipeline runs end to end through translation on CPU (capture →
denoise → VAD trim → ASR → MT), each stage validated against a
Python/ONNX reference before being trusted:

| Stage | Status |
|---|---|
| Denoise (RNNoise) | real, tested |
| VAD (Silero) | real, tested |
| ASR (PhoWhisper VI / Whisper-small EN·ZH·KO) | real, tested |
| MT (Opus-MT vi-en) + glossary term enforcement | real, tested |
| TTS (Piper) | interface only — blocked on an open GPL license question, see [docs/decisions/piper-vi-voice-license-trap.md](docs/decisions/piper-vi-voice-license-trap.md) |
| Audio I/O | file-based (WAV in/out) — live mic/speaker (cpal) pending real hardware |
| Hexagon NPU (QNN) | untested — everything above runs on CPU so far |

vi-en is the only language pair implemented so far; the other 5
directions (en-vi, vi-zh, zh-vi, vi-ko, ko-vi) aren't ported yet. See
[docs/TODO.md](docs/TODO.md) for the live punch list and
[docs/decisions/](docs/decisions/) for the reasoning behind each of the
above (what was tried, what broke, why).

## Layout
```
docs/       notes, TODOs, decisions, phase-submission drafts
experiments/ Python — model eval, fine-tuning, quantization, ONNX export
src/        Rust — on-device runtime
models/     exported .onnx / QNN artifacts (Python -> Rust handoff, gitignored)
data/       glossary termbase, eval sets, raw corpora
configs/    language-pair configs, latency budgets
scripts/    setup_models.sh, run_pipeline.sh, run_notebook.sh
```
Each directory has its own `README.md`.

## Quickstart

```sh
# 1. Python side (experiments/) — use the `audio-lab` conda env
conda activate audio-lab
pip install -r experiments/requirements.txt

# 2. Populate models/ (HF Hub downloads for vad/asr/tts; runs the MT
#    export notebook for mt — needs the audio-lab env from step 1)
scripts/setup_models.sh

# 3. Build and run the real CLI against the committed eval fixture
scripts/run_pipeline.sh
```

`scripts/run_pipeline.sh` builds `--release` and runs `onevoice`
end to end, logging the transcription/translation and per-stage
latency, then exits cleanly at the TTS stage (see Status above). See
[scripts/README.md](scripts/README.md) for running individual stages,
your own WAV files, or the `experiments/` notebooks directly.
