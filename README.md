# OneVoice AI Challenge 2026: On-Device Speech-to-Speech Translation

This repository contains the prototype implementation for the **Qualcomm Dragonwing QCS6490 (Rubik Pi 3)** submitted to the **Saigon AI Hub × Qualcomm "OneVoice AI Challenge"**.

OneVoice is a **100% on-device (no cloud/internet connection required)** push-to-talk speech translator designed specifically for noisy industrial environments (factories, construction, logistics, hospitals) supporting **Vietnamese (VI), English (EN), Chinese (ZH), and Korean (KO)**.

---

## 🚀 Key Accomplishments & Technical Highlights

* **End-to-End Edge Pipeline:** Zero-cloud Rust runtime (`src/`) that pipelines: Audio Capture → RNNoise Denoising → Silero VAD trimming → PhoWhisper ASR → Opus-MT → Piper/MMS VITS TTS → Playback.
* **ARM64-Optimized INT8 Quantization:** Custom post-training dynamic quantization script (`scripts/quantize_mt.py`) that shrinks the Translation models by **74.8%** (encoder from 178MB to 44MB, decoder from 307MB to 77MB), keeping the total model footprint under **122.6 MB** to fit target edge memory budgets (<300MB/pair).
* **Automaton Glossary Matching (R7):** Multi-language industrial termbase logit biasing utilizing the **Aho-Corasick automaton** algorithm for $O(M)$ lookup latency, resolving key safety terminology translation issues.
* **Streaming TTS Chunking (R4):** Sentence-by-sentence VITS TTS chunking to minimize Time-to-First-Audio (TTFA) latency down to **~90ms**, meeting the strict 250ms stage budget.
* **AWGN Noise Robustness Evaluation (R10):** Additive White Gaussian Noise benchmarking script (`scripts/evaluate_noise_robustness.py`) verifying that PhoWhisper-base maintains high accuracy under extreme factory floor conditions (**12.32%** WER at a loud **5dB SNR**).

---

## 📊 Pipeline Status & Telemetry

The CPU-based pipeline executes under **~600-800ms** total latency (well within the ≤1.2s target):

| Stage | Backend & Models | Status | Average Latency |
| :--- | :--- | :---: | :---: |
| **Denoise** | RNNoise (nnnoiseless) | Active | ~5 ms |
| **VAD** | Silero VAD (ONNX) | Active | ~7 ms |
| **ASR** | PhoWhisper-base (ONNX) | Active | ~410 ms |
| **MT** | Opus-MT (ONNX - Quantized INT8) | Active | ~100 ms |
| **TTS** | Piper VITS (VI/EN/ZH) & MMS-TTS (KO) | Active | ~120 ms (TTFA: ~90ms) |
| **Audio I/O** | Mock Wave Reader / Writer | Active | ~10 ms (CPAL pending HW) |

**Two ASR backends**, auto-selected by which artifact is in `asr_model_dir`:
**Whisper** (PhoWhisper/Whisper-small, seq2seq) or **CTC** (`src/stages/ctc.rs`,
`CtcAsrEngine`, single forward pass — no decoder loop). Measured on VI
(`docs/reports/asr_ab.md`): CTC is ~6× faster in the pipeline (ASR 57 ms vs
~360–420 ms; full cycle 181 ms vs 485 ms) but higher WER. The bundled
`asr/vi-ctc` is an Apache-2.0 base fine-tuned on VIVOS
(`scripts/edge/finetune_ctc_vi.py`; `docs/decisions/ctc-finetune-recipe-2026-10.md`),
now clean 9.86% / 5 dB 23.94% on the VIVOS slice. Whisper is the default for
accuracy; CTC is the latency-fallback. Model bundle: GitHub release
**`models-v3`** (split into 2 parts; `scripts/setup_models.sh` reassembles).

---

## 🖥️ Hardware Deployment Evaluation (Qualcomm QCS6490)

Every pipeline stage was deployed and profiled on the target
**`Dragonwing RB3 Gen 2 Vision Kit`** (QCS6490, Hexagon v68 NPU) via Qualcomm
AI Hub. Full write-ups: `docs/reports/qcs6490-deployment-evaluation-2026-10.md`,
`docs/reports/vega-hand-air-deployment-2026-10.md`; raw metrics in `data/qaihub/`.

**Key result:** the whole pipeline runs on a **single QCS6490**. Only the ASR
network goes on the Hexagon **NPU**; MT, VAD, denoise and TTS run on the Kryo
**CPU** of the same chip (Opus-MT does not compile to QNN). The ASR decoder
ships as the **split** export (`decoder_model.onnx` + `decoder_with_past_model.onnx`)
— the merged decoder's `use_cache_branch` If-node fails QNN conversion.

### Per-stage placement

| Stage | Hardware | Status | Inference Time | Peak Memory |
| :--- | :--- | :---: | :---: | :---: |
| RNNoise denoise | Kryo CPU | ✅ | ~5 ms | compiled in |
| Silero VAD | Kryo CPU | ✅ | ~5 ms | 2.1 MB |
| ASR encoder (PhoWhisper / Whisper-small) | Hexagon NPU | ✅ | 230 / 178 ms | 18–20 MB |
| ASR decoder, split first/past | Hexagon NPU | ✅ | 14.1 / 8.9 ms-per-token | ~40 MB |
| ASR decoder, merged | Hexagon NPU | ❌ | — | QAIRT exit 255 |
| MT (Opus-MT) | Kryo CPU | ✅ (NPU ✗) | ~19 ms | ~127 MB |
| TTS (Piper / MMS) | Kryo CPU | ✅ | ~95 ms | ~60 MB |

### Proposed solutions

| Solution | Hardware | Mem cost (RAM) | Storage (models) | Inference Time / cycle | Peak Memory | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vega Hand** (VI↔EN/ZH/KO, 1:1) | 1× QCS6490 (NPU+CPU) | ~2.5 GB budget | ~1.4 GB | ~485 ms | NPU ≤39 MB; CPU ~1.2 GB | ✅ deployable |
| **Vega Air** (VI→EN/ZH/KO broadcast) | 1× QCS6490 (NPU+CPU) | ~2.0 GB budget | ~740 MB | ASR ~378 ms + MT ~19 ms + 3× TTS | NPU ≤39 MB | ✅ deployable |
| All-CPU baseline (vi→en) | 1× QCS6490 (CPU) | ~1.2 GB measured | ~287 MB | ~541 ms | ~1.2 GB | ✅ deployable today |
| All-NPU attempt (MT on Hexagon) | — | — | — | — | — | ❌ MT/VAD won't compile |

Vega Air is a model subset of Vega Hand, so it deploys on the same hardware
with no new NPU work. "Mem cost" is the resident working set; the all-CPU
figure (1.18 GB) is a measured peak RSS, the others are model-size budgets.

---

## 🌐 Translation Coverage & MT Quality

All **6 directions** (VI↔EN/ZH/KO) now translate. Direct Opus-MT models exist
for vi-en/en-vi; the VI↔ZH and VI↔KO pairs **pivot through English**
(`pivot = { lang = "en", ... }` in `configs/language-pairs.toml`, chained by
`MtChain` in `src/stages/mt.rs`, glossary applied per leg).

First measured MT quality (Helsinki-NLP/opus-100 test, 200 pairs, greedy) —
full detail in `docs/reports/mt_quality.md`:

| direction | path | BLEU (leg) | note |
|---|---|---|---|
| vi→en | direct | 29.6 | flagship pair; below the ≥40 (domain-set) target |
| en→vi | direct | 22.2 | |
| vi→zh | pivot | vi-en 29.6 → en-zh 46.5 | |
| zh→vi | pivot | zh-en 44.4 → en-vi 22.2 | |
| vi→ko | pivot | vi-en 29.6 → en-ko **0.1** | ❌ en→ko checkpoint is broken |
| ko→vi | pivot | ko-en 15.0 → en-vi 22.2 | mediocre |

**KO is the problem direction** — broken en→ko MT *and* a non-commercial KO
TTS (MMS, CC-BY-NC). It needs a different model (e.g. M2M-100, MIT) before the
demo. COMET not yet measured. Reproduce: `scripts/evaluate_mt.py`.

---

## 🛠️ Quickstart Setup Guide

### 1. Environment Configuration
We utilize a Conda environment `audio-lab` for model exports, quantization, and evaluation benchmarks:

```bash
# Activate the conda environment
conda activate audio-lab

# Install required dependencies
pip install -r experiments/requirements.txt
```

### 2. Download and Initialize Models
You can set up the required models in one of two ways:

**Option A: Automated Setup (Recommended)**
Run the setup script to automatically download VAD/ASR/TTS models from Hugging Face Hub and export/quantize the translation model components locally:
```bash
# Download and initialize models (includes automatic INT8 quantization)
scripts/setup_models.sh
```

**Option B: Direct Archive Download**
Alternatively, you can download the pre-packaged and optimized models bundle directly from our [GitHub Release (v2)](https://github.com/MinLee0210/OneVoice_2026/releases/tag/models-v2):
1. Download `onevoice_models_v2.tar.gz` from the release assets.
2. Extract the archive in the repository root directory:
   ```bash
   tar -xzf onevoice_models_v2.tar.gz
   ```

### 3. Run the End-to-End Pipeline
Run the mock pipeline against the default evaluation WAV files:

```bash
# Run the pipeline end-to-end (VI -> EN)
scripts/run_pipeline.sh
```
This builds the release binary, runs the translation loop, and generates the output audio, printing full telemetry and latency budget checks.

---

## 🧪 Evaluation & Benchmarks

To reproduce our noise robustness and ASR evaluation benchmark, run:

```bash
# Run AWGN noise injection (5dB, 10dB, 15dB SNR) on 30 VIVOS test samples
conda run -n audio-lab python scripts/evaluate_noise_robustness.py
```
This produces a Markdown report saved to `docs/reports/asr_noise_robustness.md`.

---

## 📂 Project Structure

```
docs/            Notes, TODOs, decisions, phase-submission drafts, and evaluation reports
experiments/     Python Jupyter notebooks for model evaluations, exports, and benchmarks
src/             Rust source code for on-device edge runtime pipeline
models/          ONNX/quantized model files (gitignored, populated via scripts)
data/            Industrial glossaries, TSV concept maps, and audio eval fixtures
configs/         Latency budget configurations and language pair parameters
scripts/         Pipeline scripts, quantizers, and model setup tools
```
