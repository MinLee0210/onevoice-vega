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
Run the setup script to download pre-trained VAD/ASR/TTS models directly from the Hugging Face Hub and automatically export/quantize the translation model components:

```bash
# Download and initialize models (includes automatic INT8 quantization)
scripts/setup_models.sh
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
