# CLAUDE.md — OneVoice AI Challenge Project

## Project
On-device (no cloud) speech translation device for noisy industrial
environments (factories, construction, logistics, hospitals). Built for
Saigon AI Hub × Qualcomm "OneVoice AI Challenge." Currently Phase 2
(technical spec submission, due July 2026).

## Languages/pairs
VI↔EN, VI↔ZH, VI↔KO (6 directions total)

## Conversation model
Half-duplex, push-to-talk (PTT). One party holds button, speaks, releases;
device translates and plays back. No full-duplex streaming needed —
process complete utterances (≤10s), not partial/incremental audio.

## Hardware target
- Prototype: Rubik Pi 3 (Qualcomm Dragonwing QCS6490), 8-core Kryo 670 CPU,
  Hexagon NPU @ 12 TOPS, 8GB LPDDR4x, 128GB UFS
- Production vision: same QCS6490-class SoM shrunk into ruggedized
  handheld/neckband form factor, IP54+, physical PTT button, LED status
  (no touchscreen)

## Compute split
```
CPU (Kryo)     → audio I/O, ring buffer, RNNoise, Silero VAD,
                 orchestration, Piper TTS (runs fine on CPU)
Hexagon NPU    → ASR encoder/decoder, MT encoder/decoder
                 (the heavy transformer matmuls, quantized)
```

## Pipeline
```
Mic array → Denoise (RNNoise/DeepFilterNet3) → VAD trim (Silero)
  → ASR → MT → TTS → Speaker
```
Denoise runs during recording (zero added latency). TTS streams output
(start playback before full synthesis completes) to hide latency.

## Model stack

| Stage | Baseline | Upgrade path |
|---|---|---|
| Denoise | RNNoise (CPU) | DeepFilterNet3 (better on machinery noise) |
| VAD | Silero VAD (MIT, ONNX) | — |
| ASR (VI) | PhoWhisper-base 74M | PhoWhisper-small 244M, W8A16 on Hexagon |
| ASR (EN/ZH/KO) | Whisper-Small (AI Hub, pre-quantized) | Whisper-Large-v3-Turbo quantized |
| MT | Opus-MT per-pair (~300MB, Apache/CC-BY, INT8) | LoRA fine-tune on industrial glossary; VinAI-Translate/envit5 for VI↔EN quality |
| TTS | Piper (VITS/ONNX, CPU real-time) | Kyutai Pocket TTS if quality gap found |

**License trap:** NLLB is CC-BY-NC — avoid for anything meant to be
commercialized. MADLAD-400 (Apache 2.0) is broad-coverage but likely too
big (3B) for comfortable NPU deployment — treat as a stretch option, not
baseline.

## Memory budget (quantized, resident)
~2GB total. Lazy-load only the active language pair's MT+TTS to stay
comfortable on 8GB device RAM.

## Latency target
≤1.2s button-release → first audio (stretch: 0.8s). ASR decoder
(autoregressive) is the bottleneck — keep models small (base/small tier,
not medium/large), greedy/beam=1 decode for MT.

## Performance targets
```
End-to-end latency (release→first audio)   ≤ 1.2s
ASR WER, VI, 75dB factory noise            ≤ 15%
MT BLEU VI↔EN (PhoMT + domain test set)    ≥ 40
Glossary term accuracy                     ≥ 95% enforced
Battery                                    ≥ 8h continuous
Connectivity required                      zero (airplane-mode demo)
```

## Differentiators (write into every proposal/pitch)
1. **Industrial glossary engine** — VI/EN/ZH/KO parallel termbase
   (safety commands, machine names, logistics terms), enforced via LoRA
   fine-tune + constrained decoding. Swappable per-industry vocab packs.
2. **Noise-first audio front-end** — 2-mic beamforming + denoise,
   benchmarked WER-vs-SNR curve at 70-85dB factory noise.
3. **Latency engineering** — denoise-during-capture, greedy MT decode,
   streaming TTS playback, instrumented per-stage timing / live latency
   HUD for demos.

## Tooling / language split
```
Model eval, fine-tuning, quantization  → Python
  (PyTorch, HF Transformers, Qualcomm AI Hub Workbench, ONNX export)

On-device runtime (production loop)    → Rust (if team knows Rust) or C++
  - `ort` crate (Rust) or ONNX Runtime C++ API, QNN execution provider
  - cpal/hound for audio I/O (Rust) or ALSA (C++)
  - No Python in the real-time loop (GIL/overhead)
```
Handoff artifact between phases: the exported `.onnx` file (or QNN
context binary) — language-agnostic interface.

**Rust risk note:** `ort` crate's `qnn` feature is real and functional,
but community troubleshooting content is thin vs. Python/C++. If team is
Rust-inexperienced, prefer C++ to avoid learning Rust + QNN simultaneously
under deadline pressure. If going Rust, smoke-test `ort`+QNN loading one
quantized model on real hardware in week 1, in parallel with model eval.

## Hexagon / QNN glossary (for reference)
- **Hexagon** = dedicated low-power DSP-turned-NPU chip block on the SoC
  (separate from CPU/GPU), evolved from Qualcomm's original signal
  processor. HVX (vector) + HMX (matrix/tensor) execution units.
- **HTP** = Hexagon Tensor Processor, the AI-execution part of Hexagon.
- **QNN SDK / QAIRT** = Qualcomm's software stack/compiler that takes a
  model and hands it to Hexagon's HTP backend. `libQnnHtp.so` /
  `QnnHtp.dll` is the concrete backend library referenced in code.
- Flow: `.onnx model → QNN SDK → HTP backend → runs on Hexagon silicon`

## Hardware toolchain
Qualcomm AI Hub (pre-optimized model exports, e.g. Whisper W8A16) → AI Hub
Workbench (quantization/profiling) → QNN SDK/QAIRT (deployment). Rubik Pi
3 is Raspberry-Pi-HAT compatible — use for mic array (ReSpeaker HAT) and
PTT button/LED prototyping.

## Contest/business context
- Co-hosted with Qualcomm; part of Qualcomm's broader Vietnam AI R&D push
  (Hanoi/HCMC AI center opened June 2025, VinAI/MovianAI acquisition
  April 2025 — PhoWhisper/PhoMT lineage is now Qualcomm-owned talent).
- Judging: Technical 50% / Innovation 25% / Business viability 25%.
- Likely goal from Qualcomm's side: Dragonwing/QCS design wins, AI Hub
  toolchain adoption, reference showcase demos, a commercializable local
  product story (Vbee AI is the existing precedent — Vietnamese startup
  building offline voice solutions on Qualcomm edge platforms).
- Strategy: frame device explicitly as "Dragonwing-native," document the
  AI Hub/Workbench porting journey, position business pitch around
  potential Qualcomm ecosystem partnership (not just cash prize).
- IP: QVIC (Qualcomm's other VN program) took no equity/IP — verify
  OneVoice's terms in FAQ before Phase 2 submission.

## Timeline
```
Phase 2 (Jul 2026, current)  → technical spec submission
Phase 3 (Aug-Sep 2026)        → prototype build
Phase 4 (Oct 2026)            → field testing (real factory/warehouse)
Phase 5 (Nov 2026)             → finale demo, HCMC
```

## Open decisions / risks to track
- MADLAD-single-model vs. Opus-MT-per-pair — memory vs. speed tradeoff,
  needs benchmarking on real device.
- PhoWhisper accuracy post-quantization (W8A16 vs FP16 WER) — validate
  before committing.
- VI↔KO / VI↔ZH quality (low-resource pairs) — may need EN-pivot
  fallback (VI→EN→KO/ZH); measure quality delta.
- Field-test noise data scarcity — synthesize via VIVOS/Common Voice VI
  mixed with MUSAN/factory recordings at controlled SNRs.

## Repo layout
```
docs/       notes, TODOs, ADRs, phase-submission drafts (see docs/README.md)
notebook/   Python — model eval, fine-tuning, quantization, ONNX export
src/        Rust workspace — on-device runtime (see src/README.md)
models/     exported .onnx / QNN artifacts, the Python↔Rust handoff point
data/       glossary termbase, eval sets, raw corpora (mostly gitignored)
configs/    language-pair configs, model manifests, latency budgets
scripts/    glue scripts (AI Hub export, QNN convert, device deploy)
```
Each of `docs/`, `notebook/`, `src/`, `models/`, `data/`, `configs/`,
`scripts/` has its own `README.md` explaining what goes where.
