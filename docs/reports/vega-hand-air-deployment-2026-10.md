# Vega Hand vs Vega Air — deployment evaluation (2026-10-12)

The two solutions in the Phase-2 proposal are the two carriers sharing one
engine (QUAVIO/Vega "Vega Core"): **Vega Hand** (one-to-one PTT handheld) and
**Vega Air** (VI→EN/ZH/KO drone broadcast). Both run on a **single QCS6490**
(CPU + Hexagon v68 NPU); neither needs a second processor.

One correction to the proposal text: it says the pipeline runs "entirely on
the QCS6490 **NPU**". Measured on the real QCS6490, only the **ASR** network
goes on the Hexagon NPU; MT, VAD, denoise and TTS run on the Kryo **CPU** of
the same SoC (Opus-MT does not compile to QNN — see the deployment report).
"On-device / one chip" is correct; "all on the NPU" is not.

## Per-PTT-cycle stage table (vi→en, measured)

| stage | hardware | status | Inference Time | Peak Memory |
|---|---|---|---|---|
| RNNoise denoise | Kryo CPU | OK | 5.3 ms | compiled in |
| Silero VAD | Kryo CPU | OK | 5.1 ms | 2.1 MB |
| ASR encoder (PhoWhisper-base) | Hexagon NPU | OK | 230.4 ms | 17.9 MB |
| ASR decoder first (split) | Hexagon NPU | OK | 14.1 ms | 39.4 MB |
| ASR decoder past (split) | Hexagon NPU | OK | 8.9 ms/token | 37.8 MB |
| MT Opus-MT vi-en | Kryo CPU | OK (NPU ✗) | 18.8 ms | ~127 MB |
| TTS Piper EN | Kryo CPU | OK | 95.1 ms | ~60 MB |
| Glossary | Kryo CPU | OK | <1 ms | termbase (KB) |
| **cycle total** | **CPU+NPU** | **OK** | **≈485 ms** (CLI, split decoder) | **≈1.2 GB RSS** |

NPU rows are the real QCS6490 (AI Hub). CPU rows are the dev Mac (fast proxy;
QCS6490 Kryo will be slower). ASR decoder now uses the split export end-to-end
in the Rust runtime (`src/stages/asr.rs`, `Decoder::Split`).

## Solution A — Vega Hand (VI↔EN/ZH/KO, one-to-one)

| | |
|---|---|
| **Models** | PhoWhisper-base (ASR VI) + Whisper-small (ASR EN/ZH/KO); Opus-MT for the 6 directions; Piper VI/EN/ZH + MMS-TTS KO; Silero VAD; RNNoise (in-binary); glossary termbase |
| **Storage (models)** | ASR ≈127 MB (VI split) + ≈240 MB (EN/ZH/KO) + MT ≈6×127 MB ≈760 MB + TTS ≈290 MB (4 voices + espeak) + VAD 2 MB ≈ **1.4 GB** |
| **RAM (resident, all pairs)** | proposal budgets ~2.5 GB; measured one-pair working set ≈1.2 GB process RSS → all-pairs figure is plausible and within 8 GB |
| **Hardware** | QCS6490: Hexagon NPU (ASR) + Kryo CPU (MT/TTS/VAD/denoise) |
| **Per-cycle latency** | ≈485 ms (Mac CPU proxy for the CPU stages) — inside the ≤1.2 s target |
| **Peak memory** | NPU ≤39 MB per ASR graph; CPU process ≈1.2 GB (one pair) |
| **Status** | ✅ deployable; needs split-decoder models (now shipped by `setup_models.sh`) |

## Solution B — Vega Air (VI→EN/ZH/KO, one-to-many broadcast)

Source audio is captured cleanly at the ground station, so it needs only VI
ASR and three forward MT directions + three TTS voices.

| | |
|---|---|
| **Models** | PhoWhisper-base (ASR VI only); Opus-MT VI→EN, VI→ZH, VI→KO; Piper EN/ZH + MMS-TTS KO; Silero VAD; RNNoise; broadcast-tuned TTS config |
| **Storage (models)** | ASR ≈127 MB + MT 3×127 MB ≈380 MB + TTS ≈230 MB (3 voices + espeak) + VAD 2 MB ≈ **740 MB** |
| **RAM (resident)** | proposal budgets ~2 GB; well within 8 GB (fewer models than Hand) |
| **Hardware** | QCS6490: Hexagon NPU (ASR) + Kryo CPU (MT/TTS/VAD/denoise); FC telemetry UART (ego-noise, stretch) |
| **Per-cycle latency** | VI ASR ≈378 ms + MT ≈19 ms + TTS ×3 sequential (broadcast target ≤2.0 s/language) |
| **Peak memory** | NPU ≤39 MB; CPU ≤ that of Hand |
| **Status** | ✅ deployable on the same hardware (subset of Hand's models; no new NPU work) |

## Comparison

| Solution | Hardware | Mem cost (RAM) | Storage | Inference Time / cycle | Peak Memory | Status |
|---|---|---|---|---|---|---|
| **Vega Hand** | 1× QCS6490 (NPU+CPU) | ~2.5 GB budget | ~1.4 GB | ~485 ms | NPU ≤39 MB; CPU ~1.2 GB | ✅ |
| **Vega Air** | 1× QCS6490 (NPU+CPU) | ~2.0 GB budget | ~740 MB | ASR ~378 + MT ~19 + 3× TTS | NPU ≤39 MB | ✅ |
| All-CPU baseline (vi→en) | 1× QCS6490 (CPU) | 1.18 GB measured | ~287 MB | ~541 ms | ~1.2 GB | ✅ |
| All-NPU attempt (MT on NPU) | — | — | — | — | — | ❌ MT/VAD won't compile |

## Bottom line

Both proposed solutions deploy on one QCS6490. Air is a strict model subset of
Hand (fewer MT/TTS/ASR assets), so once Hand is brought up, Air is a config —
consistent with the "one engine, two carriers" pitch. The only non-CPU stage is
ASR, and it is proven on the NPU with the split decoder.
