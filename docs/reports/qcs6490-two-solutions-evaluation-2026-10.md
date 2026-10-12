# QCS6490 — two deployable solutions evaluated (2026-10-12)

Two ways to deploy the OneVoice pipeline on the target chip (QCS6490,
Dragonwing RB3 Gen 2 Vision Kit). Both fit on **one** QCS6490 (CPU + Hexagon
v68 NPU); neither needs a second processor.

Sources of numbers:
- **NPU figures** — Qualcomm AI Hub, measured on the real QCS6490 device
  (`data/qaihub/*.csv`, full-INT8 QNN context binaries).
- **CPU figures** — the local Rust pipeline on the dev Mac (M-series), *not*
  the QCS6490 Kryo; treat as a fast-machine proxy (QCS6490 CPU will be slower).

## Solution 1 — All-CPU (baseline, deployable today)

Runs the existing Rust pipeline unchanged: every stage on the Kryo CPU via
ONNX Runtime. No QNN, no model changes.

| stage | hardware | status | Inference Time | Peak Memory |
|---|---|---|---|---|
| capture (WAV) | CPU | OK | 2.5 ms | — |
| denoise (RNNoise) | CPU | OK | 5.4 ms | — |
| VAD (Silero) | CPU | OK | 4.9 ms | — |
| ASR (PhoWhisper-base) | CPU | OK | 407.7 ms (mel 6.7 + enc 169.0 + dec 231.8) | — |
| MT (Opus-MT vi-en) | CPU | OK | 23.4 ms | — |
| TTS (Piper EN) | CPU | OK | 98.9 ms | — |
| **total (vi→en)** | **CPU** | **OK** | **540.6 ms** | **peak RSS 1.18 GB** |

## Solution 2 — Hybrid (ASR on Hexagon NPU, rest on CPU)

ASR network on the NPU using the **split** decoder export (the merged
decoder does not compile for QNN). MT / VAD / denoise / TTS stay on CPU.

| stage | hardware | status | Inference Time | Peak Memory |
|---|---|---|---|---|
| capture | CPU | OK | 2.5 ms | — |
| denoise (RNNoise) | CPU | OK | 5.4 ms | — |
| VAD (Silero) | CPU | OK | 4.9 ms | — |
| ASR encoder (PhoWhisper-base) | **NPU** | OK | 230.4 ms | 17.9 MB |
| ASR decoder first (PhoWhisper) | **NPU** | OK | 14.1 ms | 39.4 MB |
| ASR decoder past ×N (PhoWhisper) | **NPU** | OK | 8.9 ms / token | 37.8 MB |
| ASR encoder (whisper-base) ¹ | **NPU** | OK | 63.0 ms | 17.9 MB |
| ASR encoder (whisper-small) ¹ | **NPU** | OK | 178.4 ms | 20.4 MB |
| MT (Opus-MT vi-en) | CPU | OK | 23.4 ms | — |
| TTS (Piper EN) | CPU | OK | 98.9 ms | — |
| **total vi→en (PhoWhisper, ~16 tokens)** | **NPU+CPU** | **OK** | **≈510 ms** (ASR ≈378 + CPU ≈132) | NPU ≤39 MB + CPU resident |

¹ whisper-base is the encoder used for the EN/ZH/KO path (whisper-small is the
shipped model; its split decoders are the same shape as whisper-base).

## Side-by-side

| metric | Solution 1 (all-CPU) | Solution 2 (hybrid NPU) |
|---|---|---|
| deployable on QCS6490 | ✅ (today) | ✅ (needs split-decoder runtime change) |
| models: ASR | ONNX, CPU EP | QNN context binaries (encoder + split dec) |
| models: MT / VAD / TTS / denoise | ONNX, CPU | ONNX, CPU (same) |
| ASR compute | CPU 407.7 ms | NPU 378.0 ms (PhoWhisper) / 218.6 ms (whisper-base) |
| NPU peak memory | — | ≤39.4 MB |
| CPU process peak RAM | 1.18 GB | lower (ASR offloaded) |
| model storage (vi→en pair) | ≈287 MB | ≈210 MB CPU + NPU context binaries |
| extra work | none | ship split decoders; two decoder sessions in `asr.rs` |

## Notable finding

On QCS6490 the PhoWhisper-base **encoder on the NPU (230 ms) is not faster
than the Mac CPU encoder (169 ms)** — the HTP is weak and that export is not
optimized. The decoder loop *is* faster on the NPU (147.6 ms vs 231.8 ms).
The NPU win is clear for whisper-base (63 ms encoder) used on EN/ZH/KO. So
Solution 2's benefit is real but modest for VI specifically; a stronger case
appears once whisper-small's split decoders are measured and once the same
test runs against the actual QCS6490 CPU (weaker than the Mac).

## Attempted and rejected: all-NPU (MT on Hexagon)

| stage | hardware | status | Inference Time | Peak Memory |
|---|---|---|---|---|
| MT encoder/decoder (Opus-MT) | NPU | ❌ COMPILE_FAILED | — | — |
| VAD (Silero) | NPU | ❌ COMPILE_FAILED (shape inference) | — | — |

MT's float additive attention mask is rejected by the INT8-only HTP; removing
it (safe — verified it is only the cross-attention padding mask, not
causality) clears the type error but the QAIRT quantizer then fails. VAD is a
CPU stage by design, so its NPU failure is irrelevant. Details in
`docs/reports/qcs6490-deployment-evaluation-2026-10.md`.
