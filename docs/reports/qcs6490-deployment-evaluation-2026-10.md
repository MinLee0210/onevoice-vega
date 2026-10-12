# QCS6490 deployment evaluation (2026-10-12)

Question: **can the whole OneVoice pipeline be deployed on the target chip
(QCS6490, Dragonwing RB3 Gen 2 Vision Kit on Qualcomm AI Hub), or does it
need more than one device/accelerator?**

Short answer: **one QCS6490 runs the whole pipeline.** The SoC already
contains both the Kryo CPU and the Hexagon v68 NPU. The NPU hosts the ASR
network (encoder + **split** decoder); MT, VAD, RNNoise denoise and TTS run
on the CPU. No second chip is required — "multiple hardware" is not needed.

But not every exported artifact compiles for the NPU as-is; details below.

## Method

`scripts/qaihub/benchmark_qnn.py` (server-side FP32 → full-INT8 QNN context
binary, `--target_runtime precompiled_qnn_onnx --quantize_full_type int8
--quantize_io --truncate_64bit_io`) and `scripts/qaihub/compile_and_profile.py`
on the `Dragonwing RB3 Gen 2 Vision Kit` (chipset `qualcomm-qcs6490`,
`hexagon:v68`; no `htp-supports-fp16`, no weight sharing). Raw CSVs:
`data/qaihub/`.

## Results

| Stage | Artifact | QCS6490 NPU | Latency | Notes |
|---|---|---|---|---|
| ASR encoder | whisper-base | ✅ | 63.0 ms | full-INT8 context binary |
| ASR encoder | PhoWhisper-base | ✅ | 230.4 ms | |
| ASR encoder | whisper-small | ✅ | 178.4 ms | |
| ASR decoder (**merged**) | whisper/PhoWhisper | ❌ | — | QAIRT exit 255 — the merged graph's `use_cache_branch` If-node |
| ASR decoder (**split**) | whisper-base first | ✅ | 19.1 ms | `decoder_model.onnx` |
| ASR decoder (**split**) | whisper-base past | ✅ | 9.1 ms | `decoder_with_past_model.onnx` |
| MT encoder/decoder | Opus-MT vi-en (FP32) | ❌ | — | float attention-mask `Cast` unsupported on the INT8-only HTP; after graph surgery the QAIRT quantizer still fails (exit 1) |
| VAD | Silero | ❌ | — | AI Hub shape inference fails; **CPU by design**, not needed on NPU |
| Denoise / TTS | RNNoise / Piper, MMS | CPU | — | Kryo; never NPU targets |

## What this means

- **Single chip is sufficient.** ASR (the heaviest stage) runs on the NPU.
  Everything else is CPU. The 1.2 s budget is plausible: e.g. whisper-base
  ASR ≈ encoder 63 ms + decoder loop (≈19 ms first + 9 ms × tokens) ≈ 220 ms
  for ~16 tokens, plus CPU MT/VAD/TTS (≈100–400 ms measured locally).
- **One required code change:** ship the **split** first/past decoders instead
  of the merged decoder and switch the decode loop in `src/stages/asr.rs`
  (and `mt.rs` if MT ever moves to NPU) to two sessions. The merged decoder is
  what fails on QNN; the split export runs fine.
- **MT stays on the CPU for now.** Opus-MT's float additive attention mask is
  rejected by the QCS6490 HTP. Removing it (valid for our batch=1, all-ones
  mask) clears the type error but the QAIRT quantizer then fails, so MT needs
  model-specific work (calibration data, further surgery, or a different MT
  export) before it can use the NPU. CPU MT (~90–240 ms) fits the budget, so
  this is not a blocker.
- **VAD is a non-issue** — it is a CPU stage in the design; the NPU attempt
  was only exploratory.

## Harness facts learned (SDK 0.56.0)

- `qnn_context_binary` is **no longer** a valid `--target_runtime`; the SDK
  accepts only `onnx`, `precompiled_qnn_onnx`, `qnn_dlc`, `tflite`. Any doc
  still saying `qnn_context_binary` (incl. the 2026-08 ADR) is stale — use
  `precompiled_qnn_onnx`.
- `--truncate_64bit_io` is mandatory for int64 token-id/mask inputs (all
  decoders). With it, the **compiled** graph's int64 inputs become int32, so
  inference inputs must be int32 (`common.sample_inputs` fixed accordingly).
- The bundled `setup_models.sh` ARTIFACTS are dynamic-INT8 ONNX, which the
  QAIRT converter rejects (exit 255) — they are ORT-CPU artifacts, not QNN
  inputs. NPU deployment must start from FP32 sources (as `benchmark_qnn.py`
  does).

## Remaining choices (not yet tried)

For getting **MT** onto the NPU (not required for a single-chip demo):
1. Compile the mask-stripped MT graph with representative **calibration data**
   (AI Hub hosted quantize job) instead of the default calibration.
2. Deeper graph surgery / re-export MT (e.g. bake the mask, or export a
   QNN-friendly attention with integer masks).
3. Swap the MT model for one already known to convert on QCS6490.

Beyond those, the straightforward deployment paths are exhausted; the pipeline
itself already fits on one QCS6490 chip.
