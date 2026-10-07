# scripts/qaihub — Qualcomm AI Hub edge-AI pipeline test

Runs the NPU-bound stages of the OneVoice pipeline against the **QCS6490**
on Qualcomm AI Hub, using the `qai-hub` Python SDK, and reports the
edge-AI metrics (model size, compiled-context size, inference latency,
peak memory, estimated FLOPs).

The QCS6490 is listed on AI Hub as **`Dragonwing RB3 Gen 2 Vision Kit`**
(`chipset:qualcomm-qcs6490`, `os:qc_linux`, `hexagon:v68`,
`framework:qnn`) — the same silicon/OS family as the Rubik Pi 3 prototype.

## What's tested

| Model | Stage | NPU target? |
|---|---|---|
| `vad` | Silero VAD | CPU in the design, but small enough to profile on the NPU |
| `asr_vi_encoder` / `asr_vi_decoder` | PhoWhisper-base (VI) | yes |
| `asr_en_encoder` / `asr_en_decoder` | Whisper-small (EN/ZH/KO) | yes |
| `mt_encoder` / `mt_decoder` | Opus-MT (vi-en, INT8) | yes |

TTS (Piper/MMS) and RNNoise denoise are CPU-only by design and are not
sent to the Hub.

## Setup (one-time)

```sh
# Python tooling lives in a project-local venv (not the system Python).
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install qai-hub onnx numpy

# Write ~/.qai_hub/client.ini from .env
.venv/bin/python scripts/qaihub/configure.py
```

The token must be in `.env` as `QAIHUB_API_KEY=...` (or in the
environment).

## Usage

```sh
# Compile + profile every model, print the metrics table
.venv/bin/python scripts/qaihub/compile_and_profile.py

# ...or just a subset (keys from common.MODELS)
.venv/bin/python scripts/qaihub/compile_and_profile.py --keys vad mt_encoder

# Run on-device inference (zero inputs) and dump output shapes
.venv/bin/python scripts/qaihub/inference.py --keys vad mt_encoder

# Quantize to a lower precision (int8/int16/int4) and compare size
.venv/bin/python scripts/qaihub/quantize.py --key vad --weights int4 --compile
```

## Reading the output

- **src(MB)** — the uploaded ONNX size (what ships in the model bundle).
- **ctx(MB)** — the compiled QNN context-binary size (what actually loads
  onto the Hexagon NPU at runtime; usually much smaller than the ONNX).
- **latency** — single-inference NPU time. Note this is *not* the
  button-release→audio latency: that adds CPU denoise/VAD/TTS and the
  autoregressive decode loop, measured locally at ~740 ms end-to-end via
  `scripts/run_pipeline.sh`.
- **peakRAM(MB)** — peak NPU/driver memory during inference.
- **est GFLOPs** — a best-effort count over the graph's MatMul/Gemm nodes
  (attention-softmax/conv not counted; treat as an estimate).

## Known gotchas

- The merged decoders (`*_decoder`, with 25–49 inputs and dynamic
  KV-cache shapes) are the hard case — the same "merged decoder" that
  `docs/decisions/onnx-runtime-merged-decoder-gotchas.md` documents. They
  may fail to compile for QNN; a failure here is a *finding* about the
  QCS6490 deployment path, not a bug in this harness.
- Jobs run on Qualcomm's cloud and can take a few minutes each. The
  `--keys` flag lets you re-run just the interesting ones.

## Precision / quantization — int8 vs int4 vs fp8

- **The bundled ASR/MT models are already INT8** (`setup_models.sh` pulls
  the `*_quantized` ONNX exports). int4 would be only a further 2× on the
  *weights*, not 8× vs fp32.
- **qai-hub's PTQ only offers INT8 / INT16 / INT4** — there is no FP8 in
  the SDK.
- **QCS6490 (Hexagon v68) is the constraint.** Its AI Hub device
  attributes list `hexagon:v68` but do *not* advertise
  `htp-supports-fp16` or `htp-supports-weight-sharing` (newer HTPs like
  QCS8550 / QCM6690 do). int4 weight-sharing and fp16 are therefore
  unlikely to actually accelerate on this target — worth a real compile
  test before assuming it helps.
- **Storage isn't the binding constraint.** The ~2 GB resident budget has
  headroom; latency (ASR autoregressive decode) and accuracy are the
  bottlenecks. And the compiled QNN context binary can be *larger* than
  the int8 ONNX (e.g. `mt_encoder` 44.8 MB ONNX → ~99 MB context), so
  ONNX-file size is a poor proxy for on-device footprint.
- **int4 accuracy risk lands exactly on the differentiator** — the
  industrial glossary/safety terms. int4 post-training quantization of
  Whisper/Opus-MT commonly costs measurable WER/BLEU; that's the wrong
  place to save bytes.
- Net: keep int8 as the baseline; treat int4/fp8 as a Phase-3-or-later
  experiment (needs fp32 originals + a static-shape freeze first), not a
  demo blocker.
