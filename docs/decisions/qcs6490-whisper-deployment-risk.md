# Risk: "Whisper-Small (AI Hub, pre-quantized)" isn't actually a ready path on QCS6490

AGENTS.md's model stack table lists the EN/ZH/KO ASR baseline as
"Whisper-Small (AI Hub, pre-quantized)" as if Qualcomm AI Hub already
has a working, deployable, quantized Whisper for the QCS6490. As of
writing, **that's not true** — worth flagging loudly before Phase 3
hardware bring-up planning assumes it is.

## What's actually going on

[qualcomm/ai-hub-models issue #281](https://github.com/qualcomm/ai-hub-models/issues/281)
("Quantized Whisper models for QCS6490 Linux devices") lays out three
compounding problems deploying any Whisper variant to QCS6490-class IoT
devices (Radxa Dragon Q6A, RB3 Gen 2 — same SoC family as the Rubik Pi 3):

1. **Floating-point export fails.** Non-quantized Whisper export with
   `--precision float` fails outright: `"Tensor 'input_features' has a
   floating-point type which is not supported by the targeted device."`
   The QCS6490 NPU wants quantized I/O, not fp32/fp16.
2. **The quantized path needs AIMET-ONNX, which has no aarch64 Linux
   wheel.** `whisper_small_quantized` requires AIMET-ONNX to run;
   attempting it on an aarch64 Linux device (i.e. the actual target
   hardware) fails with `"RuntimeError: AIMET-ONNX is missing but must
   be installed."` Building AIMET-ONNX from source was reported
   impractical.
3. **QCS6490 isn't a recognized deployment target in some AI Hub
   tooling paths** — pre-compiled assets and parts of the AI Hub website
   don't list it.

**Status: closed as not planned.** No fix or workaround shipped by
Qualcomm as of this writing.

Workarounds mentioned in the issue: plain HF Transformers on CPU
(~2.3x slower than realtime — likely blows the ≤1.2s latency target on
its own for anything beyond a very short utterance), or downloading a
generic ONNX Whisper and trying to run it through the QNN execution
provider directly — but the QNN EP itself isn't available via pip for
aarch64 Linux either, per the same issue thread.

## Why this matters for OneVoice

AGENTS.md's plan leans on "AI Hub Workbench porting journey" as part of
the pitch story (see AGENTS.md → Contest/business context: "document
the AI Hub/Workbench porting journey"). If AI Hub's own Whisper support
for this exact chipset is a known dead end, that part of the story needs
a real answer before Phase 3, not just an assumption it'll work.

## What we know does work today (see experiments/02_asr.ipynb)

- Pre-made ONNX Whisper/PhoWhisper (from `onnx-community`, `Xenova`, or
  `huuquyet`) runs fine via plain ONNX Runtime + `ort` on **CPU**, fp32
  or int8, validated against PyTorch and ground truth.
- Not yet validated: any of this running through the **QNN execution
  provider** specifically, on real hardware. That's the actual open
  question — CPU-only ASR at Whisper-small size is very unlikely to hit
  the latency target on a Kryo 670, per AGENTS.md's own reasoning for
  why ASR/MT need the NPU in the first place.

## Options to evaluate before Phase 3

1. Re-check AI Hub Workbench / AIMET-ONNX status closer to Phase 3 —
   this is fast-moving tooling, it may have shipped an aarch64 wheel or
   fixed the export by then.
2. Try quantizing a generic ONNX Whisper (e.g. one of the
   `onnx-community` int8/uint8 variants already validated in
   `experiments/02_asr.ipynb`) directly with the QNN execution provider,
   bypassing AI Hub Workbench's quantization step entirely — the model
   is already quantized, the question is just whether QNN EP will load
   it.
3. Budget real bring-up time in Phase 3 specifically for this — don't
   assume it's a solved problem walking in.
4. Keep PhoWhisper-base (74M, smallest) as the fallback size if
   Whisper-small can't be made to fit the latency budget even once NPU
   execution works.
