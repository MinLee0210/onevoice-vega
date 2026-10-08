# Pipeline performance research — firecrawl pass (2026-08)

Firecrawl search/scrape pass on the current CPU-only pipeline's latency
bottleneck. Bottom line: **the ASR stage is the bottleneck, and its biggest
lever — running on the QCS6490 Hexagon NPU — is no longer blocked.** The
exact recipe and pre-built binaries now exist; nothing local (AIMET etc.)
is required.

## 1. QCS6490 Whisper deployment is resolved (was the #1 risk)

The original blocker (AGENTS.md → Open decisions: "QCS6490 Whisper/AI Hub
deployment is broken — needs AIMET-ONNX, no aarch64 wheel") is now solved.
The `qualcomm-ai-hub-community` org on Hugging Face publishes **pre-compiled
full-INT8 QNN context binaries** for Whisper on QCS6490, with measured
on-device numbers:

- [`whisper-base-multilingual-qcs6490-qnn-int8`](https://huggingface.co/qualcomm-ai-hub-community/whisper-base-multilingual-qcs6490-qnn-int8)
- [`whisper-small-multilingual-qcs6490-qnn-int8`](https://huggingface.co/qualcomm-ai-hub-community/whisper-small-multilingual-qcs6490-qnn-int8)
- (more via the [`?other=qcs6490`](https://huggingface.co/models?other=qcs6490) HF filter)

Measured on the AI Hub-hosted `Dragonwing RB3 Gen 2 Vision Kit` (= QCS6490):

| Component | whisper-base | whisper-small |
|---|---|---|
| Encoder (30 s window, tail-trimmable) | **49.5 ms** / 13.8 MB | 188.7 ms / 32.7 MB |
| Decoder (per token) | **12.1 ms** / 26.4 MB | 33.8 ms / 73.8 MB |
| ~30-token transcript | **≈ 360 ms** | ≈ 1.0 s |

The **compile recipe** (server-side, no local AIMET):

```
--target_runtime qnn_context_binary --quantize_full_type int8 --quantize_io
```

Plus `--truncate_64bit_io` for the int64 token-id / attention-mask inputs.

The same recipe works for **any HF Whisper fine-tune** by changing
`hf_whisper_version` — explicitly including `vinai/PhoWhisper-small` for
Vietnamese. So the VI ASR can go to the NPU the same way.

Constraint map (why full INT8, not W8A16 or float):
1. Float graphs are rejected ("floating-point type not supported by target").
2. **W8A16 fails context-binary conversion (exit code 14)** — 16-bit
   activations unsupported on this HTP generation (Hexagon v68).
3. Full INT8 (`--quantize_full_type int8 --quantize_io`) is what works.

Caveat: blind (calibration-free) INT8 trades some accuracy — validate WER on
domain before production; use AI Hub hosted quantize jobs with calibration
audio for accuracy-critical deployments.

## 2. Projected pipeline impact

Current measured (CPU, this repo): ASR ≈ 592 ms (encoder 226 ms + decoder
359 ms for 16 tokens), full pipeline through TTS ≈ 741 ms.

With PhoWhisper-base on the NPU (whisper-base numbers): encoder ≈ 50 ms +
decoder ≈ 12 ms × 16 tokens ≈ 194 ms → **ASR ≈ 245 ms**, a ~2.4× ASR speedup
and ~350 ms off the end-to-end budget, *and* the CPU is freed for
denoise/VAD/TTS orchestration. The ≤1.2 s (stretch 0.8 s) target becomes
comfortable with headroom.

## 3. Other techniques found (secondary, keep in view)

- **distil-whisper** ([GitHub](https://github.com/huggingface/distil-whisper)):
  6× faster, 49% smaller, within 1% WER on short-form — but English-centric
  (`distil-large-v3` is multilingual). Not a VI lever; worth revisiting for the
  EN leg and as a proof that ASR distillation works.
- **WhisperRT** ([arXiv 2508.12301](https://arxiv.org/html/2508.12301v2)):
  causal streaming Whisper with a CTC head + two-pass decoding — the CTC
  pass removes the autoregressive decode loop for latency. Same family as the
  already-bookmarked VietASR (CTC) idea.
- **faster-whisper / CTranslate2**: ~4× via CTranslate2 + int8 — but a
  different runtime; the Rust `ort` path can't consume it directly.
- **KV cache**: already implemented (merged decoder + cached KV); the
  remaining optimizations (paged/quantized KV) target much longer sequences
  than ≤10 s utterances — low value here.
- **Static shapes preferred by NPUs** — the fixed 30 s padded encoder is
  actually *correct* for the HTP (static graph); the "tail-trimmable" window
  means decoder cross-attention can be trimmed to real audio length, which is
  the remaining encoder-side latency win.

## 4. Recommendations (prioritized)

1. **Compile PhoWhisper-base encoder/decoder to QCS6490 QNN context binaries**
   with `--target_runtime qnn_context_binary --quantize_full_type int8
   --quantize_io --truncate_64bit_io`, or consume the pre-built
   `whisper-base-...-qnn-int8` binaries. Biggest single win (~2.4× ASR).
2. **Enable `ort`'s `qnn` feature and load the context binaries** in
   `src/stages/asr.rs` / `mt.rs` (QNN EP) — week-1 Phase 3 smoke test.
3. **Trim the encoder tail** to the VAD-detected utterance length before the
   decoder cross-attention (the "tail-trimmable" note) — removes the 30 s
   padding cost on short utterances.
4. Update `scripts/qaihub/` compile options (done below) to the proven recipe;
   re-run compile/profile to confirm QNN context binaries (not `.tflite`).
5. Keep int8 (not int4/fp8) — see `scripts/qaihub/README.md` precision notes.

## 5. Compiled + profiled on QCS6490 (validated 2026-08)

Ran the actual compile/profile path on the target device (`Dragonwing RB3 Gen 2
Vision Kit`). Two things had to be corrected first:

- **The bundled ONNX is already INT8** (optimum dynamic quant) — the QAIRT/QNN
  converter rejects it (`exit code 255`) on both `qnn_dlc` and
  `precompiled_qnn_onnx`. The working path is **FP32 source → server-side
  full-INT8** (`--quantize_full_type int8 --quantize_io`), i.e. exactly the
  `qualcomm-ai-hub-community` recipe. `scripts/qaihub/benchmark_qnn.py` does this.
- **qai-hub reports profile time in microseconds**, not ms (SDK divides by
  1000 for display). `profile_summary` now converts; the 1000× bug is fixed.

Measured so far (full-INT8 QNN context binary, QCS6490):

| Model (FP32 source) | Inference | Peak mem |
|---|---|---|
| PhoWhisper-base encoder (VI) | **230.4 ms** | 17.9 MB |
| whisper-base encoder | **63.0 ms** | 17.9 MB |

The 3.7× gap between the two same-architecture encoders is an **export
artifact** (huuquyet's PhoWhisper-base FP32 export converts to a slower QNN
graph than onnx-community's whisper-base) — worth using `onnx-community`'s
export or re-exporting PhoWhisper cleanly before trusting the VI NPU number.
The QCS6490 NPU is not automatically faster than the ~226 ms CPU encoder here;
the win is architecture/export dependent, so benchmark the exact artifact.

A per-model × per-device latency matrix is produced by
`scripts/qaihub/benchmark_qnn.py` (CSV at `/tmp/onevoice_qnn_latency.csv`).

### Device matrix — PhoWhisper-base encoder (FP32 → full-INT8 QNN)

| Device | Chipset | Hexagon | Inference | Peak mem |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit **(target)** | qcs6490 | v68 | 230.4 ms | 17.9 MB |
| Arduino VENTUNO Q | qcs8275 | v73 | 56.5 ms | 16.1 MB |
| Dragonwing IQ-9075 EVK | qcs9075 | v73 | 56.2 ms | 20.0 MB |
| QCS8550 (Proxy) | qcs8550-proxy | v73 | 71.3 ms | 143.1 MB |
| Dragonwing Q-6690 MTP | qcm6690 | v73 | 300.6 ms | 152.9 MB |

Two takeaways: (1) newer Hexagon v73 HTPs run the encoder ~4× faster than the
target's v68, so the QCS6490 is genuinely the constrained part; (2) the target
is the *slowest per-model* result here (230 ms) while `whisper-base` on the
*same* QCS6490 was 63 ms — an export-quality problem, not a hardware ceiling.


## Sources (firecrawl)

- HF `qualcomm-ai-hub-community/whisper-{base,small}-multilingual-qcs6490-qnn-int8`
  (scraped — measured tables + recipe + W8A16/float constraint map)
- Qualcomm support forum "How to use qnn to load whisper model in QCS6490"
  (login-gated; related-discussion index scraped)
- HF Hub `?other=qcs6490` filter (all QCS6490 models)
- github.com/huggingface/distil-whisper; arXiv 2508.12301 (WhisperRT);
  arXiv 2407.05858 (llm.npu, static shapes)
