# Whisper encoder tail-trim — analysis & plan (2026-10-13)

## Why it's not a one-liner

Whisper always runs a fixed **30 s / 3000-frame** mel window. The ONNX
encoder's positional embeddings are baked in at export, so feeding a shorter
window **hard-errors** — documented in `src/stages/asr.rs` module docs
(confirmed by testing variable-length input). So "trim the tail" is not a
runtime flag; it needs a re-export.

## What trimming would require

1. **Re-export the encoder** with positional embeddings of length
   `L = ctx_seconds * 50 + 1` (e.g. 2 s→101, 5 s→251, 10 s→501, 30 s→1501),
   or a genuinely dynamic-position variant.
2. **Rust**: parameterize `N_SAMPLES` / `N_FRAMES` in `src/stages/asr.rs`
   (currently fixed at 480000 / 3000), pad/truncate to the chosen ctx, and
   reflect-pad accordingly.
3. **Decoder**: it consumes `encoder_hidden_states` cross-attention; the
   cross-KV length must match the new ctx (our split-decoder cache sizing
   already derives from the graph, so it should follow).

## Is it worth it?

- **On the QCS6490 NPU**, the whisper-base encoder measured **63 ms**
  (`data/qaihub/encoder_qnn_latency.csv`) — already small; the decoder loop
  dominates ASR. Trimming saves little.
- **On CPU**, the Mac encoder is ~170–230 ms for PhoWhisper; there the 30 s
  window is a real cost. But the **CTC backend** (`src/stages/ctc.rs`) removes
  the whole decoder loop and runs ASR in ~57 ms — a bigger, simpler win.
- Public evidence is **mixed**: shrinking Whisper's audio context gives ~3×
  encoder speedup but can raise WER and increase hallucination (zero-padding
  helps suppress hallucination). It must be **measured**, not assumed.

## Recommendation

Deprioritized behind the CTC backend (shipped) and the ASR confidence gate
(shipped). Before spending time here:
1. Measure WER at ctx = 2/5/10/30 s on the VIVOS slice (re-export needed).
2. Only pursue if the encoder is still the bottleneck after CTC/on-device
   profiling — on the NPU it currently isn't.

## Scaffold (when pursued)

- `scripts/edge/export_whisper_ctx.py`: export encoder/decoder with a chosen
  `N_FRAMES` (slice the positional embedding initializer before export).
- `src/stages/asr.rs`: replace the `N_SAMPLES`/`N_FRAMES` consts with a
  `ctx_frames` field derived from the loaded encoder graph.
- Re-run `scripts/evaluate_asr.py` for WER and the pipeline for latency.
