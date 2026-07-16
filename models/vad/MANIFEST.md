# vad

- Source: [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad), `onnx/model.onnx`
  (fp32) -- pre-made ONNX, not exported by us. Original model:
  [snakers4/silero-vad](https://github.com/snakers4/silero-vad).
- License: MIT
- Downloaded/repackaged: 2026-07-16 (via experiments/01_denoise_vad.ipynb)
- Precision: fp32. The source repo also ships fp16/int8/uint8/q4/bnb4
  variants (same pattern as the Whisper/PhoWhisper ONNX repos) --
  quantization not yet evaluated for this stage, tracked in docs/TODO.md.
- I/O signature (used by src/stages/vad.rs):
  - `input`: `[1, 512]` f32 -- one 512-sample chunk (32ms @ 16kHz)
  - `state`: `[2, 1, 128]` f32 -- recurrent hidden state, carried between
    calls; reset to zeros at the start of each utterance
  - `sr`: scalar int64 -- sample rate (16000 here; Silero also supports
    8000 with a 256-sample chunk, not used in this pipeline)
  - `output`: `[1, 1]` f32 -- speech probability for this chunk
  - `stateN`: `[2, 1, 128]` f32 -- updated state, feed into the next call
- Validated (this notebook): mean speech probability ~0.5-0.7 on 3 real
  VIVOS utterances, ~0.007-0.009 on pure silence and quiet white noise --
  correctly separates speech from non-speech. Not yet validated against
  a labeled VAD benchmark (precise boundary accuracy), only this coarse
  separation check.
- Rust: `src/stages/vad.rs`'s `Vad::trim` chunks the input, runs each
  512-sample chunk through the model carrying `state` forward, then
  trims samples before the first and after the last chunk with
  probability > 0.5. Interior pauses are left alone. Real integration
  test against a committed WAV fixture passes (`cargo test -- --ignored`).

Files: `model.onnx`.
