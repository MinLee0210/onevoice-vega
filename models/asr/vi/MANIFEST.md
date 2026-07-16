# asr/vi

- Source: huuquyet/PhoWhisper-base (pre-converted ONNX -- not exported by us; underlying
  weights are vinai/PhoWhisper-base, BSD-3-Clause)
- Packaged: 2026-07-15
- Precision: int8 (quantized variant from the source repo, not quantized
  by us). On a 10-sample VIVOS slice, identical WER to fp32
  (0.022) at ~3.8x smaller
  (77MB vs. 291MB) -- see
  experiments/02_asr.ipynb section 5.
- Decode: greedy (beam=1) via `model.generate(num_beams=1)` default,
  language/task forced to vi/transcribe
- Eval: WER against 10 VIVOS test-set samples (quocanh34/viet_vivos
  mirror -- the original `vivos` HF dataset uses a loading script no
  longer supported by recent `datasets` versions)
- Preprocessing note: needs a log-mel spectrogram feature extractor
  (`preprocessor_config.json` has the params) *before* the ONNX
  encoder -- unlike MT's plain tokenizer, this isn't just a lookup
  table, and isn't part of the ONNX graph. Not yet ported to Rust; see
  docs/TODO.md.

Files:
- `encoder_model.onnx`, `decoder_model_merged.onnx` (int8 quantized)
- `tokenizer.json` (already a proper fast tokenizer -- Whisper's BPE
  tokenizer has a native fast/tokenizer.json variant, unlike Marian's
  sentencepiece tokenizer in the MT stage; no manual conversion needed)
- `preprocessor_config.json` (feature extractor params: n_mels, hop
  length, sampling rate, etc.)
- `generation_config.json` (decoder_start_token_id, forced language/task
  token ids, suppress_tokens list)
