# asr/en-zh-ko

- Source: onnx-community/whisper-small (pre-converted ONNX -- not exported by us;
  underlying weights are openai/whisper-small, MIT)
- Packaged: 2026-07-15
- Precision: int8 (quantized variant from the source repo)
- One shared multilingual model for EN/ZH/KO per CLAUDE.md's model
  stack table -- language is selected at generate() time via the
  `language=` kwarg, not by swapping models.
- Eval: only the EN path validated so far (5 LibriSpeech-dummy samples,
  WER logged in experiments/02_asr.ipynb section 7). ZH/KO not tested
  yet -- same artifact, tracked as a follow-up in docs/TODO.md.
- Same preprocessing note as asr/vi/MANIFEST.md (needs a log-mel
  feature extractor ahead of the ONNX encoder).

Files: same shape as asr/vi/ (encoder_model.onnx, decoder_model_merged.onnx,
tokenizer.json, preprocessor_config.json, generation_config.json).
