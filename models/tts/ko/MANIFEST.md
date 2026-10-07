# tts/ko

- Source: facebook/mms-tts-kor (Meta MMS-TTS, VITS). PyTorch checkpoint exported to
  ONNX here (unlike Piper, which ships ONNX) -- see experiments/04_tts.ipynb
  section 5.
- License: **CC-BY-NC 4.0** (facebook/mms-tts-kor model card, checked
  2026-10-07; the MMS *code* is MIT, the weights are not). Non-commercial
  only -- fine for the hackathon prototype, a commercialization blocker.
  Earlier versions of this file said MIT; that was wrong. No espeak/GPL
  dependency, unlike the Piper stages. See THIRD_PARTY_LICENSES.md.
- Packaged: 2026-07-16
- Precision: fp32. Exported with noise_scale=0 and noise_scale_duration=0
  for deterministic, reproducible output (VITS is stochastic by default);
  ONNX matches the PyTorch reference to <1e-2 on the probe sentences.
- Sample rate: 16000Hz.
- **Text front-end: uroman romanization is REQUIRED** (tokenizer is
  is_uroman=True; its raw vocab is only ~25 Latin
  symbols). Python applies it automatically via the `uroman` package.
  The Rust port needs a Korean Hangul->Latin romanizer in front of the
  tokenizer -- this is the one TTS language that does NOT go through
  espeak; tracked as a real porting task in docs/TODO.md.
- Eval: round-trip TTS -> Whisper-small ASR, see section 5 below.

Files:
- `mms_tts_kor.onnx` (input_ids -> waveform, dynamic seq length)
- tokenizer files (`tokenizer_config.json`, `vocab.json`, ...) from
  save_pretrained -- the char->id map + is_uroman/normalize flags.
