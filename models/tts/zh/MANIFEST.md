# tts/zh

- Source: rhasspy/piper-voices, zh/zh_CN/huayan/medium/zh_CN-huayan-medium (Piper VITS voice, natively ONNX --
  not exported by us)
- Packaged: 2026-07-16
- License: MIT (rhasspy/piper-voices repo tag); underlying voice's own
  dataset license not separately audited -- fine for the hackathon
  prototype, audit before commercialization (same espeak-ng GPL note as
  tts/vi applies to the phonemizer, not the model).
- Phonemizer: espeak-ng (zh voice), same front-end as tts/vi.
- Quality: medium tier, single speaker, 22050Hz.
- Eval: round-trip TTS -> Whisper-small ASR intelligibility check, see
  experiments/04_tts.ipynb section 5.

Files: `zh_CN-huayan-medium.onnx`, `zh_CN-huayan-medium.onnx.json`.
