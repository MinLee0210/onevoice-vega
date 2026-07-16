# tts/vi

- Source: rhasspy/piper-voices, vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium (Piper voices are natively ONNX --
  not exported by us)
- Packaged: 2026-07-16
- License: CC BY 4.0 (VAIS-1000 dataset) -- the *only* commercially
  usable Vietnamese voice in this repo. Do NOT substitute
  `vivos-x_low` (CC-BY-NC-SA, blocked) or `25hours_single-low`
  (unlicensed). See docs/decisions/piper-vi-voice-license-trap.md.
- **Open licensing question, not resolved here:** Piper's phonemizer
  (piper-phonemize/espeak-ng) is GPL-3.0. This voice's config declares
  `"espeak": {"voice": "vi"}` -- Vietnamese phonemization requires it,
  there's no way around it for this specific model. Whether embedding
  it in the shipped on-device binary GPL-taints the product is an open
  question -- needs a real answer before Phase 3 commits to this
  architecture. Not a blocker for prototyping.
- Quality: medium (best available for VI in this repo), single speaker,
  22050Hz
- Eval: round-trip TTS -> PhoWhisper-ASR intelligibility check on 4 VI
  sentences (1 general + 3 industrial) -- 3/4 essentially verbatim, 1
  industrial sentence significantly misheard. See
  experiments/04_tts.ipynb section 3. No human-listener check done yet.

Files:
- `vi_VN-vais1000-medium.onnx`, `vi_VN-vais1000-medium.onnx.json`
  (phoneme_id_map + audio/inference config -- both needed, this isn't
  a pure ONNX-in-ONNX-out interface, the JSON side carries the
  phoneme vocabulary)
