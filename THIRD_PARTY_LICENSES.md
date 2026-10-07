# Third-party licenses

Every model and phonemizer shipped in the `models/` bundle, with its license.
Per-model details live in `models/*/MANIFEST.md`; `scripts/pack_models.sh`
refuses to pack a model directory whose manifest has no `- License:` line.

Last audited: 2026-10-07. "Checked" means read from the upstream model card;
"unverified" means taken from our own earlier notes and not re-checked.

| Component | Used for | License | Status | Commercial use |
|---|---|---|---|---|
| `nnnoiseless` / RNNoise | denoise | BSD-3-Clause | unverified | Yes |
| Silero VAD | VAD | MIT | unverified | Yes |
| PhoWhisper-base | ASR (VI) | BSD-3-Clause | unverified | Yes |
| Whisper-small | ASR (EN/ZH/KO) | MIT | unverified | Yes |
| Opus-MT (Helsinki-NLP) | MT | manifests say Apache-2.0; upstream cards for `opus-mt-*` are, to my recollection, CC-BY-4.0 | **verify** | Yes, with attribution (either license) |
| Piper `vais1000-medium` | TTS (VI) | CC BY 4.0 | unverified | Yes, with attribution |
| Piper `lessac` (EN), `huayan` (ZH) | TTS | MIT (repo); underlying voice-dataset licenses not audited | partial | **Audit first** |
| `facebook/mms-tts-kor` | TTS (KO) | **CC-BY-NC 4.0** | **checked 2026-10-07** | **No** |
| espeak-ng + data | phonemizer for VI/EN/ZH | GPL-3.0 | unverified | Open legal question, see `docs/decisions/piper-vi-voice-license-trap.md` |

## Blockers before any public redistribution or commercial use

1. **Korean TTS is non-commercial.** AGENTS.md and `models/tts/ko/MANIFEST.md`
   previously said MIT; the model card says CC-BY-NC 4.0 (the MMS code is MIT,
   the weights are not). Fine for the hackathon demo. Needs a replacement
   (a Korean voice with a commercial license, or a self-trained VITS) before
   the product ships.
2. **The model bundle mixes licenses.** `scripts/upload_hf.py` uploads all of
   `models/` to one HF repo. Keep that repo private until the NC and GPL items
   above are resolved or the bundle is split by license.
3. **Attribution.** CC-BY components (Piper VI, Opus-MT if CC-BY) require credit
   in the shipped product or docs.
