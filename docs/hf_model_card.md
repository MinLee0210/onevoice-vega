---
license: other
license_name: onevoice-vega-mixed
license_link: https://huggingface.co/LakoreAI/onevoice-vega#license-note
language:
  - vi
  - en
  - zh
  - ko
pipeline_tag: audio-to-audio
library_name: onnxruntime
tags:
  - onnx
  - speech-translation
  - automatic-speech-recognition
  - text-to-speech
  - translation
  - vietnamese
  - edge
  - qualcomm
  - dragonwing
  - qcs6490
  - cascaded
  - industrial
  - noise-robust
base_model:
  - huuquyet/PhoWhisper-base
  - onnx-community/whisper-small
  - Helsinki-NLP/opus-mt-vi-en
  - Helsinki-NLP/opus-mt-en-vi
  - rhasspy/piper-voices
  - facebook/mms-tts-kor
  - onnx-community/silero-vad
model-index:
  - name: OneVoice-Vega
    results:
      - task:
          type: automatic-speech-recognition
          name: Vietnamese ASR (PhoWhisper-base)
        dataset:
          name: VIVOS
          type: vivos
        metrics:
          - type: wer
            value: 2.2
            name: WER (clean speech)
      - task:
          type: translation
          name: Vietnamese-to-English industrial glossary enforcement
        dataset:
          name: OneVoice industrial glossary carrier set
          type: onevoice-glossary
        metrics:
          - type: accuracy
            value: 100.0
            name: glossary term accuracy
---

# OneVoice / Vega — on-device speech translation model bundle

The full **ONNX model bundle** for QUAVIO/Vega, a 100% offline (no-cloud)
speech-to-speech translator for noisy industrial environments, built for the
Saigon AI Hub × Qualcomm "OneVoice AI Challenge". One cascaded engine
(`denoise → VAD → ASR → MT → TTS`) shipped on the **Qualcomm Dragonwing
QCS6490** (Rubik Pi 3 / RB3 Gen 2), in two carriers:

- **Vega Hand** — ruggedized push-to-talk handheld (one-to-one).
- **Vega Air** — drone-mounted multilingual broadcast payload (one-to-many).

Languages: **Vietnamese ↔ English / Chinese (Mandarin) / Korean** (6 directions).

This repo holds the `.onnx` artifacts (and Piper's `espeak-ng-data` phoneme
tables) that the Rust runtime (`src/`) loads via ONNX Runtime / QNN. It is the
same tree `scripts/pack_models.sh` ships as the GitHub Release bundle and that
`scripts/setup_models.sh` installs.

## Models

| Stage | Model | Size | License | Source |
|---|---|---|---|---|
| VAD | Silero VAD | 2.1 MB | MIT | `onnx-community/silero-vad` |
| ASR (VI) | PhoWhisper-base (int8) | 22 MB enc / 51 MB dec | BSD-3-Clause | `huuquyet/PhoWhisper-base` |
| ASR (EN/ZH/KO) | Whisper-small (int8) | 88 MB enc / 149 MB dec | MIT | `onnx-community/whisper-small` |
| MT (VI↔EN) | Opus-MT (INT8) | 45 MB enc / 78 MB dec | Apache-2.0 / CC-BY | `Helsinki-NLP/opus-mt-*` |
| TTS (VI) | Piper `vais1000-medium` (VITS) | 60 MB | CC BY 4.0 | `rhasspy/piper-voices` |
| TTS (EN) | Piper `lessac-medium` (VITS) | 60 MB | CC BY 4.0 | `rhasspy/piper-voices` |
| TTS (ZH) | Piper `huayan-medium` (VITS) | 60 MB | CC BY 4.0 | `rhasspy/piper-voices` |
| TTS (KO) | MMS-TTS Korean (VITS) | 109 MB | MIT | `facebook/mms-tts-kor` |
| TTS runtime data | `espeak-ng-data` (Piper phonemes) | 19 MB | **GPL-3.0** | `piper-tts` PyPI package |

Denoise (RNNoise) has no model file — its weights are compiled into the Rust
`nnnoiseless` crate. Encoders/decoders use the merged-decoder KV-cache ONNX
layout (`decoder_model_merged.onnx`) for autoregressive greedy decode.

## Repository structure

```
├── vad/model.onnx
├── asr/vi/{encoder_model.onnx, decoder_model_merged.onnx, tokenizer.json, ...}
├── asr/en-zh-ko/{...}
├── mt/vi-en/{encoder_model.onnx, decoder_model_merged.onnx, tokenizer.json, ...}
├── mt/en-vi/{...}
├── tts/{vi,en,zh,ko}/*.onnx (+ Piper *.onnx.json sidecars)
└── tts/espeak-ng-data/       # Piper phoneme data (vi/en/zh)
```

Each directory carries a `MANIFEST.md` recording source checkpoint, export
date, quantization scheme, and eval metrics.

## Usage

```bash
git lfs install
git clone https://huggingface.co/LakoreAI/onevoice-vega models/

# run the full Rust pipeline (denoise → VAD → ASR → MT → TTS)
# see the project repo: https://github.com/MinLee0210/onevoice-vega
scripts/run_pipeline.sh data/eval/ptt_sample_vi_48k.wav out.wav -- --source vi --target en
```

Or download only what `setup_models.sh` needs from the project repo — it
points at this bundle as the "rebuild" source for `vad`/`asr`/`tts`.

## Performance

Measured on the QCS6490 (Hexagon v68 HTP), full-INT8 QNN context binaries —
see the companion `qualcomm-ai-hub-community/whisper-*-qcs6490-qnn-int8` repos:

| Stage | Latency |
|---|---|
| ASR encoder (PhoWhisper-base, 30 s) | ~49.5 ms (NPU) |
| ASR decoder (per token) | ~12.1 ms (NPU) |
| End-to-end pipeline (CPU dev machine) | ~741 ms button-release → audio |

## License note

This is a **mixed-license bundle** — each stage carries its own license (table
above). In particular, `espeak-ng-data/` is **GPL-3.0** (sourced from the
`piper-tts` package); embedding it in a distributed binary raises an
unresolved GPL question, relevant only to commercial distribution, not to the
prototype/demo. See `docs/decisions/piper-vi-voice-license-trap.md` in the
project repo for the full analysis. The Korean TTS path (MMS-TTS, MIT) avoids
espeak entirely.

## Attribution

- Whisper — OpenAI (MIT); PhoWhisper — VinAI / huuquyet.
- Opus-MT — Helsinki-NLP (Apache-2.0).
- Silero VAD — Silero Team (MIT).
- Piper voices — rhasspy/piper-voices; espeak-ng data from piper-tts.
- MMS-TTS — Meta (MIT).

Project: [MinLee0210/onevoice-vega](https://github.com/MinLee0210/onevoice-vega).
