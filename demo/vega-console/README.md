# Vega Console

Judge-facing desktop console for the real, local OneVoice Rust engine. It is a
Tauri shell around `onevoice::Engine` and `BroadcastEngine`; it does not run a
browser or cloud translation stack.

## What it demonstrates

- **Vega Hand:** PTT one-to-one translation for the valid VI↔EN/ZH/KO routes.
- **Vega Air:** one Vietnamese operator instruction, played sequentially in
  EN, ZH, and/or KO.
- **Safety evidence:** ASR confidence gates, enforced-glossary chips, and
  per-stage engine telemetry.
- **Typed speech:** enter a short reply and speak it directly with the local
  VI/EN/ZH/KO voice, without recording or translating it.
- **Offline operation:** all model inference is invoked locally through Tauri.

## Run

From this directory, after installing the project model bundle:

```bash
npm install
npm run tauri dev
```

Set `ONEVOICE_ROOT` if the model/config/data tree is not three directories
above this app. A bundled Vietnamese WAV fixture is available through **Run VI
Sample**, so a microphone is not required for a repeatable demo.

For a distributable desktop build:

```bash
npm run tauri build
```

## Interaction notes

- Hold the large PTT control (mouse, touch, pen, or Space), speak, then
  release. Captured audio is normalized to mono 48 kHz before it enters the
  RNNoise front end.
- Choose **Text to speech** at the top of the workspace, enter up to 280
  characters, select the voice language, and press **Speak this text aloud**.
  The spoken reply can be played again. After a voice translation, use
  **Speak this translation again** inside its output panel.
- Air playback is deliberately sequential. The active queue row marks the
  language currently playing through the simulated horn output. After the
  sequence finishes, each language row has a **Speak** button to replay only
  that translated WAV.
- **To playback** measures capture-release to Web Audio scheduling in this
  console. **Engine first chunk** is a separate internal timing: the current
  Tauri command still returns a completed WAV, so it is not presented as the
  user-perceived speaker-start time.

The next latency improvement is streaming TTS chunks across Tauri; the UI is
already labelled so that this distinction remains honest in a judging demo.
