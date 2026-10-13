# demo/vega-console — Tauri desktop demo

A tiny desktop UI that drives the **real on-device Rust engine**
(`onevoice::Engine`) — no Python, no cloud. Hold to talk (or Space), speak,
and see the transcript → translation, a glossary/latency HUD, and the
synthesized audio played back. Design: `docs/decisions/demo-frontend-design-2026-10.md`.

- **Frontend**: React 19 + TypeScript + Vite (`src/`).
- **Backend**: Tauri v2 (`src-tauri/`) with two commands — `ready` and
  `process_utterance` (WAV bytes in → translated WAV + telemetry out).
- **Mic/playback**: Web Audio (48 kHz mono, encoded to WAV client-side).

## Run

Prereqs: `models/` populated (from the repo root: `scripts/setup_models.sh`),
Node 20+, Rust, and a desktop with a WebView (macOS WKWebView, Linux WebKitGTK).

```sh
cd demo/vega-console
npm install
npm run tauri dev
```

The Rust side resolves the repo root automatically (this crate's
`CARGO_MANIFEST_DIR`/`../../..`). Override with `ONEVOICE_ROOT=/path/to/repo`
if you run from elsewhere or ship a bundle.

Build a distributable: `npm run tauri build` (outputs under
`src-tauri/target/release/bundle/`).

## Status / next

- ✅ Step 0: `onevoice::Engine` + `BroadcastEngine` in-memory APIs (WAV in/out
  + telemetry) — `src/lib.rs`.
- ✅ Hand mode: mic capture, PTT (button + Space), transcript/translation
  cards, **glossary chips**, latency HUD, error/no-speech/low-confidence states.
- ✅ Air mode: one utterance → EN/ZH/KO legs, each shown + played in sequence
  (broadcast-tuned slower TTS).
- ⏳ Not yet: waveform visualisation, LED/haptic simulation, LAN-server variant
  (see the design doc's build plan, steps 4, 6, 7).
- Note: mic capture uses the deprecated `ScriptProcessorNode` (simple, works);
  the production path is an `AudioWorklet`.

## Input audio (no mic needed)

The **Input audio** row has two buttons that run the engine without a mic:
**⬆ Load audio file** (any sample rate — it's decoded and resampled to
48 kHz client-side) and **▶ Use sample (VI)** (the committed eval fixture).
Use these to check the UI, or when the mic is unavailable.

## Microphone

WKWebView (Tauri's macOS webview) only exposes `navigator.mediaDevices` in a
**secure context**, so **the mic does not work in `tauri dev`** (plain
`http://localhost`) — you'd see `navigator.mediaDevices` undefined. It works in
a **bundled build** (`npm run tauri build`, served from `tauri://localhost`),
which needs `src-tauri/Info.plist`'s `NSMicrophoneUsageDescription` (included).
So: use **Input audio** while developing; use the built app for live PTT.

## Notes

- The engine needs the full model set; first cycle loads models (~seconds),
  then it's warm. The HUD timings are per-stage ms from Rust.
- `pcm_wav` currently crosses IPC as a JSON number array (fine for ~0.2 MB);
  switch to a raw `ipc::Response`/`Channel` for larger/streamed payloads.
