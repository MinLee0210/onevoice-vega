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

- ✅ Step 0 done: `onevoice::Engine` in-memory API (WAV in/out + telemetry).
- ✅ This scaffold: Hand mode — mic capture, PTT (button + Space), result
  cards, latency HUD, error/no-speech/low-confidence states.
- ⏳ Not yet: **Air mode** (broadcast EN/ZH/KO via a Tauri Channel), waveform
  visualisation, glossary chips, LED/haptic simulation, LAN-server variant
  (see the design doc's build plan, steps 5–9).
- Note: mic capture uses the deprecated `ScriptProcessorNode` (simple, works);
  the production path is an `AudioWorklet`.

## Notes

- The engine needs the full model set; first cycle loads models (~seconds),
  then it's warm. The HUD timings are per-stage ms from Rust.
- `pcm_wav` currently crosses IPC as a JSON number array (fine for ~0.2 MB);
  switch to a raw `ipc::Response`/`Channel` for larger/streamed payloads.
