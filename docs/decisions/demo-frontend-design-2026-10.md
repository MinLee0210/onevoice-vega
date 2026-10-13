# Demo front-end — design & tech-stack sketch (2026-10-13)

Goal: a **judge-facing demo UI** for the OneVoice/Vega engine that runs the
**real on-device Rust pipeline** (not a Python reimplementation), works
**fully offline** (airplane mode), and shows both carriers — **Vega Hand**
(one-to-one PTT) and **Vega Air** (one-to-many broadcast).

This is a sketch to agree on before building. Sources are listed at the end.

---

## 1. The one decision that drives everything

**Does the demo run the shipping Rust engine, or reimplement the pipeline in
the browser?**

- The pitch is "*the actual engine, offline, on a Qualcomm chip*". If the demo
  re-implements ASR→MT→TTS in JS/WASM, a sharp judge will (rightly) call it a
  different system, and the latency numbers won't match the CLI.
- So: **drive the real Rust engine.** The UI is a shell (webview) around the
  existing `Pipeline`/`Broadcast` code. This also means one codebase, and the
  demo *is* the product's runtime.

That rules out Gradio/Streamlit (Python, not the engine) as the primary demo.

---

## 2. Options considered

| # | Approach | Runs real engine? | Offline | Effort | Verdict |
|---|---|---|---|---|---|
| A | **Tauri v2 desktop app** (Rust backend links the engine; web UI) | ✅ | ✅ | M | **Primary** |
| B | **Local axum server + browser UI** (device hosts it over LAN) | ✅ | ✅ (LAN only) | M | **Secondary — for the Rubik Pi / no-screen demo** |
| C | **WASM in browser** (transformers.js + onnxruntime-web) | ❌ (re-impl) | ✅ | M–H | Stretch — "try it in your browser" marketing page only |
| D | Electron | ✅ | ✅ | M | ✗ heavier than Tauri for no benefit |
| E | Gradio/Streamlit | ❌ (Python stack) | ✅ | L | ✗ internal debugging only, never the judged demo |
| F | Flutter/mobile | ✅ | ✅ | H | ✗ overkill; no device build target yet |

**Recommendation: A for the demo; B for the device; C only if we want a
public link.** A and B share the same engine API, so building A gives B almost
free (swap the IPC transport).

---

## 3. Recommended architecture (Option A)

```
┌───────────────────────── Vega Console (Tauri v2, single binary) ─────────────────────────┐
│  WebView (React + Vite)                         Rust core (the onevoice crate as a lib)   │
│  ┌───────────────┐   invoke(cmd, {pcm})  ┌────────────────────────────────────────────┐   │
│  │ PTT button /  │ ─────────────────────▶│  Engine::process_utterance(pcm, src, tgt)  │   │
│  │ Space key     │                        │    Denoise → VAD → ASR → MT(gloss) → TTS   │   │
│  │ Waveform)     │ ◀───── Channel (timings)│  → returns WAV bytes + per-stage telemetry │   │
│  │ Latency HUD   │ ◀───── command result ──│                                            │   │
│  │ Result cards  │   play(WAV) via WebAudio└────────────────────────────────────────────┘   │
│  └───────────────┘                                                                         │
│  Mic: getUserMedia(48 kHz, 2ch) → AudioWorklet → i16 WAV (in a Worker)                     │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

Key points:
- **Mic capture stays in the browser** (Web Audio `getUserMedia` +
  `AudioWorklet`, 48 kHz — RNNoise's required input rate), so no cpal/ALSA
  port needed for the demo. PCM is sent to Rust as one WAV blob per PTT press.
- **Engine returns PCM/WAV**; playback via Web Audio. For long broadcast
  outputs, stream chunks with a Tauri **Channel** so Air mode plays the first
  language while later ones synthesize.
- **Telemetry** (per-stage ms, ASR confidence, glossary hits) streamed to the
  HUD via Channel/events — showcases the "latency engineering" differentiator.

### Required engine change (small)
The current runtime is a **binary** with file-based `audio_io`. To embed it:
1. Add `src/lib.rs` exposing a small `Engine` API:
   `process_utterance(pcm_i16_48k, source, target) -> Result<CycleResult>`
   and `broadcast(pcm, source, targets) -> Result<Vec<BroadcastLeg>>`, where
   `CycleResult { wav: Vec<u8>, transcript, translation, telemetry }`.
2. Keep `src/main.rs` as the CLI (it just calls the lib) so `run_pipeline.sh`
   and CI keep working unchanged.
3. Move WAV encoding/decoding behind the lib (reuse `audio_io` helpers).

This is a ~1-day refactor and benefits everything (tests, server mode).

---

## 4. Option B — LAN server for the Rubik Pi

Same `Engine`, wrapped in:
- `axum` (HTTP) + `tokio-tungstenite` (WebSocket) + `tower-http` (serve the
  built SPA). Routes: `GET /` (UI), `POST /utterance` (WAV in → WAV+telemetry
  out), `WS /air` (stream broadcast legs).
- The device runs this locally; a **phone/tablet browser** on the same LAN is
  the screen + mic + speaker. This is genuinely offline (no internet) and
  matches the handheld-with-no-display reality.
- Same frontend bundle as Option A (it's just HTTP instead of Tauri `invoke`).

---

## 5. Tech stack (per layer)

| Layer | Choice | Why |
|---|---|---|
| App shell | **Tauri v2** (Rust; MIT/Apache) | tiny binaries (uses OS webview), first-class Rust IPC, cross-platform, offline. Vite/React templates built in |
| Frontend | **React 19 + TypeScript + Vite** | fastest iteration; huge ecosystem; Tauri's documented default |
| Styling | **Tailwind CSS + shadcn/ui** | fast, consistent, accessible |
| State | **Zustand** | minimal, no boilerplate |
| Animation | **Framer Motion** | the listening→translating→done state machine reads well |
| Waveform | **Wavesurfer.js** | mic + playback visualisation with little code |
| Icons | **Lucide** | clean |
| Mic capture | **Web Audio API** (`getUserMedia` + `AudioWorklet`), WAV i16 in a **Web Worker** | 48 kHz capture without native audio code; keeps the UI thread free |
| Playback | **Web Audio API** (`AudioBufferSourceNode`) | sub-100 ms start; enables chunked Air playback |
| IPC | Tauri **commands** (typed request/response) + **Channel** (PCM/timings stream) + **events** (state) | commands are type-safe; Channel is the documented path for streaming data |
| Engine | existing Rust crate (`Pipeline`, `Broadcast`) via a new `src/lib.rs`; **ORT CPU** now, **QNN EP** on the Pi | runs the shipping code; matches CLI latency |
| LAN server (B) | **axum** + `tokio-tungstenite` + `tower-http` | same engine, browser transport |
| Packaging | `tauri build` → `.app`/`.deb`/`.AppImage`; Pi ships CLI + server + UI | one command per platform |
| Optional (C) | **transformers.js** + **onnxruntime-web** (WebGPU), `whisper-web`, `text-to-speech-webgpu`, `react-translator` | reusable browser demos for a public "try it" page; **not** the judged demo |

---

## 6. IPC / data contract

Tauri command (typed):
```
invoke('process_utterance', {
  pcm: ArrayBuffer,        // 48 kHz mono i16 WAV bytes
  source: 'vi', target: 'en',
  glossary: true           // enforce industrial terms
}) -> {
  wav: ArrayBuffer,        // synthesized translation (target voice)
  transcript: string,      // ASR text
  translation: string,     // MT text
  confidence: number,      // ASR confidence (gate signal)
  glossaryTerms: string[], // which termbase terms were forced
  telemetry: { captureMs, denoiseMs, vadMs, asrMs, mtMs, ttsFirstChunkMs, ttfaMs, totalMs }
}
```
Air mode:
```
invoke('broadcast', { pcm, source:'vi', targets:['en','zh','ko'] })
  -> a Channel<BroadcastLeg> where BroadcastLeg = { lang, translation, wav, ms }
```
Registers in `src-tauri/src/lib.rs` via `#[tauri::command]` +
`generate_handler![]`; binary PCM via `tauri::ipc::Response`. The engine work
runs in an **async command** so the UI never freezes.

---

## 7. UI / UX sketch

Single window, two modes (Hand / Air), everything offline.

```
┌───────────────────────────────────────────────────────────────────────┐
│  VEGA  ▸ [ Hand ]  [ Air ]                       ● Offline  ▰▰▰▱ 82%   │  ← airplane-mode badge
├───────────────────────────────────────────────────────────────────────┤
│   VI  ⇄  EN        (source/target selectors; flags, swap)              │
│                                                                         │
│      ╭───────────────────────────────────────────────────────────╮     │
│      │            [ HOLD TO TALK  ·  or press Space ]            │     │  ← big glove-friendly PTT
│      ╰───────────────────────────────────────────────────────────╯     │
│      ▁▂▅█▇▅▂▁▁▂▃▅▇█▅▃▁  (live input waveform, Wavesurfer)          │
│                                                                         │
│   ▸ transcript (vi)      "Xe nâng đang đến gần khu vực số ba."        │
│   ▸ translation (en)     "forklift approaching sector three"  🔒forklift│  ← glossary chip
│   ⏱  TTFA 0.61s   ASR 0.41 · MT 0.09 · TTS 0.11   conf 0.97           │  ← latency HUD
│   [ ▶ replay ]   [ copy ]   [ ★ safety phrase ]                       │
├───────────────────────────────────────────────────────────────────────┤
│  Air:  [ EN ]▶  [ ZH ]  [ KO ]   broadcast says it in order…          │
└───────────────────────────────────────────────────────────────────────┘
```

States (Framer Motion): `idle → listening (pulsing red) → translating
(spinner + stage chips lighting up) → speaking (waveform of output) → done`.
Error states: **no-speech** (VAD empty) and **low-confidence** (the new ASR
gate) show a distinct amber "Didn't catch that — try again" instead of a
translation — a safety story judges remember.

Language-pair chips: VI↔EN, VI↔ZH, VI↔KO (6 directions), with a small "pivot
via EN" hint on the ZH/KO pairs.

Air mode: target-language toggles + an animated broadcast queue; each leg's
transcript + a per-leg time; a "would play through the horn speaker" note.

Extras that map to differentiators:
- **Glossary** toggle + which terms were enforced (🔒 chip).
- **Latency HUD** with per-stage bars (from telemetry) — the "latency HUD for
  demos" bullet in AGENTS.md.
- **Offline badge** that flips to red if any network call is attempted (there
  are none) — the airplane-mode proof.
- **Haptic/LED simulation** for the Hand form factor (state → vibration
  pattern / LED colour) — the unclaimed differentiator.

---

## 8. Judge-facing demo script (2 min)

1. Put the machine in **airplane mode**, show the badge.
2. Hand mode, VI→EN: hold PTT, say a safety instruction, release; show
   transcript → translation + **glossary term forced** + TTFA.
3. Switch to VI→ZH (show "via EN pivot"); repeat.
4. Toggle **Air**; speak one evacuation instruction; show EN/ZH/KO broadcast
   legs playing in sequence.
5. Show the **latency HUD** and the **no-speech/low-confidence** state by
   staying silent / coughing.
6. One sentence on the CLI/golden CI as evidence this is the shipping engine.

---

## 9. Build plan

| Step | Work | Depends on |
|---|---|---|
| 0 | `src/lib.rs` engine API (`process_utterance`, `broadcast`) + keep CLI | — (1 day) |
| 1 | `create-tauri-app` (React+TS+Vite); blank window; wire ONE command (file-based first) | 0 |
| 2 | Web Audio mic capture (48 kHz AudioWorklet) + WAV encode; PTT (button + Space) | 1 |
| 3 | Result cards + latency HUD from telemetry | 1–2 |
| 4 | Waveform (Wavesurfer), state machine (Framer Motion), language selectors | 3 |
| 5 | Air mode (Channel streaming) | 3 |
| 6 | Glossary chips, offline badge, LED/haptic sim | 4–5 |
| 7 | `axum` LAN-server variant reusing the SPA | 0, 4 |
| 8 | Optional: WASM marketing page (transformers.js) | — |
| 9 | Package: `tauri build`; Pi image = CLI + server + UI | 7 |

Milestone “demoable” = steps 0–4 (Hand, real engine, HUD). Air = step 5.
Pi/wireless = step 7.

---

## 10. Risks & mitigations

- **Tauri webview on Linux/Pi** — WebKitGTK needs GPU/GL; on a headless Pi the
  *server* mode (B) avoids the webview entirely. Test early on the Pi.
- **Browser sample-rate** — some devices only give 44.1 kHz; resample to 48 kHz
  in the AudioWorklet before encoding (RNNoise needs 48 kHz).
- **Big binary / model loading time** — model load is one-time; show a
  "warming up" progress (Tauri events) so the first PTT isn't the slow one.
- **Air model set is larger** — lazy-load target TTS voices; only 3 targets.
- **Latency mismatch vs CLI** — same engine, so numbers should match; the
  browser mic/playback path adds a little — measure and report honestly.
- **DO NOT** put the demo behind a cloud host "for convenience" — it breaks the
  whole offline premise and the judging rule (any internet dependency =
  disqualification).

---

## 11. Sources

- Tauri v2 architecture / calling Rust (commands, events, Channels,
  `ipc::Response`, async commands): https://v2.tauri.app/concept/architecture/ ,
  https://v2.tauri.app/develop/calling-rust/
- ONNX Runtime Web (onnxruntime-web, WASM/WebGPU/WebNN, offline, large-model
  note): https://onnxruntime.ai/docs/tutorials/web/
- Transformers.js (pipelines, WASM/WebGPU, ASR + TTS tasks, example apps:
  `whisper-web`, `realtime-whisper-webgpu`, `text-to-speech-webgpu`,
  `react-translator`): https://huggingface.co/docs/transformers.js/en/index ,
  https://github.com/huggingface/transformers.js-examples
- Piper (archived; dev moved to `OHF-Voice/piper1-gpl` — note the GPL shift,
  relevant to the licensing trap): https://github.com/rhasspy/piper
- Existing repo assets to reuse: `src/pipeline.rs`, `src/broadcast.rs`
  (`--broadcast`), `src/stages/*`, `configs/`, `data/glossary/`.
