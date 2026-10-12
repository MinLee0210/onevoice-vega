# AGENTS.md — QUAVIO/Vega (OneVoice AI Challenge)

## Project
On-device (no cloud) speech translation for noisy industrial environments
(factories, construction, logistics, hospitals), built for the Saigon AI
Hub × Qualcomm "OneVoice AI Challenge." One translation engine, deployed
in two carriers that share the same QCS6490 silicon, models, and runtime:

- **Vega Hand** — ruggedized push-to-talk handheld, one-to-one exchanges
  on the floor (the original single-device design).
- **Vega Air** — drone-mounted multilingual broadcast payload, one-to-many
  (a single spoken instruction broadcast in VI/EN/ZH/KO simultaneously).

Phase 2 technical spec was submitted 03/08/2026 (see
docs/reports/TECHNICAL PROPOSAL_ QUAVIO_Vega.pdf). Phase 3 (prototype
build) is next.

## Languages/pairs
VI↔EN, VI↔ZH, VI↔KO (6 directions total)

## Conversation model
Half-duplex, push-to-talk (PTT). One party holds button, speaks, releases;
device translates and plays back. No full-duplex streaming needed —
process complete utterances (≤10s), not partial/incremental audio.

Vega Air is broadcast (operator-initiated), not conversational: the
operator speaks into the ground station, and the payload plays the
translation back in each target language sequentially.

## Hardware target
- Prototype: Rubik Pi 3 (Qualcomm Dragonwing QCS6490), 8-core Kryo 670 CPU,
  Hexagon NPU @ 12 TOPS, 8GB LPDDR4x, 128GB UFS
- **Vega Hand** — ruggedized handheld/neckband, IP54+, physical PTT
  button, LED status (no touchscreen), ~140×60×25mm, ≤300g, ≥8h on 5,000mAh
- **Vega Air** — payload pod (~120×70×50mm, ≤300g, IP54) under a DJI
  Matrice-class multirotor; directional horn speaker intelligible at
  20–30m; MAVLink/UART flight-controller telemetry feed for the ego-noise
  model. Broadcast only in the primary mode (source audio is captured
  cleanly at the ground station — the hard acoustic problem is *output*
  intelligibility, not input); two-way capture under rotor wash is a
  stretch.

## Compute split
```
CPU (Kryo)     → audio I/O, ring buffer, RNNoise, Silero VAD,
                 orchestration, Piper TTS (runs fine on CPU)
Hexagon NPU    → ASR encoder/decoder, MT encoder/decoder
                 (the heavy transformer matmuls, quantized)
```

## Pipeline
```
Mic array → Denoise (RNNoise/DeepFilterNet3) → VAD trim (Silero)
  → ASR → MT → TTS → Speaker
```
Denoise runs during recording (zero added latency). TTS streams output
(start playback before full synthesis completes) to hide latency.

RNNoise's feature/band structure is hardcoded to 48kHz — confirmed while
implementing it (`src/stages/denoise.rs`), not just a spec assumption.
Mic capture must run at 48kHz; denoise resamples down to 16kHz for
VAD/ASR on its way out. Not a free architectural choice, a constraint
this pipeline has to design around.

## Model stack

| Stage | Baseline | Upgrade path |
|---|---|---|
| Denoise | RNNoise via the `nnnoiseless` Rust crate (BSD-3-Clause, weights compiled in — RNNoise has no official ONNX export, unlike every other stage) | DeepFilterNet3 (better on machinery noise). Has a real pre-exported ONNX (MIT), but it takes DeepFilterNet's own ERB/STFT features as input, not raw audio — needs its own feature-extraction implementation, not a drop-in swap. See docs/decisions/pipeline-improvement-research-2026-07.md. |
| VAD | Silero VAD (MIT, ONNX) — validated | — |
| ASR (VI) | PhoWhisper-base 74M (BSD-3-Clause) — validated, WER 2.19% clean / 9.84% @5dB (VIVOS, 20 utts) | **CTC backend now implemented** (`src/stages/ctc.rs`, auto-selected by `asr_model_dir`); measured `wav2vec2-base-vi` is ~6× faster (ASR 57 ms vs 360 ms) but 10.4% WER — Whisper stays default. Still open: a *better* VI CTC checkpoint (VietASR 13M). See docs/reports/asr_ab.md |
| ASR (EN/ZH/KO) | Whisper-Small — **not** via AI Hub pre-quantized (that path is currently broken for QCS6490, see Hardware toolchain below); use a generic pre-made ONNX instead (e.g. onnx-community/whisper-small, MIT) | Whisper-Large-v3-Turbo quantized |
| MT | Opus-MT per-pair (~300MB, Apache/CC-BY, INT8) — all 6 directions wired: vi-en/en-vi direct, VI↔ZH/KO via EN pivot (`MtChain`); Rust port tested end-to-end | LoRA fine-tune on industrial glossary; replace broken en→ko checkpoint (tc-big-en-ko BLEU 0.1) or use M2M-100 for KO |
| TTS | All 4 target langs (device is bidirectional): Piper (VITS/ONNX, CPU real-time) for VI (`vais1000-medium`) / EN (`lessac`) / ZH (`huayan`, Mandarin); MMS-TTS (`facebook/mms-tts-kor`, **CC-BY-NC 4.0** — non-commercial, see THIRD_PARTY_LICENSES.md; VITS/ONNX) for KO — Piper has no Korean voice. **Real Rust implementation, not a stub** — `piper-rs` crate for VI/EN/ZH (VITS + espeak-ng phonemization), hand-rolled `uroman` + ported char-tokenizer + raw `ort` inference for KO (no crate covers MMS). All 4 produce genuine audio, round-trip-ASR verified through the real Rust path, not just the Python notebook. See license trap below (not a hackathon blocker) and `src/stages/tts.rs` module docs for the `piper-rs` build gotcha | Kyutai Pocket TTS once it supports Vietnamese (English-only as of this writing, checked directly) |

**License trap:** NLLB is CC-BY-NC — avoid for anything meant to be
commercialized. MADLAD-400 (Apache 2.0) is broad-coverage but likely too
big (3B) for comfortable NPU deployment — treat as a stretch option, not
baseline. Piper's Vietnamese voices (rhasspy/piper-voices): only
`vais1000-medium` is CC BY 4.0 (usable); `vivos-x_low` is CC-BY-NC-SA
(blocked) and `25hours_single-low` is unlicensed — don't substitute
either. Separately, Piper's phonemizer (`piper-phonemize`/`espeak-ng`,
used for VI/EN/ZH) is GPL-3.0, and whether embedding it in the shipped
binary GPL-taints the product is an unresolved legal question — **but
this is a commercialization question, not a hackathon blocker** (GPL
binds at distribution, not a prototype/demo); revisit before Phase 3+
commercial commits. The Korean path (MMS-TTS) avoids espeak entirely but its weights are
CC-BY-NC 4.0 (checked 2026-10-07; earlier notes said MIT, which was wrong), so
it needs a commercially licensed replacement before the product ships. Full detail: docs/decisions/piper-vi-voice-license-trap.md.

## Memory budget (quantized, resident)
~2GB total. Lazy-load only the active language pair's MT+TTS to stay
comfortable on 8GB device RAM.

## Latency target
≤1.2s button-release → first audio (stretch: 0.8s). ASR decoder
(autoregressive) is the bottleneck — keep models small (base/small tier,
not medium/large), greedy/beam=1 decode for MT.

## Performance targets
```
End-to-end latency (release→first audio)   ≤ 1.2s
ASR WER, VI, 75dB factory noise            ≤ 15%
MT BLEU VI↔EN (PhoMT + domain test set)    ≥ 40
Glossary term accuracy                     ≥ 95% enforced
Battery                                    ≥ 8h continuous
Connectivity required                      zero (airplane-mode demo)
```

## Differentiators (write into every proposal/pitch)
1. **Two communication shapes, one engine** — one-to-one (Hand) and
   one-to-many multilingual broadcast (Air) from one codebase, one BOM,
   one model set. No competitor spans both, and no drone loudspeaker does
   on-device translation at all (see
   docs/decisions/vega-air-and-feature-research-2026-08.md).
2. **Industrial glossary engine** — VI/EN/ZH/KO parallel termbase
   (safety commands, machine names, logistics terms). Enforcement today
   is a working decoder-logit-bias technique (validated vi-en only,
   100% term accuracy on a 10-sentence test set vs. 50% baseline, at a
   real but acceptable fluency cost — see
   docs/decisions/glossary-runtime-enforcement-2026-07.md), shipped as
   the Phase 2 stopgap. LoRA fine-tune remains the intended Phase 3
   upgrade for better fluency. Swappable per-industry vocab packs.
3. **Noise-first audio front-end** — 2-mic beamforming + denoise,
   benchmarked WER-vs-SNR curve at 70-85dB factory noise. iFlytek (a real
   competitor) uses a 4-mic array — worth a direct benchmark once hardware
   exists in case 2 mics turns out to be optimistic, not yet a reason to
   change the spec. Note: iFlytek *does* list Vietnamese among its 18
   offline languages, so the pitch should claim "Vietnamese-first" and
   "Vietnamese *industrial*", not "nobody supports offline VI".
4. **Latency engineering** — denoise-during-capture, greedy MT decode,
   streaming TTS playback, instrumented per-stage timing / live latency
   HUD for demos.
5. **Haptic/vibration feedback** — unclaimed by every competitor
   researched; cheap, glove-friendly state signaling (listening /
   translating / done) without looking at an LED.

## Tooling / language split
```
Model eval, fine-tuning, quantization  → Python
  (PyTorch, HF Transformers, Qualcomm AI Hub Workbench, ONNX export)

On-device runtime (production loop)    → Rust (if team knows Rust) or C++
  - `ort` crate (Rust) or ONNX Runtime C++ API, QNN execution provider
  - cpal/hound for audio I/O (Rust) or ALSA (C++)
  - No Python in the real-time loop (GIL/overhead)
```
Handoff artifact between phases: the exported `.onnx` file (or QNN
context binary) — language-agnostic interface.

**Rust risk note — resolved for CPU, still open for QNN:** `ort` confirmed
working well in practice, not just in theory — real Rust implementations
exist and are validated end-to-end for denoise, VAD, MT, and ASR
(`src/stages/*.rs`), each tested against real ONNX models with exact
expected output. Every real bug hit along the way was a model/
preprocessing correctness issue, not an `ort` problem (see
docs/decisions/onnx-runtime-merged-decoder-gotchas.md and
docs/decisions/mel-spec-crate-whisper-compat-bugs.md). What's genuinely
still untested: the QNN execution provider itself on real Hexagon
hardware — `ort`'s `qnn` Cargo feature hasn't been enabled or exercised
at all yet, everything so far runs on CPU. Smoke-test that specifically
in week 1 of Phase 3 hardware bring-up; don't assume CPU-validated Rust
code just works once QNN is turned on.

## Hexagon / QNN glossary (for reference)
- **Hexagon** = dedicated low-power DSP-turned-NPU chip block on the SoC
  (separate from CPU/GPU), evolved from Qualcomm's original signal
  processor. HVX (vector) + HMX (matrix/tensor) execution units.
- **HTP** = Hexagon Tensor Processor, the AI-execution part of Hexagon.
- **QNN SDK / QAIRT** = Qualcomm's software stack/compiler that takes a
  model and hands it to Hexagon's HTP backend. `libQnnHtp.so` /
  `QnnHtp.dll` is the concrete backend library referenced in code.
- Flow: `.onnx model → QNN SDK → HTP backend → runs on Hexagon silicon`

## Hardware toolchain
Qualcomm AI Hub (pre-optimized model exports) → AI Hub Workbench
(quantization/profiling) → QNN SDK/QAIRT (deployment). Rubik Pi 3 is
Raspberry-Pi-HAT compatible — use for mic array (ReSpeaker HAT) and PTT
button/LED prototyping.

**Whisper does follow this path now (resolved 2026-08).** The old blocker
(AIMET-ONNX, no aarch64 wheel, `ai-hub-models` #281) is moot:
`qualcomm-ai-hub-community` publishes pre-built full-INT8 QNN context
binaries for whisper-base/small on QCS6490, and the server-side recipe
(`--target_runtime qnn_context_binary --quantize_full_type int8
--quantize_io --truncate_64bit_io`) also works for any HF Whisper
fine-tune including `vinai/PhoWhisper-small`. See
docs/decisions/pipeline-performance-research-2026-08.md.

## Contest/business context
- Co-hosted with Qualcomm; part of Qualcomm's broader Vietnam AI R&D push
  (Hanoi/HCMC AI center opened June 2025, VinAI/MovianAI acquisition
  April 2025 — PhoWhisper/PhoMT lineage is now Qualcomm-owned talent).
- Judging from the prototype phase on (public Luma page, checked 2026-10-07):
  Technical Performance 50% (BLEU + COMET on a hidden benchmark, MOS for TTS
  naturalness, RTF < 1.0, end-of-speech → start-of-synthesis latency with > 2.0s
  penalised) / Innovation 25% / Business Viability 25%. Any internet
  dependency during testing = disqualification. See
  docs/decisions/deep-research-2026-10.md.
- Judging (Phase 2 rubric, superseded): AI Approach & Technical Design 35% / Hardware
  & Device 25% / Business Solution 15% / Problem & Impact 15% / Team &
  Execution 10%.
- Likely goal from Qualcomm's side: Dragonwing/QCS design wins, AI Hub
  toolchain adoption, reference showcase demos, a commercializable local
  product story (Vbee AI is cited as the existing precedent — Vietnamese
  startup building offline voice solutions on Qualcomm edge platforms —
  **but this specific claim is unverified**: competitor research could
  only confirm Vbee's cloud TTS SaaS product, not an edge-device/Qualcomm
  angle. Verify directly before repeating it in Phase 2/3 materials, see
  docs/decisions/competitor-landscape-2026-07.md).
- Strategy: frame device explicitly as "Dragonwing-native," document the
  AI Hub/Workbench porting journey, position business pitch around
  potential Qualcomm ecosystem partnership (not just cash prize).
- IP: QVIC (Qualcomm's other VN program) took no equity/IP — verify
  OneVoice's terms in FAQ before Phase 2 submission.

## Timeline
```
Phase 2 (Jul 2026, current)  → technical spec submission
Phase 3 (Aug-Sep 2026)        → prototype build
Phase 4 (Oct 2026)            → field testing (real factory/warehouse)
Phase 5 (Nov 2026)             → finale demo, HCMC
```

## Open decisions / risks to track
- MADLAD-single-model vs. Opus-MT-per-pair — memory vs. speed tradeoff,
  needs benchmarking on real device.
- PhoWhisper accuracy post-quantization — partially validated: pre-made
  int8 dynamic quantization (CPU) shows no measurable WER loss vs. fp32
  on a small VIVOS sample. Still not validated: full-INT8 on the actual
  Hexagon NPU (W8A16 fails on this HTP — see the RESOLVED note below);
  blind calibration-free INT8 trades some accuracy, so re-validate WER
  on domain before production.
- **QCS6490 Whisper/AI Hub deployment — RESOLVED (2026-08).** The old
  blocker (AI Hub's quantized Whisper needs AIMET-ONNX, no aarch64 wheel,
  `ai-hub-models` #281) is moot: `qualcomm-ai-hub-community` publishes
  pre-built full-INT8 QNN context binaries for whisper-base/small on
  QCS6490, with a server-side recipe
  (`--target_runtime qnn_context_binary --quantize_full_type int8
  --quantize_io`) that also works for `vinai/PhoWhisper-small`. Measured:
  whisper-base encoder 49.5 ms / decoder 12.1 ms-per-token on device.
  Still open: actually loading the context binary via `ort`'s `qnn`
  feature on real hardware (week-1 Phase 3 smoke test). See
  docs/decisions/pipeline-performance-research-2026-08.md.
- **Piper's Vietnamese TTS has a license trap and an unresolved GPL
  question** (phonemizer dependency) — see the model stack license trap
  note above and docs/decisions/piper-vi-voice-license-trap.md. Now
  actually shipping in `src/`'s real dependencies (not just a Python
  notebook), which is exactly the point this question needs a real
  legal answer before commercialization — the hackathon prototype
  proceeds regardless per the explicit call to defer legal to that
  later stage.
- VI↔KO / VI↔ZH quality (low-resource pairs) — **EN-pivot now
  implemented and measured** (docs/reports/mt_quality.md): VI→ZH pivot is
  fine (en-zh leg 46.5 BLEU), but the `tc-big-en-ko` en→ko checkpoint is
  broken (BLEU 0.1) so VI→KO is not viable yet — needs a different en→ko
  model or a multilingual model (M2M-100, MIT) for KO.
- **Glossary enforcement is real but only validated for vi-en, on 10
  test sentences, with a fluency cost.** Decoder-logit-bias technique
  (`src/stages/mt.rs`) hits 100% term accuracy vs. 50% baseline, but
  corrected translations are grammatically rougher than the baseline
  ("forklift approaching sector three." drops "The"/"is"). Not native-
  speaker reviewed, not benchmarked on BLEU, and the other 5 language
  pairs have no MT model ported yet so glossary bias is untested there.
  LoRA fine-tune (the original plan) is still the intended Phase 3
  upgrade. See docs/decisions/glossary-runtime-enforcement-2026-07.md.
- Field-test noise data scarcity — synthesize via VIVOS/Common Voice VI
  mixed with MUSAN/factory recordings at controlled SNRs.
- **Business model has no answer yet** — every competitor found
  (Weavix, Relay, Vasco, Pocketalk, ...) monetizes via subscription/
  data-plan tied to cloud connectivity, which this device's offline
  architecture structurally can't replicate. Needs an explicit answer
  for the Phase 2/3 pitch — business viability is 25% of judging. See
  docs/decisions/competitor-landscape-2026-07.md.
- **No genuinely-offline industrial PTT translator was found among
  competitors researched** — every real one (industrial or consumer)
  degrades to no-translation when disconnected. This makes "zero
  connectivity required" a real, differentiated claim worth leading
  with in the pitch, not just a spec line. Same doc as above.
- **Vega Air has no direct competitor.** No drone loudspeaker (DJI's
  Mavic 3 Enterprise Speaker, or the Shenzhen PSDK aftermarket: Puyu,
  CZI, Ramoton) does speech translation — "multilingual" there means TTS
  of pre-typed text. The 20–30m intelligibility bar is trivial vs. even
  DJI's weakest speaker (100m @70dB); the differentiation is entirely the
  on-device ASR→MT→TTS engine. See
  docs/decisions/vega-air-and-feature-research-2026-08.md.
- **Vega Air ego-noise (two-way stretch) is unproven.** RPM-harmonic
  subtraction from flight-controller telemetry is novel (no published
  precedent) and the literature warns that naive subtractive front-ends
  blow up streaming false-alarm rate ~10× — prefer *conditioning* the
  neural denoiser over hard subtraction, add a near-rotor reference mic,
  and train/eval on DroneAudioSet. Keep this out of the Phase 3 demo.
- **iFlytek DOES list Vietnamese offline** (18 languages) — the "nobody
  supports offline VI" claim is wrong; use "Vietnamese-first / industrial"
  instead. Timekettle now ships a handheld offline translator (T1, 44
  packs) but still has no Vietnamese offline pack. RelayX sets the
  ruggedization bar (MIL-STD-810H, IP68+IP69K, 110dB, Class 1 Div 2).

## Repo layout
```
docs/       notes, TODOs, ADRs, phase-submission drafts (see docs/README.md)
experiments/ Python — model eval, fine-tuning, quantization, ONNX export
src/        Rust — on-device runtime (single crate, see src/README.md)
models/     exported .onnx / QNN artifacts, the Python↔Rust handoff point
data/       glossary termbase, eval sets, raw corpora (mostly gitignored)
configs/    language-pair configs, model manifests, latency budgets
scripts/    glue scripts (AI Hub export, QNN convert, device deploy)
```
Each of `docs/`, `experiments/`, `src/`, `models/`, `data/`, `configs/`,
`scripts/` has its own `README.md` explaining what goes where.
