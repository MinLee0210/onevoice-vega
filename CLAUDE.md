# CLAUDE.md — OneVoice AI Challenge Project

## Project
On-device (no cloud) speech translation device for noisy industrial
environments (factories, construction, logistics, hospitals). Built for
Saigon AI Hub × Qualcomm "OneVoice AI Challenge." Currently Phase 2
(technical spec submission, due July 2026).

## Languages/pairs
VI↔EN, VI↔ZH, VI↔KO (6 directions total)

## Conversation model
Half-duplex, push-to-talk (PTT). One party holds button, speaks, releases;
device translates and plays back. No full-duplex streaming needed —
process complete utterances (≤10s), not partial/incremental audio.

## Hardware target
- Prototype: Rubik Pi 3 (Qualcomm Dragonwing QCS6490), 8-core Kryo 670 CPU,
  Hexagon NPU @ 12 TOPS, 8GB LPDDR4x, 128GB UFS
- Production vision: same QCS6490-class SoM shrunk into ruggedized
  handheld/neckband form factor, IP54+, physical PTT button, LED status
  (no touchscreen)

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
| ASR (VI) | PhoWhisper-base 74M (BSD-3-Clause) — validated, WER 2.2% on VIVOS sample | PhoWhisper-small 244M, W8A16 on Hexagon; also worth evaluating VietASR (13M, CTC not seq2seq — no autoregressive decode loop at all, Apache-2.0), see docs/decisions/pipeline-improvement-research-2026-07.md |
| ASR (EN/ZH/KO) | Whisper-Small — **not** via AI Hub pre-quantized (that path is currently broken for QCS6490, see Hardware toolchain below); use a generic pre-made ONNX instead (e.g. onnx-community/whisper-small, MIT) | Whisper-Large-v3-Turbo quantized |
| MT | Opus-MT per-pair (~300MB, Apache/CC-BY, INT8) — validated, Rust port working end-to-end | LoRA fine-tune on industrial glossary; VinAI-Translate/envit5 for VI↔EN quality |
| TTS | All 4 target langs (device is bidirectional): Piper (VITS/ONNX, CPU real-time) for VI (`vais1000-medium`) / EN (`lessac`) / ZH (`huayan`, Mandarin); MMS-TTS (`facebook/mms-tts-kor`, MIT, VITS/ONNX) for KO — Piper has no Korean voice. **Real Rust implementation, not a stub** — `piper-rs` crate for VI/EN/ZH (VITS + espeak-ng phonemization), hand-rolled `uroman` + ported char-tokenizer + raw `ort` inference for KO (no crate covers MMS). All 4 produce genuine audio, round-trip-ASR verified through the real Rust path, not just the Python notebook. See license trap below (not a hackathon blocker) and `src/stages/tts.rs` module docs for the `piper-rs` build gotcha | Kyutai Pocket TTS once it supports Vietnamese (English-only as of this writing, checked directly) |

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
commercial commits. The Korean path (MMS-TTS) is MIT and avoids espeak
entirely. Full detail: docs/decisions/piper-vi-voice-license-trap.md.

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
1. **Industrial glossary engine** — VI/EN/ZH/KO parallel termbase
   (safety commands, machine names, logistics terms). Enforcement today
   is a working decoder-logit-bias technique (validated vi-en only,
   100% term accuracy on a 10-sentence test set vs. 50% baseline, at a
   real but acceptable fluency cost — see
   docs/decisions/glossary-runtime-enforcement-2026-07.md), shipped as
   the Phase 2 stopgap. LoRA fine-tune remains the intended Phase 3
   upgrade for better fluency. Swappable per-industry vocab packs.
2. **Noise-first audio front-end** — 2-mic beamforming + denoise,
   benchmarked WER-vs-SNR curve at 70-85dB factory noise. Note: iFlytek
   (a real competitor, see docs/decisions/competitor-landscape-2026-07.md)
   uses a 4-mic array for its noisy-environment claims — worth a direct
   benchmark once hardware exists in case 2 mics turns out to be
   optimistic, not yet a reason to change the spec.
3. **Latency engineering** — denoise-during-capture, greedy MT decode,
   streaming TTS playback, instrumented per-stage timing / live latency
   HUD for demos.

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

**Whisper specifically does not follow this path today.** AI Hub's
quantized Whisper for QCS6490 needs AIMET-ONNX, which has no aarch64
Linux wheel — confirmed via Qualcomm's own `ai-hub-models` issue #281,
closed as not planned, not just an unconfirmed rumor. Options: re-check
AI Hub Workbench closer to Phase 3 (fast-moving tooling), or take a
generic pre-quantized ONNX Whisper (already validated in
`experiments/02_asr.ipynb`) and try the QNN execution provider directly
against it, bypassing AI Hub's quantization step entirely. Full detail:
docs/decisions/qcs6490-whisper-deployment-risk.md.

## Contest/business context
- Co-hosted with Qualcomm; part of Qualcomm's broader Vietnam AI R&D push
  (Hanoi/HCMC AI center opened June 2025, VinAI/MovianAI acquisition
  April 2025 — PhoWhisper/PhoMT lineage is now Qualcomm-owned talent).
- Judging: Technical 50% / Innovation 25% / Business viability 25%.
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
  on a small VIVOS sample. Still not validated: W8A16 on the actual
  Hexagon NPU via AI Hub Workbench — that whole deployment path is
  currently broken, not just unproven, see below.
- **QCS6490 Whisper/AI Hub deployment is broken today, not just
  unproven** — AI Hub's quantized Whisper needs AIMET-ONNX, which has
  no aarch64 Linux wheel (Qualcomm's own `ai-hub-models` issue #281,
  closed as not planned). Needs a real plan before Phase 3 hardware
  bring-up. See docs/decisions/qcs6490-whisper-deployment-risk.md.
- **Piper's Vietnamese TTS has a license trap and an unresolved GPL
  question** (phonemizer dependency) — see the model stack license trap
  note above and docs/decisions/piper-vi-voice-license-trap.md. Now
  actually shipping in `src/`'s real dependencies (not just a Python
  notebook), which is exactly the point this question needs a real
  legal answer before commercialization — the hackathon prototype
  proceeds regardless per the explicit call to defer legal to that
  later stage.
- VI↔KO / VI↔ZH quality (low-resource pairs) — may need EN-pivot
  fallback (VI→EN→KO/ZH); measure quality delta.
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
