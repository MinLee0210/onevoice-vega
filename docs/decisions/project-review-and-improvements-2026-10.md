# Project review + improvement research (2026-10-07)

Read of the whole repo (AGENTS.md, docs/, configs/, `src/`, `scripts/`, CI,
uncommitted work) plus a short web pass. Goal: find where the repo's claims
and its actual state disagree, and rank what to do next. Nothing here was run
on hardware; "measured" numbers are the repo's own CPU-dev-machine numbers.

## 1. State of the project in one paragraph

A working CPU-only Rust pipeline (denoise → VAD → ASR → MT → TTS) that is
validated end to end for **one direction (vi→en)**, plus strong Python tooling
for QCS6490 compile/profile. Evidence quality in the docs is high (ADRs record
real bugs and rejected options). The gaps are coverage (5 of 6 directions have
no MT model), hardware (nothing has run on the Rubik Pi or the Hexagon), and
a few claims in AGENTS.md that the code or data don't back yet.

## 2. Claims vs. reality

| Claim (AGENTS.md / pitch) | What the repo shows | Gap |
|---|---|---|
| 6 translation directions | `configs/language-pairs.toml`: only `mt/vi-en` exists; the other 5 are "not exported yet" | Device cannot do 5/6 pairs |
| Glossary accuracy ≥ 95 % enforced | `data/glossary/termbase.tsv` has **10 rows**; validated on 10 sentences, vi-en only | Claim is untestable at this size |
| 2-mic beamforming + denoise is differentiator #3 | No beamforming stage in `src/stages/`; only RNNoise | Differentiator is unimplemented |
| ASR WER ≤ 15 % at 75 dB factory noise | `docs/reports/asr_noise_robustness.md` uses **white Gaussian noise**, 30 utterances, and a different bar (12 % at 5 dB, which it fails at 12.32 %). Noisy audio was not shown going through the denoiser | Not the target condition; target is inconsistent between docs |
| MT BLEU VI↔EN ≥ 40 | No BLEU measurement exists anywhere | Unmeasured |
| Streaming TTS hides latency | `Pipeline::run_once` calls `tts.synthesize()` to completion, then `play()`; the logged `through_tts_ms` is time to *full* synthesis | Metric is conservative, but streaming playback is not wired |
| Piper/MMS licensing is "MIT for Korean" | `facebook/mms-tts-kor` weights are **CC-BY-NC 4.0** (checked on the HF card 2026-10-07; the code is MIT). Fixed in AGENTS.md and the ko MANIFEST | Korean TTS is non-commercial; needs a replacement before shipping |
| Phase 3 prototype build Aug–Sep 2026 | Today is 2026-10-07; repo has no live capture (cpal), no PTT/LED/haptic code, `ort` `qnn` feature never enabled | Hardware bring-up is behind the timeline (Phase 4 field test is this month) |

## 3. Findings and improvements, ranked

### P0 — blocks the demo / the pitch

1. **Close the 5 missing MT directions.** Fastest credible path: EN-pivot.
   You already have vi-en; add en-vi, then `en-zh`, `en-ko`, `zh-en`, `ko-en`
   Opus-MT (check each exists and its license) and chain through EN. Cost is
   roughly 2× MT latency (measured MT ≈ 90–240 ms CPU, budget 150 ms), which
   still fits 1.2 s. Report the pivot-vs-direct quality delta (already an open
   item). The `Pipeline` needs a `pivot: Option<&str>` in the pair config.
2. **QNN smoke test, today, before more CPU work.** Risks to check in order:
   - The prebuilt ORT that `ort` downloads usually has no QNN EP for Linux
     aarch64; expect to build ORT with `--use_qnn` on/for the Pi and point
     `ort` at it (`ORT_LIB_LOCATION`). Verify; this can eat days.
   - QNN SDK version of the context binary must match the runtime on the
     device.
   - **Inconsistency in uncommitted work:** `scripts/qaihub/common.py` now uses
     `--target_runtime precompiled_qnn_onnx` (SDK 0.56), while the ADR,
     README and AGENTS.md all say `qnn_context_binary`. Pin the qai-hub
     version in `pyproject` and make the docs match whichever the pinned SDK
     accepts, or the recipe will silently rot.
3. **Live I/O + a long-running loop.** `Pipeline::new` loads every model for
   one pair and `run_once` handles one WAV. A handheld needs: resident models,
   cpal capture (48 kHz), a PTT GPIO event, and pair switching without a full
   reload (AGENTS.md's "lazy-load only the active pair" is not implemented).
   Smallest step: `Pipeline::run_loop()` + a `Hardware` trait (button, LED,
   haptic) with a file/keyboard fake so it's testable off-device.
4. **License audit before anything is public.** The uncommitted
   `scripts/upload_hf.py` pushes the entire `models/` tree to
   `LakoreAI/onevoice-vega`. That bundle mixes CC-BY (Piper VI, Opus-MT:
   attribution required), possibly CC-BY-NC (MMS-TTS), and espeak-ng data
   (GPL-3.0). Keep the repo private until a `THIRD_PARTY_LICENSES.md` + per-model
   `MANIFEST.md` license line exists; make `pack_models.sh` fail if a model
   dir lacks one. (`.env` is correctly gitignored.)

### P1 — makes the differentiators real

5. **Safety-phrasebook fast path.** For the 20–50 safety commands ("stop the
   machine", "evacuate", "forklift approaching"), pre-render TTS audio in all
   4 languages at build time and match ASR text against the termbase
   (normalised exact/fuzzy match). A hit bypasses MT and TTS: ~0 ms
   translation latency, 100 % term accuracy, no fluency cost, and it's the
   natural Vega Air payload (broadcast = fixed phrases most of the time). Free-form
   speech keeps using the normal path. This also sidesteps the logit-bias
   fluency problem for the most safety-critical cases.
6. **Scale the glossary and test it properly.** 10 terms → a few hundred per
   industry pack, with a held-out sentence set (not the one used to tune the
   bias strength), measured term accuracy *and* BLEU/chrF on the same set so
   the fluency cost is a number. Do this before LoRA; LoRA needs the same data.
7. **Noise evaluation that matches the claim.** Re-run R10 with real machinery
   noise (MUSAN is not factory noise; use DEMAND, MS-SNSD, FSD50K machinery
   classes, or recordings), SNR 0–20 dB, ≥200 utterances, and report **three
   arms**: raw, RNNoise, RNNoise + VAD. Denoisers frequently *raise* ASR WER
   because of artifacts; if RNNoise hurts, feeding raw audio to ASR (or
   fine-tuning PhoWhisper on noised data) is better than the current stage
   order. Also fix the target so AGENTS.md and the report agree.
8. **Encoder tail trimming: validate, don't assume.** The performance ADR
   recommends trimming the 30 s padded window. Public findings are mixed:
   shrinking Whisper's audio context gives ~3× encoder speedup but quality
   drops, and zero-padding helps suppress hallucination. On the NPU the
   static 30 s graph is only ~50 ms (whisper-base), so the win is smaller than
   on CPU. Measure WER delta at 2 s / 5 s / 10 s contexts first; the NPU may make
   this unnecessary.
9. **ASR confidence gate.** Whisper hallucinates on noise-only input that VAD
   lets through. Use mean token log-prob / no-speech prob from the decoder
   already running; below threshold → "didn't catch that" haptic/LED instead of
   a confident wrong translation. Matters more for a safety device than a
   few WER points.

### P2 — quality and process

10. **Wire streaming playback.** `TtsEngine` already chunks per sentence;
    return an iterator/callback so `play()` starts on the first chunk and
    report true "release → first audio". Likely 100–300 ms off for
    multi-sentence output.
11. **CI regression for the pipeline.** All real tests are `#[ignore]` because
    models are gitignored, so CI only checks formatting and trivial units.
    Add a nightly/manual workflow that pulls the model bundle (the HF/Release
    bundle already exists) and runs `cargo test -- --ignored` plus a golden
    WER/BLEU/latency check that fails on regression.
12. **Close the ASR model question with data.** Candidates to compare on one
    shared VI set (clean + noisy): PhoWhisper-base (current), PhoWhisper-small
    on NPU (ADR path), VietASR-style CTC (no decode loop), Qwen3-ASR-0.6B
    (Apache-2.0, on the Vietnamese Open ASR Leaderboard, but 0.6B is large for
    the NPU budget). Zipformer-30M is already rejected on accuracy.
13. **Vega Air specifics.** Translate once, synthesize the 3 target languages
    in parallel on spare cores (audio is operator-initiated, no latency
    sensitivity but the broadcast total should stay short); pre-cache phrasebook
    audio (item 5); add speaker EQ/limiter tuning to the outdoor-SPL work. Keep
    ego-noise out of Phase 3 as the existing ADR says.
14. **Housekeeping.** Commit or discard the 6 uncommitted files; `__pycache__`
    dirs exist under `scripts/qaihub` (ignored, fine). Reconcile TODO.md's
    stale "Plan" status vs. the timeline, and AGENTS.md's "validated" vs.
    "ASR WER 2.2 %" (10 samples) vs. report's 4.93 % clean (30 samples):
    use the larger number in the pitch.

## 4. Suggested order for the next two weeks

1. Days 1–3: QNN EP build + load whisper-base context binary on the Pi (item 2).
2. Days 2–5 (parallel, no hardware): MT pivot for all pairs (1), phrasebook
   path (5), license audit (4).
3. Week 2: live loop + hardware trait (3), real-noise eval with denoise ablation (7),
   glossary scale-up (6).
4. Then streaming playback (10), confidence gate (9), CI golden tests (11).

## 5. Open questions I could not resolve from the repo

- ~~Is the MMS-TTS Korean license really CC-BY-NC?~~ Yes, confirmed. Still unverified: the Opus-MT cards (Apache-2.0 per our manifests vs. CC-BY-4.0 as I recall).
- Does the `ort` 2.0.0-rc.12 build chain support a custom QNN-enabled ORT cleanly on aarch64?
- Do `opus-mt-en-ko` / `ko-en` exist with acceptable quality? (If not, KO may
  need a different MT model, e.g. a small multilingual one.)
- What does the team already have in hand for hardware (Rubik Pi, mic HAT, button)?
  The repo can't tell, and it changes the order above.

## Sources

- ONNX Runtime EPContext / QNN: https://microsoft.github.io/winml-cli/dev/concepts/compile-and-epcontext/
- NPUsper (Whisper on mobile NPUs): https://arxiv.org/pdf/2607.01108
- Whisper padding vs. hallucination / audio_ctx tradeoff: https://www.arxiv.org/pdf/2506.15912
- VietASR: https://arxiv.org/pdf/2505.21527
- Vietnamese Open ASR Leaderboard: https://vietaudio-team-vietnamese-asr-leaderboard.hf.space/
- Qwen3-ASR (52 languages, 0.6B/1.7B), per leaderboard listing above
