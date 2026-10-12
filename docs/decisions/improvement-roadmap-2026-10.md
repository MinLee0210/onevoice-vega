# Improvement roadmap & execution plan (2026-10-12)

Derived from `pipeline-improvement-research-2026-10.md` (online research) and
the deployment findings (`qcs6490-*-2026-10.md`, `vega-hand-air-2026-10.md`).
Ordered by leverage against the judging rubric — **Technical Performance 50%
(BLEU+COMET, TTS MOS, RTF, latency) / Innovation 25% / Business 25%**.

Start-here principle: the pipeline must do the job it advertises **before**
optimizing it. It currently translates 2 of 6 directions and has no BLEU/COMET
measurement.

## Workstream 1 — Translation coverage + quality harness  ← START HERE

**Goal:** all 6 directions translate, and quality is a measured number.

| # | Task | Status | Acceptance |
|---|---|---|---|
| 1.1 | EN-pivot support: `LanguagePair.pivot` in `config.rs`/`language-pairs.toml`; `MtChain` chains src→pivot→tgt | ✅ done | unit tests + `cargo test` green |
| 1.2 | Export/fetch missing MT legs (en-zh, en-ko, zh-en, ko-en) | ✅ done (optimum) | `models/mt/*` populated |
| 1.3 | `scripts/evaluate_mt.py`: sacrebleu BLEU + chrF on opus-100 (FLORES gated) | ✅ done | `docs/reports/mt_quality.md` |
| 1.4 | Run 1.3 for all 6 legs/directions | ✅ done | numbers written |

Result: all 6 directions wired and runnable; VI↔EN BLEU 29.6/22.2; EN↔ZH strong
(46.5/44.4); **KO blocked** (en→ko BLEU 0.1, tc-big checkpoint broken). See
`docs/reports/mt_quality.md`. Remaining follow-ups in
`deferred-work-runbook-2026-10.md`.

**Why first:** biggest functional gap; directly drives the 50%-weighted score;
needs no hardware.

## Workstream 2 — ASR

- 2.1 ✅ **done (partial):** `scripts/evaluate_asr.py` → `docs/reports/asr_ab.md`.
  PhoWhisper-base (clean 2.19% / 5 dB 9.84% WER, RTF 0.16) **beats**
  Qwen3-ASR-0.6B (4.92% / 10.38%, RTF 0.72). Keep PhoWhisper for VI.
- 2.2 ✅ **CTC backend implemented** (`src/stages/ctc.rs`, `CtcAsrEngine`;
  `Asr::load` auto-selects). Exported `dragonSwing/wav2vec2-base-vietnamese`
  (`scripts/edge/export_ctc_asr.py`); Rust pipeline runs it end-to-end (ASR
  57 ms / total 181 ms vs Whisper 485 ms) at higher WER.
- 2.3 ◑ **CTC sweep done** (`docs/reports/asr_ctc_sweep.md`): no
  commercially-clean VI CTC good enough yet → **fine-tune recipe** ready at
  `scripts/edge/finetune_ctc_vi.py` +
  `docs/decisions/ctc-finetune-recipe-2026-10.md` (needs a GPU).

## Workstream 3 — TTS

- 3.1 Replace KO (CC-BY-NC MMS) and de-risk VI (espeak GPL): eval Apache-2.0
  `parler-tts-vietnamese` / `spark-tts-vietnamese`, `Kokoro-82M` (en/zh).
- 3.2 Native-speaker MOS pass (judging scores MOS).

## Workstream 4 — Differentiators made real

- 4.1 Glossary **hard-example LoRA** (proven +11% term accuracy) — needs data/GPU.
- 4.2 ◑ **real-noise eval done** (`scripts/evaluate_asr_noise.py`,
  `docs/reports/asr_noise_real.md`): PhoWhisper 11.62% at 5 dB on ESC-50
  industrial noise (≤15% holds; AWGN was conservative). Still open: the
  **denoise ablation** (needs RNNoise/VAD in Rust — Python bindings broken)
  and the **2-mic beamformer** front-end.
- 4.3 QE-gate + MBR self-correction (Hand) — safety + BLEU/COMET.

## Workstream 5 — Deployment (critical path, blocked on hardware)

- Enable `ort` `qnn` feature; load the split-decoder context binary on the
  Rubik Pi (week-1 smoke test). Requires the board.

## Execution notes (this session)

Tasks 1.1–1.4 are executed here. 2.x/3.x/4.x deliver scaffolding/scripts and
measurements where the environment allows, and are otherwise left as
runbook-ready tasks because they need torch training runs or physical hardware.
