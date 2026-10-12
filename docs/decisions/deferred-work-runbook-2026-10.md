# Deferred-work runbook (2026-10-12)

Workstreams 2–5 from `improvement-roadmap-2026-10.md` that need model
training or the Rubik Pi, so they can't be finished in a code-only session.
Each entry gives the concrete first command/step so it's turnkey.

## Workstream 2 — ASR A/B (PhoWhisper vs Qwen3-ASR vs VietASR)

Goal: WER + RTF + latency on one shared VI set (clean + noisy).

`scripts/evaluate_asr.py` implements this (VIVOS test slice via `quancoanh`/`quocanh34/viet_vivos`,
soundfile-decoded; PhoWhisper via the ASR pipeline, Qwen3-ASR via its
multimodal API). Run:

```sh
uv run --no-project --python 3.12 --with torch --with "transformers>=5.13.0" \
  --with jiwer --with datasets --with soundfile --with librosa \
  python scripts/evaluate_asr.py --num-samples 20 --snr clean 5 \
  --out docs/reports/asr_ab.md
```

Qwen3-ASR needs `transformers>=5.13.0` + `librosa`.

**Done since:** CTC backend implemented (`src/stages/ctc.rs`) + a CTC sweep
(`docs/reports/asr_ctc_sweep.md`); the only Apache-2.0 VI CTC is too weak, the
strong ones are CC-BY-NC. Follow-up = fine-tune the Apache-2.0 one:
`scripts/edge/finetune_ctc_vi.py` (+ recipe in
`docs/decisions/ctc-finetune-recipe-2026-10.md`). Needs a GPU.

Acceptance: fine-tuned CTC ≤7% clean / ≤15% @5 dB → switch the runtime default
to CTC and re-measure end-to-end (expected ASR ~57 ms, total ~181 ms).

## Workstream 3 — License-clean TTS (esp. KO, then VI)

Problem: KO = MMS-TTS **CC-BY-NC** (blocks commercialization); VI uses the
GPL espeak-ng path. Candidates (verify each card):
- `parler-tts-vietnamese` (Apache-2.0), `spark-tts-vietnamese` (Apache-2.0).
- `hexgrad/Kokoro-82M` (Apache-2.0) — en + zh only, no VI.
- For KO, find an Apache/MIT KO VITS or license a voice; MMS cannot ship.

Steps: download candidate → ONNX export (reuse `scripts/edge/export_mt.py`
pattern) → drop into `models/tts/<lang>/` → round-trip ASR check → native MOS.
Acceptance: all 4 voices commercially licensed + a native-speaker MOS number
(judging scores MOS).

## Workstream 4.1 — Glossary hard-example LoRA

Proven recipe (WMT26, `2609.09999`): keep only pairs where the base model
contradicts the glossary → +11% term accuracy at fixed data.

Steps:
1. Build the training set from `data/glossary/termbase.tsv`: for each term,
   translate glossed sentences, keep only the ones Opus-MT gets wrong.
2. LoRA fine-tune the Opus-MT student (needs GPU; ~hours).
3. Export + int8-quantize (`scripts/edge/quantize_mt.py`); re-run
   `scripts/evaluate_mt.py` to confirm no BLEU regression.

Acceptance: term accuracy measured on a **held-out** sentence set (not the
one used to tune the logit bias) + BLEU/chrF unchanged.

## Workstream 4.2 — 2-mic beamformer + real-noise eval

Reference: neural MVDR, 15 ms/frame, mixed precision (`2609.33755`); dual-mic
RDSE (`2610.12476`). The unimplemented differentiator #3.

Steps:
1. Extend `scripts/evaluate_noise_robustness.py` to **three arms**
   (raw / RNNoise / RNNoise+VAD) and real machinery noise (DEMAND / MS-SNSD /
   FSD50K machinery), SNR 0–20 dB, ≥200 utts — the current AWGN-only report
   doesn't match the AGENTS claim.
2. Implement a 2-mic delay-and-sum / MVDR stage in `src/stages/` (new stage),
   gated behind the Rubik Pi ReSpeaker HAT.

Acceptance: WER-vs-SNR curve for all three arms; decision on denoise order
(denoising can *raise* WER — keep the ablation).

## Workstream 5 — QNN bring-up (critical path, needs the board)

`ort` `qnn` feature is never enabled/tested. Week-1 Phase-3 smoke test:
build ORT with QNN EP on aarch64, load the ASR split-decoder context binary,
verify on the Rubik Pi. Blocked on hardware; see AGENTS.md → Hardware toolchain.

## Not-here-because

- **Training** (LoRA, distillation) needs a GPU and data curation time.
- **Hardware** (beamformer, QNN) needs the Rubik Pi / mic HAT.
- **MOS / COMET** need human raters and `unbabel-comet` (heavy); BLEU/chrF
  proxies are in place.
