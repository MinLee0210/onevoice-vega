# MT quality (Helsinki-NLP/opus-100 test slice)

| leg | model | BLEU | chrF | COMET | n |
|---|---|---|---|---|---|
| vi-en | opus-mt-vi-en | 29.6 ±4.2 | 47.2 ±3.9 | — | 200 |
| en-vi | opus-mt-en-vi | 22.2 ±3.5 | 37.2 ±3.3 | — | 200 |
| en-zh | opus-mt-en-zh | 46.5 ±4.3 | 41.9 ±4.0 | — | 200 |
| zh-en | opus-mt-zh-en | 44.4 ±5.5 | 66.7 ±4.2 | — | 200 |
| en-ko | opus-mt-tc-big-en-ko | 0.1 ±0.1 | 1.4 ±0.4 | — | 200 |
| ko-en | opus-mt-ko-en | 15.0 ±3.5 | 33.9 ±3.2 | — | 200 |

## Pipeline directions

| direction | path | legs | leg BLEU |
|---|---|---|---|
| vi-en | direct | vi-en | vi-en=29.6 |
| en-vi | direct | en-vi | en-vi=22.2 |
| vi-zh | pivot | vi-en→en-zh | vi-en=29.6 + en-zh=46.5 |
| zh-vi | pivot | zh-en→en-vi | zh-en=44.4 + en-vi=22.2 |
| vi-ko | pivot | vi-en→en-ko | vi-en=29.6 + en-ko=0.1 |
| ko-vi | pivot | ko-en→en-vi | ko-en=15.0 + en-vi=22.2 |

## Analysis

- **VI↔EN** (flagship) is the weakest leg (vi→en 29.6, en→vi 22.2 on Opus-100
  test). The ≥40 target is on *PhoMT + a domain set*; Opus-100 is broad. Still
  below target → justifies the glossary LoRA / domain data.
- **EN↔ZH is strong** (46.5 / 44.4): the VI→ZH pivot is bottlenecked by VI→EN.
- **KO is broken:** `tc-big-en-ko` yields BLEU 0.1 (garbage); VI→KO is wired
  but not usable. Needs a different en→ko model or M2M-100.
- Values are BLEU/chrF with **95% bootstrap CIs** (n=200). Pivot rows are the
  *leg* scores, not a composed corpus number (legs use different sentences).
- **COMET not run here:** `unbabel-comet`'s `fastBPE` native build fails on
  macOS. The `--comet` path is implemented in `scripts/evaluate_mt.py`; run it
  on Linux/CI (`scripts/ci/golden.sh` runs there) or a GPU box for the judged
  metric. `docs/decisions/deep-research-2026-10.md` notes COMET is scored.

## Method

Helsinki-NLP/opus-100 test split, first 200 pairs; greedy decode; sacrebleu
BLEU (13a / zh tokenizer), chrF; bootstrap CIs (300 resamples). Same
checkpoints the runtime exports (fp32 here vs int8 at runtime).
