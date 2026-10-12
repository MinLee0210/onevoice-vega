# MT quality (Helsinki-NLP/opus-100 test slice)

| leg | model | BLEU | chrF | n |
|---|---|---|---|---|
| vi-en | opus-mt-vi-en | 29.6 | 47.2 | 200 |
| en-vi | opus-mt-en-vi | 22.2 | 37.2 | 200 |
| en-zh | opus-mt-en-zh | 46.5 | 41.9 | 200 |
| zh-en | opus-mt-zh-en | 44.4 | 66.7 | 200 |
| en-ko | opus-mt-tc-big-en-ko | 0.1 | 1.4 | 200 |
| ko-en | opus-mt-ko-en | 15.0 | 33.9 | 200 |

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

- **VI↔EN** (the flagship pair) is the weakest: vi→en BLEU 29.6, en→vi
  22.2 on Opus-100 test. The project target is BLEU ≥40, but that is on
  *PhoMT + a domain test set*; Opus-100 test is broad/general. Still, this is
  the first real number and it's below target — LoRA/glossary + a domain test
  set are justified. (COMET not run: `unbabel-comet` was not installed in the
  ephemeral env; add it for the judge-facing number.)
- **EN↔ZH is strong** (46.5 / 44.4) — the VI→ZH pivot is bottlenecked by the
  VI→EN leg, not by EN→ZH.
- **KO is broken.** `opus-mt-tc-big-en-ko` yields BLEU **0.1** and garbage
  output even for "Hello, how are you?" (verified directly with and without
  `sacremoses`, and with a `>>ko<<` prefix). ko→en (15.0) is merely mediocre.
  **VI→KO is therefore not viable via the EN pivot with this checkpoint.**
  Options: a different en→ko model, or a single multilingual model (M2M-100,
  MIT) for the KO pair. This stacks with the existing KO TTS problem
  (MMS-TTS-KO is CC-BY-NC) — KO is the direction that needs the most work.
- Pivot scores above are the *leg* BLEUs, not the composed corpus BLEU (the
  legs are evaluated on independent sentences). The true pivoted numbers will
  be ≤ the weaker leg; measure them once an aligned VI↔ZH/KO set is available
  (FLORES-200 is gated; a PhoMT or in-house domain set is the alternative).

## Method / caveats

- Helsinki-NLP/opus-100 test split, first 200 pairs per config; greedy
  (beam=1) decode; sacrebleu BLEU (13a / zh tokenizer) and chrF. Same
  checkpoints the runtime exports, so numbers reflect the shipped weights
  (fp32 here vs int8 at runtime).
- Opus-100 license is per-corpus (mixed OPUS); internal-eval use only.
