# Glossary runtime enforcement

**Update 2026-07-16: a working technique exists now.** Decoder logit
biasing — described in the new section below — is implemented in
`src/stages/mt.rs` and wired into the real pipeline
(`Pipeline::run_once`). Placeholder substitution (original content of
this doc, kept below) is still rejected; it was superseded by biasing,
not fixed.

## What ships today: decoder logit bias (2026-07-16)

Validated in `experiments/03_mt.ipynb` sections 7-7d against the real
vi-en Opus-MT model, then ported to `src/stages/mt.rs`
(`MtEngine::translate` / `run_decoder_greedy`). Mechanism: run a cheap
unbiased greedy decode first; if every glossary target term relevant to
the sentence (`Glossary::find_matches`) is already present, return it
unchanged; otherwise re-decode with a fixed logit bias (`GLOSSARY_BIAS =
12.0`) toward the missing term's tokens, tracking exactly one "next
expected token" position per glossary sequence at a time.

That last clause mattered in practice — getting there took three real,
sequentially-discovered bugs, each caught by checking actual decode
output/logits rather than trusting the first plausible-looking result:

1. **First attempt: bias only fired for single-token terms.** The bias
   check only handled "the full prefix except the last token has just
   been generated, so bias the last token" — for a 1-token term that's
   trivially "always", but for a multi-token term like "forklift"
   (`_fork` + `lift`), the *first* token (`_fork`) was never biased, so
   the sequence could never start. Result: bias values up to 14 had
   **zero** effect on the known "xe nâng" -> "lift" (should be
   "forklift") failure case. Caught by dumping the raw per-step logits
   directly: the gap between `_lift` (5.72) and `_fork` (-3.74) was
   9.46, so a bias of 14 mathematically should have flipped it if it
   were being applied at all — it wasn't.
2. **Second attempt: fixed (1), but biased every matching position
   independently, with no exclusivity.** Once a sequence was partway
   through (e.g. `_fork` already emitted), the next step biased *both*
   "restart the sequence from token 0" and "continue to token 1"
   simultaneously, because both checks independently matched. When the
   model's natural preference for repeating the first word was strong
   enough, "restart" kept winning for many steps: `bias=9.5` produced
   `"The forklift is approaching fork fork fork fork fork fork..."`
   (repeated ~50 times, cut off by `max_new_tokens`).
3. **Third attempt: track a single current match position per sequence**
   (`progress: Vec<usize>`, one biased token per unsatisfied sequence
   per step, standard prefix-matching) — fixed the repetition loops.
   Term accuracy on the known failure case: `"forklift approaching
   sector three."` Clean.
4. **Remaining problem after (3): fluency cost on sentences that didn't
   need fixing.** Running the biased pass on every glossary-flavored
   sentence — even the ones the unbiased model already translated
   correctly — visibly degraded them: `"This area is dangerous."` (baseline,
   fine) became `"danger zone."` (biased, worse) even though both
   contain "danger". Fix: only run the biased pass when the unbiased
   pass is actually missing the term (see mechanism above). This is
   the version that shipped.

Final numbers on a 10-sentence carrier set (one real Vietnamese sentence
per termbase entry, e.g. "Xe nâng đang đến gần khu vực số ba." for
forklift): baseline term accuracy 50% (5/10), final (conditional-bias)
term accuracy 100% (10/10), with the 5 already-correct sentences left
byte-for-byte unchanged. The 5 corrected sentences are grammatically
rougher than a human translation (`"forklift approaching sector
three."` drops "The"/"is"; `"loading docks parked in the area."` drops
"The truck's" and pluralizes "dock") but the safety-relevant term is
present and the sentence is intelligible — judged acceptable for a
safety-critical device where "understood, imperfect grammar" beats
"fluent, wrong term," but not yet reviewed by a native speaker for
whether the roughness crosses into actually confusing. Not benchmarked
against BLEU or a larger glossary — 10 terms, 10 test sentences, one
language pair (vi-en) only. LoRA fine-tuning (CLAUDE.md's original
plan) would likely fix the fluency cost more thoroughly and should
still happen before this is called done for Phase 3; logit biasing is
the cheap-to-ship stopgap for Phase 2, not a replacement for it.

Real gap not addressed here: `Glossary::find_matches` does a plain
case-insensitive substring match against the *source* sentence to
decide which terms are "relevant" — no morphological awareness, so a
source term that appears as a substring of an unrelated word would
false-positive (not observed in the 10-term termbase, but worth
knowing before growing it).

## Rejected: placeholder substitution

Tried the cheapest possible technique for making MT respect the
industrial glossary at runtime, before assuming it would work.
It doesn't, reliably enough to ship.

## The idea

Before calling MT, replace a known glossary source term (e.g. "xe nâng"
= forklift) with a placeholder string the model has presumably never
seen in training, translate, then replace the placeholder in the output
with the correct target term. No model changes, no beam search, works
fine with the greedy/beam=1 decoding CLAUDE.md already requires for
latency. If it worked, it'd be a same-day fix for the "Glossary term
accuracy ≥ 95% enforced" target.

## What actually happened

Tested against the real, already-validated `Helsinki-NLP/opus-mt-vi-en`
model on the known failure case: "Xe nâng đang đến gần khu vực số ba."
baseline-translates to "The lift is approaching sector three." (wrong —
"xe nâng" should be "forklift", not "lift").

Substituted "Xe nâng" with six placeholder candidates and translated
each:

| placeholder | survived verbatim? |
|---|---|
| `QQQZZZ` | yes |
| `Zzz9` | yes |
| `XNAAA` | no — came out as "XAAA's" |
| `<XNAAA>` | no — dropped entirely, decoder produced "We're" |
| `FORKLIFTTOKEN` | no — came out as "FORKIFTRON" |
| `xxxxx` | no — came out as "sexx" |

Two out of six survived. The other four got silently mangled into
something else, in ways that would be hard to detect programmatically
(the model doesn't signal "I couldn't handle this token," it just
produces something plausible-looking and wrong).

## Why this happens

Opus-MT's vocabulary is a fixed SentencePiece/BPE inventory learned
from training data. An arbitrary string gets segmented into whatever
subword pieces the tokenizer happens to produce for it — there's no
reserved "pass this through untouched" mechanism. Whether a given
placeholder survives depends on incidental properties of its
tokenization (how many pieces it splits into, whether those pieces
happen to also appear in other contexts the model learned strong
priors for) — not something predictable without testing each candidate
against the specific model.

## Why this is a hard no, not a "keep trying placeholders"

Even if a placeholder is found that survives on *this* test sentence,
there's no guarantee it survives in different sentence contexts, or
across the other 5 language directions (each a different underlying
model with a different vocabulary), or after any future model
swap/fine-tune. A glossary system with an ~unknown, per-term,
per-context failure rate is worse than no glossary system for a
safety-critical use case (CLAUDE.md's own example terms are "safety
commands") — it would create false confidence that a term is enforced
when it silently isn't, some fraction of the time.

## What's actually real right now (updated 2026-07-16)

- `src/stages/glossary.rs`: real file-backed termbase loading
  (`data/glossary/termbase.tsv`, parses into all directional pairs
  among whatever language columns are present), plus `find_matches`
  for locating which terms apply to a given sentence. Consulted by
  `mt` — see "What ships today" above.
- `src/stages/mt.rs`: `MtEngine::translate` runs the conditional
  decoder-bias enforcement described above. Wired into
  `Pipeline::run_once` — this is live in the real pipeline, not just a
  notebook prototype.
- `data/glossary/termbase.tsv`: 10 real VI/EN/ZH/KO industrial terms
  (forklift, stop-the-machine, safety glasses, alarm, emergency exit,
  fire extinguisher, helmet, danger, loading dock, first aid). ZH/KO
  translations are my best-effort common-vocabulary translations, not
  reviewed by a native speaker — flag before treating as
  safety-authoritative. Only vi-en enforcement has been validated (this
  model/tokenizer); the other 5 language pairs don't have MT models
  ported yet at all (see docs/TODO.md), so glossary bias there is
  untested by construction, not just unvalidated.

## What actually needs to happen next

The 2026-07-16 update above corrects this doc's earlier assumption that
decoder-logit biasing "needs real source-target alignment info" — it
doesn't; biasing toward a term "wherever the model is willing to place
it" (no forced position) turned out to be enough once the bias-tracking
bugs were fixed. What's still open:

- **Fluency cost is real and unmeasured beyond spot-checking 5
  sentences.** CLAUDE.md's original plan — LoRA fine-tune on the
  termbase — would very likely produce more natural output than
  logit-patching a base model's decode, and is still the right target
  for Phase 3. This is the stopgap, not the destination.
- **Native-speaker review of ZH/KO termbase entries** and of whether the
  biased-pass English output (grammatically rough in ~half the corrected
  cases) is actually acceptable to a factory-floor listener, not just
  "technically contains the right word" — the substring-match eval
  metric used in experiments/03_mt.ipynb section 7 doesn't capture that.
- **`GLOSSARY_BIAS = 12.0` is a single global constant**, picked from a
  sweep against one sentence ("xe nâng") and reused everywhere. Whether
  it's the right value for every termbase entry / every language pair's
  model is unverified — it happened to work for all 10 carrier
  sentences tested, which isn't the same as a systematic sweep.
