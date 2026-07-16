# Piper TTS: another license trap, same shape as the NLLB one

CLAUDE.md's model stack picks Piper as the TTS baseline with no license
caveat attached — worth one, in the same spirit as the existing NLLB
CC-BY-NC callout. Two separate issues, both hiding behind a
repo-level `license: mit` tag that doesn't tell the whole story.

## Issue 1 — per-voice dataset license, not just the repo license

`rhasspy/piper-voices` (the HF repo hosting all Piper `.onnx` voices) is
tagged MIT at the repo level, but that's the *code/packaging* license —
each individual voice is fine-tuned on a specific dataset with its own
license, declared per-voice in `MODEL_CARD`. For the three Vietnamese
voices available:

| Voice | Quality | Dataset | License | Commercial use? |
|---|---|---|---|---|
| `vi_VN-vais1000-medium` | medium | VAIS-1000 | CC BY 4.0 | **Yes**, with attribution |
| `vi_VN-vivos-x_low` | x_low | VIVOS (InfoRe Technology 1) | **CC BY-NC-SA 4.0** | **No** — NonCommercial |
| `vi_VN-25hours_single-low` | low | InfoRe Technology 1 | **Unknown** | Can't confidently claim yes |

Ironically the VIVOS-based voice is disqualified even though we already
use the VIVOS *dataset* for ASR eval (`experiments/02_asr.ipynb`) — VAD/ASR
eval-set use and TTS-training-derivative use are different legal
situations, don't assume one license clears the other just because the
underlying speech corpus is the same.

**Only `vais1000-medium` is clearly safe for a commercial pitch**, and
conveniently it's also the best quality tier available for Vietnamese
in this repo — no quality/license tradeoff here, just pick the right
one.

## Issue 2 — phonemization needs espeak-ng, which is GPL-3.0

Piper's ONNX model takes *phoneme IDs* as input, not raw text — text
has to be converted to phonemes first by `piper-phonemize`, which links
against `espeak-ng`. **`espeak-ng` is GPL-3.0.** This is why active
Piper development moved from the archived MIT-licensed `rhasspy/piper`
to `OHF-Voice/piper1-gpl` (GPL-3.0) in October 2025 — the maintainers
essentially conceded the MIT label was untenable given the embedded
GPL dependency.

This was raised directly with the maintainer twice and never got a
firm resolution:
[rhasspy/piper issue #93](https://github.com/rhasspy/piper/issues/93)
(2023, no maintainer reply on record) and
[rhasspy/piper discussions #271](https://github.com/rhasspy/piper/discussions/271),
where the maintainer's own words were: "most voices use the espeak-ng
phonemizer from piper-phonemizer, which has a GPL license" and told
users to "beware" — without saying what that means for a product
embedding it.

**What GPL-3.0 on the phonemizer plausibly means for us:** if our
on-device Rust binary statically or dynamically links `espeak-ng` (or a
port of it) to do phonemization, the FSF's position on GPL "linking"
would treat the combined work as a derivative subject to GPL-3.0 —
which would require us to release our own source under GPL-3.0 too if
we distribute the device. That's a real conflict with CLAUDE.md's
commercialization ambitions (Qualcomm ecosystem partnership,
"commercializable local product story"). Running `espeak-ng` as a
*separate process* (not linked) is a common way projects argue around
this (the "mere aggregation" exception), but that's untested for our
case and not a substitute for legal review before shipping.

## What this means for next steps

- Use `vais1000-medium` for prototyping (Phase 3), not the VIVOS or
  unknown-license voices.
- The phonemizer dependency is the open question, and it's a licensing
  question, not just an engineering one — resolving it needs either
  (a) a non-GPL phonemizer for Vietnamese (need to research if one
  exists with adequate quality), (b) a legal read on the
  separate-process argument, or (c) accepting GPL for the phonemizer
  component specifically and scoping what that requires (may be fine
  if it's an isolated component, but "may be fine" isn't a decision,
  someone needs to actually make this call before Phase 3 commits to
  an architecture).
- Not a blocker for building and testing a prototype now — it's a
  blocker for the commercialization story specifically, which is a
  Phase 2/3 business concern, not a Phase 2 technical spec concern.
