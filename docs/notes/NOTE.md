# 2026-07-14 — MT stage (vi-en): baseline, ONNX export, Rust port

Implemented and validated the MT pipeline stage end to end, starting
from the translation task since it's independent of the audio stages.

## What was done

- Renamed `notebook/` → `experiments/`, kept it flat (`.ipynb` files
  only, no per-stage subdirectories).
- Set up the `audio-lab` conda env for all Python/notebook work in this
  project (installed missing packages: onnx, onnxruntime, optimum,
  optimum-onnx, sentencepiece, sacremoses, ipykernel; registered it as a
  Jupyter kernel). See `experiments/README.md` → Setup.
- `experiments/03_mt.ipynb`: loaded `Helsinki-NLP/opus-mt-vi-en`
  (Apache-licensed Opus-MT baseline per AGENTS.md's model stack),
  translated test sentences with PyTorch as the reference, exported to
  ONNX via `optimum` (encoder + merged decoder-with-past), verified the
  ONNX output matches PyTorch exactly, converted Marian's slow
  tokenizer (sentencepiece + vocab.json — no fast variant exists) into
  a `tokenizer.json` for the Rust `tokenizers` crate, and manually
  reimplemented the greedy KV-cache decode loop with raw `onnxruntime`
  as a dry run for the Rust port.
- Artifacts packaged into `models/mt/vi-en/` (encoder_model.onnx,
  decoder_model_merged.onnx, tokenizer.json, MANIFEST.md).
- `src/stages/mt.rs`: real `ort`-based `MtEngine` (was a `todo!()`
  stub) — loads tokenizer + both ONNX sessions, runs the encoder once,
  greedy-decodes with the KV cache. Two `#[ignore]`d tests exercise it
  against the real model files and assert against the same reference
  translations; both pass, including an industrial-glossary sentence
  ("xe nâng" / forklift).
- `Pipeline::run_once` and all stage stubs changed `&self` → `&mut self`
  (forced by `ort::Session::run` needing `&mut`).
- `configs/language-pairs.toml`: `mt_model` → `mt_model_dir` (MT needs
  a directory of 3 files, not a single file, unlike ASR/TTS).

## Findings worth remembering

- Generic Opus-MT mistranslates domain terms ("xe nâng" → "lift" not
  "forklift", "vận hành máy" → "the engine works" not "operate the
  machine") — the exact gap the glossary engine differentiator is meant
  to close. Not yet wired into MT decode (`Glossary` exists but isn't
  consulted yet).
- Two ONNX Runtime merged-decoder gotchas (cross-attention KV only
  valid on step 0; default execution provider can silently break
  zero-length KV tensors) — full writeup in
  `docs/decisions/onnx-runtime-merged-decoder-gotchas.md`, since ASR's
  decoder will hit the same graph shape later.
- Current MT export is fp32 (~500MB), over the ~300MB/pair memory
  budget — INT8 quantization is still open (tracked in `docs/TODO.md`).

See `docs/TODO.md` for the follow-up list (quantization, other 5
language directions, glossary wiring, QNN execution provider port).

---

# 2026-07-15 — ASR stage (vi, en-zh-ko): pre-made ONNX, no conversion

Moved to speech-to-text. Unlike MT, didn't export anything ourselves —
searched for and used pre-converted ONNX Whisper/PhoWhisper directly,
per the ask.

## What was done

- `experiments/02_asr.ipynb`: downloaded
  [huuquyet/PhoWhisper-base](https://huggingface.co/huuquyet/PhoWhisper-base)
  (ONNX conversion of `vinai/PhoWhisper-base`, the VI baseline from
  AGENTS.md) and
  [onnx-community/whisper-small](https://huggingface.co/onnx-community/whisper-small)
  (ONNX conversion of `openai/whisper-small`, the EN/ZH/KO baseline) —
  both already ship encoder/decoder ONNX *and* a proper fast
  `tokenizer.json`, no export or tokenizer-conversion step needed at
  all this time.
- Validated fp32 ONNX output matches PyTorch exactly (10/10 VIVOS
  samples, VI), then also validated the **pre-made int8 quantized**
  variant both repos ship — same WER as fp32 on the test slice, ~3.8x
  smaller (77MB vs. 291MB for PhoWhisper-base). Packaged the int8
  variant as the real artifact (unlike MT, no reason to wait on
  quantization here — it's already done and already validated).
- VI WER: 2.2% (10 VIVOS samples, `quocanh34/viet_vivos` — the
  official `vivos` HF dataset uses a loading script recent `datasets`
  versions no longer support; this parquet mirror works).
- EN WER: ~26-27%, but that's mostly casing/punctuation normalization
  noise against LibriSpeech's un-normalized reference text ("mister"
  vs. "mr.", missing commas), not real transcription errors — see the
  notebook for the actual hypothesis/reference pairs, they read as
  correct.
- Artifacts packaged into `models/asr/vi/` and `models/asr/en-zh-ko/`
  (encoder_model.onnx, decoder_model_merged.onnx, tokenizer.json,
  preprocessor_config.json, generation_config.json, MANIFEST.md each).
- Did **not** port to Rust this round (scope was "find and use the
  pre-made ONNX", not "implement the stage") — see the follow-up note
  below on why that's a bigger lift than MT was.

## Findings worth remembering

- **Big one:** Qualcomm's own AI Hub does not currently have a working
  quantized-Whisper deployment path for QCS6490 (the Rubik Pi 3's SoC
  family) — [ai-hub-models issue #281](https://github.com/qualcomm/ai-hub-models/issues/281),
  closed as not planned. AGENTS.md's model stack table casually lists
  "Whisper-Small (AI Hub, pre-quantized)" as if this is solved; it
  isn't. Full writeup with sources and options in
  `docs/decisions/qcs6490-whisper-deployment-risk.md` — worth reading
  before Phase 3 hardware planning locks in assumptions.
- Both PhoWhisper and Whisper-small ONNX repos ship a wide range of
  precisions off the shelf (fp32, fp16, int8, uint8, q4, even bnb4) —
  quantization for ASR may not need any work from us at all, unlike MT
  where we're still carrying an fp32-only artifact.
- Whisper's tokenizer (BPE/GPT-2 style) already has a native fast
  `tokenizer.json` — no repeat of MT's sentencepiece-to-Unigram
  conversion dance was needed.
- Whisper needs a log-mel spectrogram feature extractor *before* the
  ONNX encoder — this is real preprocessing (FFT + mel filterbank),
  not a lookup table like MT's tokenizer, and it's not baked into the
  ONNX graph. This is the main reason a Rust port of ASR is a bigger
  lift than MT's was; no Rust crate for it has been evaluated yet.
- Licenses are clean: `vinai/PhoWhisper-base` is BSD-3-Clause,
  `openai/whisper-small` is MIT — no repeat of the NLLB CC-BY-NC trap.
- Minor: `max_new_tokens=64` truncated the longest LibriSpeech test
  sentence mid-transcription. Not a concern for this device's actual
  use case (≤10s push-to-talk utterances are short), but worth knowing
  the cap is there if a stage produces a long transcript for another
  reason.

See `docs/TODO.md` for the follow-up list (Rust port + feature
extractor research, QCS6490 deployment risk, ZH/KO validation).

---

# 2026-07-15 (cont.) — TTS stage (vi): Piper, license research first

Moved to TTS next (completes an audio-in → audio-out demo loop with MT
+ ASR already done). Piper voices are natively ONNX, so — like ASR — no
export/conversion. But picking *which* Vietnamese voice turned into a
real license investigation before any implementation happened.

## What was done

- `experiments/04_tts.ipynb`: downloaded the pre-made ONNX voice
  (`rhasspy/piper-voices`, `vi_VN-vais1000-medium`), synthesized the
  same 4 VI test sentences used throughout MT/ASR, then closed the loop
  by feeding the synthesized audio back through the already-validated
  PhoWhisper ASR and comparing to the original text (no human listener
  available in this environment — this is a lower-bound intelligibility
  proxy, not a substitute for actually listening to it).
- Result: general greeting + 2/3 industrial sentences came back
  essentially verbatim; one ("Dừng máy..." / "Stop the machine...")
  came back significantly wrong ("Xuân máy..."). Same pattern as MT:
  generic models are noticeably weaker on industrial/safety vocabulary
  specifically — another data point for the glossary engine argument,
  this time on the TTS side too, not just MT.
- Packaged into `models/tts/vi/` (the .onnx + its .onnx.json — Piper's
  phoneme vocabulary lives in the JSON side, not just the ONNX graph).

## Findings worth remembering

- **Big one — another license trap, sneakier than NLLB's:**
  `rhasspy/piper-voices` is tagged `license: mit` at the repo level, but
  that only covers the packaging. Each voice is fine-tuned on a
  specific dataset with its *own* license, declared per-voice in each
  `MODEL_CARD`, and for Vietnamese only one of three available voices
  is actually commercially usable:
  - `vais1000-medium` — CC BY 4.0 — **usable**, also happens to be the
    best quality tier available. Used this one.
  - `vivos-x_low` — CC-BY-NC-SA 4.0 — **blocked** (NonCommercial).
    Ironic: same VIVOS corpus we already use for ASR eval, but
    eval-set use and TTS-training-derivative use are different legal
    situations — one being fine doesn't clear the other.
  - `25hours_single-low` — license **unknown** — can't confidently
    claim usable.
- **Second, separate issue:** actually using any Vietnamese Piper voice
  requires phonemizing text through `espeak-ng` first (confirmed in
  this voice's own config: `"espeak": {"voice": "vi"}` — no way around
  it for this model family). `espeak-ng` is GPL-3.0, and it's *why*
  Piper's own upstream development relicensed from MIT to GPL-3.0
  (`OHF-Voice/piper1-gpl`, Oct 2025) — the maintainers essentially
  conceded the MIT label didn't hold given this embedded dependency.
  Raised with maintainers twice (2023, and again later), never
  resolved either time. Whether shipping this in our on-device binary
  GPL-taints the product is a real open question, not something this
  notebook (or anyone upstream) has answered. Full writeup:
  `docs/decisions/piper-vi-voice-license-trap.md`.
  - Using `espeak-ng` here, in the notebook, to *validate* the model is
    fine (nothing shipped) — this only matters if/when it ends up in
    the actual device binary.
- These two issues compound the same lesson from the NLLB callout in
  AGENTS.md: a clean-looking top-level license tag doesn't mean the
  thing is actually clear to ship — check the *specific* asset and its
  *specific* runtime dependencies, not just the repo badge.

See `docs/TODO.md` for the follow-up list (non-GPL phonemizer research,
human-listener QA pass, Rust port).

---

# 2026-07-15 (cont. 2) — deep research: can the pipeline be improved?

Full writeup: `docs/decisions/pipeline-improvement-research-2026-07.md`.
Short version: mostly confirms current choices are still reasonable;
one real lead, one confirmed dead end.

## The one real lead

**VietASR** (dangvansam/viet-asr, Apache-2.0) — a 13M-parameter *CTC*
model for Vietnamese ASR, vs. PhoWhisper-base's 74M attention-based
seq2seq. The size difference is almost beside the point: CTC has no
autoregressive decode loop at all, which is a structurally different
latency profile than everything else in the pipeline. AGENTS.md itself
names the autoregressive decoder as *the* latency bottleneck — this
sidesteps that bottleneck entirely rather than just shrinking it.
Caveat: couldn't confirm WER vs. PhoWhisper or ONNX export availability
from what's public, so this is "worth prototyping," not "confirmed
better." Also trained partly on 8kHz call-center/noisy audio, which
lines up well with the 75-85dB factory noise target.

## The confirmed dead end

Went looking specifically for a fix to the Piper/espeak-ng GPL trap
from earlier today. No fix exists:
- OpenPhonemizer (the one credible non-GPL espeak-ng alternative) has
  no Vietnamese support and was archived in March 2026.
- Kyutai Pocket TTS (AGENTS.md's own listed TTS upgrade path) doesn't
  support Vietnamese yet either, as of checking directly against its
  HF repos — English-only at the base model, the May 2026 multilingual
  update added French/general voices but not Vietnamese. (Worth
  re-checking later — it doesn't seem to need a phonemizer at all,
  which would make it a genuinely clean fix *if* Vietnamese support
  ever lands.)
- sherpa-onnx (k2-fsa) hit this exact wall building an Apache-2.0-clean
  project and their fix was running `piper-phonemize` as a separate
  external process rather than linking it — confirms that's a real
  pattern people use, but it's an architectural mitigation, not a
  license opinion. Doesn't change what `piper-vi-voice-license-trap.md`
  says needs to happen (an actual legal read).

## Also checked, no action taken

- MT: LLM-based translation (Qwen-MT, GPT-4.1-class) reportedly beats
  classic NMT on VI-EN now, but at LLM scale — not viable in our
  latency/memory budget. Current plan (Opus-MT baseline, VinAI-Translate/envit5
  upgrade path) still holds.
- Qwen3-ASR-0.6B/1.7B (Apache-2.0, 52 languages, Jan 2026, reportedly
  SOTA open ASR) — interesting one-model-for-EN/ZH/KO-and-maybe-VI
  consolidation play for *later*, but 0.6B is ~8x PhoWhisper-base's
  size, not practical now.
- QCS6490 Whisper/QNN deployment: re-checked against newer sources,
  same blocked status as `qcs6490-whisper-deployment-risk.md` already
  says — AI Hub Workbench improved generally, Whisper quantization
  specifically still stuck on AIMET-ONNX's missing aarch64 wheel.
- Denoise/VAD (RNNoise/DeepFilterNet3/Silero VAD): nothing found
  suggests these are outdated — no action.

---

# 2026-07-15 (cont. 3) — denoise + VAD implemented

Moved on to the remaining audio front-end stages. Both went in as real
Rust implementations directly (not just notebook validation first) —
they're much simpler than MT/ASR/TTS: single forward pass per audio
frame, no autoregressive decode loop, no tokenizer.

## What was done

- **Denoise**: RNNoise never shipped an official ONNX export (unlike
  literally everything else so far) — it's traditionally a C library.
  Found `nnnoiseless` instead: a pure-Rust reimplementation
  (BSD-3-Clause) with the trained weights compiled directly into the
  crate, no model file to manage at all. Real implementation in
  `src/stages/denoise.rs`.
- **VAD**: Silero VAD does have an official ONNX (`onnx-community/silero-vad`,
  MIT) — same "just use the pre-made ONNX" pattern as ASR/TTS. Real
  implementation in `src/stages/vad.rs`, stateful (512-sample chunks,
  hidden state carried between calls), trims leading/trailing silence
  only (leaves interior pauses alone).
- Both wired into `Pipeline` in `src/pipeline.rs`; `cargo test` +
  `cargo test -- --ignored` + `cargo clippy` all clean (11 tests total,
  8 always-on + 3 needing real model files).

## Findings worth remembering

- **RNNoise is hardcoded to 48kHz, fixed 480-sample frames — not
  configurable.** Everything else in the pipeline (VAD, ASR) wants
  16kHz. This means the mic capture rate had to change from an
  arbitrary 16kHz (what `AudioCapture` was stubbed with) to 48kHz, and
  the denoise stage now also resamples 48kHz → 16kHz once on its way
  out. Rather than pull in a general-purpose resampling crate (`rubato`'s
  current API turned out to need an extra `audioadapter` buffer-wrapping
  layer for a fairly involved interface), wrote a small self-contained
  windowed-sinc low-pass + decimate-by-3 filter — exact for this one
  fixed 48000/16000 ratio, no new dependency. Two unit tests cover it:
  one checks output length/rate, one checks a 20kHz tone gets
  attenuated while a 1kHz tone survives (i.e. the filter actually
  filters, not just resamples by naive decimation which would alias).
- **Silero VAD is a trained speech/non-speech classifier, not an energy
  detector — learned this the hard way.** First test attempt used a
  loud synthetic 440Hz sine tone padded with silence, expecting the
  tone to register as "speech." It didn't (near-zero probability,
  same as pure silence) — a pure tone has none of the spectral/formant
  structure the model was trained to recognize as speech, no matter how
  loud. Fixed by using a real ~2.4s VIVOS utterance as a committed test
  fixture (`data/eval/vad_sample_vi.wav`) instead. Worth remembering
  for any future audio-stage test: synthetic signals are fine for
  testing signal-processing code (the denoise resampler test uses
  sine tones correctly, because a low-pass filter *is* just frequency
  math), but not for testing anything that's actually a learned
  classifier.
- Both models' source repos also ship quantized variants
  (Silero VAD: int8/uint8/q4/fp16/bnb4) — not yet evaluated, same
  "probably free efficiency win" pattern as ASR's PhoWhisper/Whisper.

See `docs/TODO.md` for follow-ups (DeepFilterNet3 comparison, quantized
VAD, QNN execution provider, `AudioCapture`'s still-stubbed 48kHz
capture).

---

# 2026-07-15 (cont. 4) — ASR ported to Rust

Asked what would most improve the pipeline; picked porting ASR to Rust
over TTS, since MT/denoise/VAD already had real implementations and ASR
was the harder of the two remaining gaps. It's real now:
`src/stages/asr.rs`, encoder + greedy KV-cache decoder (same pattern as
MT) + a from-scratch log-mel feature extractor.

## What was done

- Found `mel_spec` (wavey-ai), the one Rust crate claiming
  Whisper-compatible mel spectrograms.
- **Didn't trust the claim — validated it first, and it was wrong.**
  Dumped a reference mel-spectrogram from Python's
  `WhisperFeatureExtractor` for a real audio sample, computed the same
  thing with `mel_spec` in a scratch crate, diffed numerically before
  writing any real implementation code. First result: mean abs diff
  0.35 — way too large. Found two real bugs in the crate (not usage
  mistakes on my end, verified by reading its source): it uses natural
  log where Whisper's formula needs log10 (~2.3x scale error), and its
  "centered" STFT mode zero-pads the signal edges where Whisper's
  actual preprocessing (via `torch.stft`) reflects them instead. Fixed
  both in my own wrapper code (divide by `ln(10)`; reflect-pad the
  audio myself and turn off the crate's own padding). After fixing:
  mean abs diff ~0.00035. Full writeup:
  `docs/decisions/mel-spec-crate-whisper-compat-bugs.md`.
- Confirmed the fix actually matters, not just numerically: fed the
  Rust-computed features through the real ONNX model and got the exact
  expected transcription, byte-identical to using Python's official
  feature extractor.
- `AsrEngine::transcribe` now runs for real: mel extraction → encoder →
  greedy KV-cache decode (reusing MT's proven algorithm and its
  documented gotchas) → tokenizer decode. Real test in
  `src/stages/asr.rs` against the same committed WAV fixture VAD uses,
  passes: transcribes to the exact expected text.
- `asr_model_dir` replaces `asr_model` in `configs/language-pairs.toml`
  (same reasoning as MT: needs a directory of 3 files, not one).

## Findings worth remembering

- **Don't trust a crate's compatibility claim without checking.** This
  is the second time this exact lesson has paid off this week (first
  with the ONNX merged-decoder gotchas, now with mel_spec) — a
  numerical diff against the real reference implementation, before
  writing the "real" code, caught a bug that would otherwise have
  shipped a Rust ASR stage that runs cleanly and transcribes garbage.
- **New latency concern, confirmed not assumed**: tried feeding the
  ONNX encoder variable-length (unpadded) mel input instead of the full
  30-second pad, since AGENTS.md's ≤1.2s target makes 30s-worth of
  encoder compute for a short utterance look wasteful. It hard-errors —
  the exported graph's positional embeddings are a fixed-size constant
  baked in at export time, despite the graph metadata claiming a
  dynamic sequence-length dimension. So every ASR call really does cost
  a fixed ~30s-worth of encoder compute right now, confirmed not
  theorized. Whether that's actually a problem depends on real
  hardware timing, which hasn't been measured yet.
- Whisper's special-token scheme (decoder_start/eos/task/language
  tokens, suppress_tokens, begin_suppress_tokens) is identical between
  PhoWhisper-base and generic Whisper-small — both fine-tunes of the
  same multilingual Whisper vocab, so the same Rust constants serve
  both models, only the language token id changes per language.

See `docs/TODO.md` for follow-ups (30s-padding latency measurement,
ZH/KO Rust-path testing, QNN execution provider).

---

# 2026-07-15 (cont. 5) — the missing notebook: 01_denoise_vad.ipynb

`experiments/README.md` documented `01_denoise_vad.ipynb` from the very
first session as part of the expected notebook layout, but it never
actually got created — denoise and VAD went straight to real Rust
implementations (see the "denoise + VAD implemented" entry above)
without a Python companion notebook first, unlike every other stage.
Wrote and ran it now.

## What was done

- VAD section: straightforward, same pattern as ASR/TTS — re-ran the
  Silero VAD validation (real VIVOS speech vs. silence vs. quiet white
  noise) as a proper committed notebook instead of a scratch script,
  repackaged `models/vad/`.
- Denoise section: tried to give RNNoise the same Python-validation
  treatment as everything else, and hit real dead ends on both attempts
  — documented rather than hidden:
  - `pyrnnoise` (wraps the actual C RNNoise library): broken import
    chain, `ModuleNotFoundError: No module named 'av.option'` — an
    unmaintained transitive dependency (`audiolab`) incompatible with
    the `av` version available now.
  - `deepfilternet` (AGENTS.md's denoise *upgrade path*, tried as an
    alternative Python baseline): also broken —
    `torchaudio.backend.common` doesn't exist in current `torchaudio`
    anymore, and no older `torchaudio` compatible with our `torch` is
    available to install instead.
  - Found and inspected a legitimate pre-exported DeepFilterNet3 ONNX
    (`soniqo/DeepFilterNet3-ONNX`, MIT, validated against PyTorch to
    ~1e-6 in its own export manifest) as a workaround — but its input
    is DeepFilterNet's own ERB/STFT features, not raw audio, so using
    it means implementing that whole feature-extraction pipeline from
    scratch. Comparable scope to ASR's mel-spectrogram work
    (`docs/decisions/mel-spec-crate-whisper-compat-bugs.md`). Documented
    as a real follow-up, not attempted in this pass.

## Findings worth remembering

- Both mature-sounding Python denoise packages being simultaneously
  broken in a reasonably current environment is itself useful evidence,
  not just an annoyance — it's a real signal that the Python
  RNNoise/DeepFilterNet ecosystem is less actively maintained than
  Whisper's or Piper's, and retroactively justifies going straight to a
  from-scratch Rust crate (`nnnoiseless`) for RNNoise rather than
  fighting Python tooling first.
- General lesson for this project's docs going forward: when something
  doesn't work, the honest move is recording *why* it failed (exact
  error, exact cause) in the notebook itself, not silently skipping the
  cell or leaving it out. A future reader (or future me) hitting the
  same wall gets to skip straight to "already tried, here's why it
  doesn't work" instead of rediscovering it.

No `docs/TODO.md` items newly opened here beyond what was already
tracked for DeepFilterNet3 — this pass just made the existing "not yet
evaluated" note concrete about *why*.

---

# 2026-07-15 (cont. 6) — competitor landscape research

Asked to research other brands' products in this space, note their
use-cases, and evaluate relevance to OneVoice. Full writeup:
`docs/decisions/competitor-landscape-2026-07.md`. This one matters more
for the Phase 2 business pitch than for engineering decisions.

## The headline finding

Researched ~10 products across two categories — consumer/travel
translators (Vasco, Pocketalk, Timekettle, iFlytek, ili, Muama Enence,
Waverly Labs Pilot) and industrial PTT-translation platforms (Weavix
Walt Smart Radio, Relay). **The industrial category is a direct,
proven precedent for exactly what OneVoice is building — Weavix alone
has served 2 billion translated messages on factory floors — but every
single competitor found in *both* categories is cloud/connectivity-
dependent in practice, including ones marketed as "offline."** Weavix's
Walt radio explicitly falls back to plain analog voice (FRS) when the
network drops — translation stops working entirely. Same story for
Relay, Vasco, Pocketalk, Muama Enence. Only Timekettle's earbuds are
genuinely offline, and only for 8 non-Vietnamese language pairs.

No genuinely-offline industrial translator was found anywhere. That's
either a real, defensible market gap for OneVoice's "zero connectivity
required" positioning, or a reason nobody's built one — worth treating
as an open question for the business pitch to address head-on, not an
assumption to lean on uncritically.

## Findings worth remembering

- **Vasco's own data undercuts headline accuracy claims industry-wide**:
  they cite "96% accuracy" from the WMT2022 text benchmark, but their
  own reviewers report real-world speech accuracy dropping to 78-84% in
  65+dB street noise. Useful for calibrating what AGENTS.md's ≤15% WER
  @ 75-85dB target should be benchmarked against — a competitor's
  marketing number, not their real-world number.
- **Waverly Labs Pilot was discontinued in April 2024** — the company
  shut off its translation service centrally, bricking every unit
  already sold. Concrete, citable evidence for a pitch argument only a
  local-only architecture can honestly make: this device can't be
  bricked by a vendor decision.
- **No competitor treats Vietnamese as a priority language** — not
  Vasco, Pocketalk, iFlytek, Weavix, Relay, or even Timekettle's
  offline set. Reinforces that PhoWhisper/VinAI's Vietnamese lineage is
  a genuine, uncontested specialization, not a "nice to have."
- **iFlytek's audio front-end (4-mic U-shaped array) is the closest
  real technical comparator** to AGENTS.md's noise-first design —
  worth a direct benchmark once hardware exists, since they use 2x the
  mic count AGENTS.md currently plans.
- **Every competitor's business model is subscription/data-plan
  attached to connectivity** — structurally incompatible with
  OneVoice's offline architecture. This needs an explicit, prepared
  answer for the Phase 2/3 pitch (judging weighs business viability at
  25%), not something to leave implicit.
- **Could not independently verify** AGENTS.md's claim that Vbee is "a
  Vietnamese startup building offline voice solutions on Qualcomm edge
  platforms" — everything findable about Vbee is their cloud TTS SaaS
  product. Not contradicted, just unconfirmed by this research; flagged
  in docs/TODO.md to verify directly before repeating it in Phase 2/3
  materials.

See `docs/decisions/competitor-landscape-2026-07.md` for the full
per-product breakdown and comparison table.

---

# 2026-07-16 — first real end-to-end pipeline run

Asked what to do next; picked the highest-leverage gap identified while
answering that: every stage had been validated in isolation, but
`Pipeline::run_once()` had never actually executed — `audio_io.rs` was
still a `todo!()` stub. Wired it up and ran it for real.

## What was done

- `src/stages/audio_io.rs`: real implementation, not a mock — reads/
  writes WAV files via `hound` (moved from dev- to a real dependency).
  Stands in for live mic/speaker capture until real hardware (Rubik Pi
  3) is in hand; designed as the one piece to swap out later, everything
  else in `Pipeline` is unaffected.
- `Pipeline::new` and `main.rs` now take input/output WAV paths as CLI
  args instead of looping forever on live PTT (which doesn't make sense
  for file-based testing) — `cargo run -- <in.wav> <out.wav>`.
- Created a real 48kHz test fixture (`data/eval/ptt_sample_vi_48k.wav`,
  the existing VIVOS sample properly resampled up, not synthetic) since
  RNNoise needs 48kHz input and the only committed fixture before this
  was 16kHz.
- Ran it. Real result, through the actual binary, not a test harness:
  ```
  ASR: "nửa vòng trái đất hơn bẩy năm."
  MT:  "Half the world is more than five."
  ```
  capture → denoise → VAD trim → ASR → MT all composed correctly on the
  first real run. Then hit TTS, exactly where expected.
- At TTS: asked how to handle it rather than deciding unilaterally,
  since real Rust TTS means adding a GPL-3.0 phonemizer to `src/`'s
  actual dependencies (not just a research notebook) — a materially
  different, harder-to-reverse choice than using it in Python for
  validation only. Chose to stop the milestone at text output rather
  than add the GPL dependency or fake the audio. Turned `tts.rs`'s
  `todo!()` (a raw panic) into a proper `Result::Err` with a message
  pointing at the license doc, so the binary now completes everything
  it honestly can and exits cleanly instead of crashing.

## Findings worth remembering

- The milestone revealed a real, minor ASR artifact worth knowing
  about: transcribed "bẩy" instead of the ground-truth "bảy" (a
  diacritic/tone variant) — most likely introduced by the 16kHz→48kHz
  upsample step this test fixture went through, compounding with
  ASR's normal error rate. Small, expected, and a genuine illustration
  of how resampling artifacts can propagate through the real chain —
  not visible when testing each stage in isolation with its "native"
  sample rate.
- `main.rs`'s error handling already worked correctly the moment
  `tts.rs` returned a real `Err` instead of panicking — no changes
  needed there, it was designed for this from early on.
- Worth remembering as a pattern for the *next* boundary hit like this:
  a `todo!()` is the right stub *while a stage is actively being
  built*; once a boundary becomes a deliberate, known, possibly
  long-lived stopping point (like TTS pending legal review), converting
  it to a structured `Result::Err` is a small, honest improvement — the
  code still doesn't do the thing, it just says so cleanly instead of
  crashing.

See `docs/TODO.md`'s "End-to-end pipeline wiring" section for
follow-ups (live hardware capture, latency instrumentation — this run
proved composition, not performance).

---

# 2026-07-16 (cont.) — first real latency numbers, glossary data (not enforcement), a real CLI

Asked to keep going and flag anything worth knowing. Three things done:
timing the actual pipeline for the first time, trying (and rejecting)
the obvious glossary-enforcement shortcut, and a proper `clap` CLI.

## Latency: the decoder, not the padded encoder, is the bigger cost

Added per-stage `Instant` timing to `Pipeline::run_once` and
`AsrEngine::transcribe`, ran it several times on this dev machine
(release build). Real numbers for a ~1.7s VI utterance producing 16
output tokens:

- capture/denoise/VAD: <15ms combined (as expected — lightweight)
- ASR encoder (the fixed 30s-padded pass): ~130-310ms
- ASR decoder (autoregressive, 16 tokens): ~250-405ms — **bigger than
  the encoder**
- MT: ~90-240ms
- Through MT total: ~530-970ms (real variance between runs on a shared
  machine, not a device-representative number — this is a much faster
  CPU than the target Kryo 670, and doesn't touch the Hexagon NPU at
  all)

The genuinely useful finding: I'd assumed (reasonably, given how loudly
the "30s-padding is forced" discovery stood out earlier) that the fixed
encoder cost was the dominant problem. It isn't — the autoregressive
decoder loop is bigger, even generating only 16 tokens. This directly
strengthens the case for evaluating VietASR (the CTC alternative
bookmarked in the 2026-07-15 research pass) — a model with *no*
autoregressive decode loop at all could plausibly save more time than
fixing the 30s-padding issue would, which reframes that as the
higher-value experiment to run next, not just "another option."

## Glossary: tried the cheap fix, it doesn't work reliably

Tested whether substituting a glossary source term with a placeholder
string before MT, then restoring the correct term after, would work as
poor-man's constrained decoding — against the real, already-validated
Opus-MT vi-en model, on the known failure case ("xe nâng" / forklift
mistranslated as "lift"). Tried 6 placeholder candidates; only 2
survived translation verbatim, the other 4 got silently mangled
("XNAAA" → "XAAA's", "FORKLIFTTOKEN" → "FORKIFTRON", etc.) — no way to
predict which without testing each one against the specific model.
Full writeup: `docs/decisions/glossary-runtime-enforcement-2026-07.md`.

Rejected it rather than ship something with an unknown, silent failure
rate for a *safety-critical* glossary — a device that sometimes
mistranslates "forklift" without any signal that it did is worse than
one that's honestly not enforcing the glossary yet.

Did make real, uncontroversial progress on the part that isn't
risky: `Glossary::load` now actually reads a real termbase
(`data/glossary/termbase.tsv`, 10 VI/EN/ZH/KO industrial terms) instead
of being a no-op stub. Not wired into MT — this is data infrastructure
for whenever the real fix (LoRA fine-tune, per AGENTS.md's own plan)
happens, not a substitute for it. ZH/KO terms in that file are my own
best-effort translations of common vocabulary, not reviewed by a native
speaker — flagged in the file/decision doc as needing review before
being trusted as safety-authoritative.

## CLI

Added `clap` (derive API) — `cargo run -- --help` now gives real usage,
`--source`/`--target` (default vi/en), and
`--configs-dir`/`--models-dir`/`--data-dir` overrides. Small thing, but
the language pair was hardcoded in `main.rs` before this, which was a
real gap given the project targets 6 directions.

See `docs/TODO.md` for follow-ups (VietASR prototyping now has stronger
justification, LoRA fine-tune for real glossary enforcement, live
hardware capture).

## 2026-07-16: scripts/, and the glossary fix that actually worked

Added three real scripts (not scaffolding): `run_notebook.sh` (renamed
from `run-experiments.sh`), `run_pipeline.sh` (builds `--release`,
treats `onevoice` as a real CLI app instead of typing `cargo run --
...` by hand), `setup_models.sh` (downloads vad/asr/tts directly from
the HF Hub — no Jupyter needed for those — and shells out to
`run_notebook.sh 03_mt` for MT, the one stage without a pre-made ONNX).

Then went back to the glossary question, since the placeholder-
substitution rejection above left it genuinely unsolved. Tried decoder
logit biasing this time — bias the decode toward a glossary target
term's tokens, but only when the term is actually missing from a cheap
unbiased pass first. Took three real, sequentially-caught bugs to get
right, each one a case of "the first result looked like a clean
negative, don't trust that without checking why":

1. First version: bias only ever fired for single-token terms. Checked
   the actual per-step logits (not just trusted "bias=14 had zero
   effect") and found the bug — multi-token terms like "forklift"
   (`_fork` + `lift`) never got their first token biased at all, so the
   sequence could never start.
2. Fixed that, got a *new* failure: repetition loops
   ("fork fork fork fork..."), because the fix biased every matching
   position independently with no exclusivity — "restart" and
   "continue" competed on the same step.
3. Fixed that (track one current match position per sequence, not
   independent per-position checks) — clean output, 100% term accuracy
   on a 10-sentence test set. But biasing *every* glossary-flavored
   sentence, even ones already correct, cost fluency for no reason
   ("This area is dangerous." -> "danger zone."). Final fix: only run
   the biased pass when the unbiased pass actually got something wrong.

Ended up with something real: 50% -> 100% term accuracy, the 5/10
already-correct sentences untouched, ported to `src/stages/mt.rs` with
Rust tests that exact-match the Python-validated outputs. Not a
finished story — corrected sentences are grammatically rougher than
fluent translation (dropped articles, mostly), not native-speaker
reviewed, vi-en only. But it's a working stopgap where the previous
conclusion was "nothing cheap works, wait for LoRA fine-tuning." Full
writeup: `docs/decisions/glossary-runtime-enforcement-2026-07.md`
(updated in place, not a new file, since it's the same question).

The meta-lesson, consistent with everything else in this file: a
prototype that "doesn't work" on the first try is not the same as a
technique that doesn't work — worth one targeted look at *why* before
writing it off, as long as the fix is specific and understood rather
than a blind guess.

## 2026-07-17: TTS actually speaks now — first real audio out of this pipeline

User call: since this is a hackathon, not a shipping product, stop
treating GPL as a build blocker (it binds at commercial distribution,
not a prototype) — go implement TTS for real instead of leaving it
stubbed on the license question. Recorded as a standing preference.

Before writing any phonemization code by hand, checked whether a crate
already covers Piper — it does. `piper-rs` (MIT) loads a Piper
`.onnx`+`.onnx.json` directly and does espeak-ng phonemization + VITS
inference internally, using the *same* `ort` version this project
already depends on. Its default build failed on a real bug, not a
missing dependency: `Error processing file '.../phsource/intonation':
No such file or directory` — the file exists in the package; the error
path (`target/debug/../phsource/intonation`) is a broken relative path
in the crate's own CMake step. `default-features = false` (skips the
`compile-espeak-intonations` feature) works around it — but that has a
side effect: it also skips compiling espeak-ng's core runtime data
(`phontab`/`phondata`/`phonindex`), which then makes espeak-ng hard
*abort* at runtime (no Rust panic, signal 5) with zero error output —
a much nastier failure mode than the build error. Sourced a complete
copy from the `piper-tts` PyPI package (already installed, bundles its
own working `espeak-ng-data/`) rather than adding a new system
dependency for something already on hand.

Korean (MMS-TTS) has no covering crate at all, so that side was
hand-rolled: the `uroman` Rust crate (verified byte-identical to
Python's `uroman` output before trusting it) + a direct port of HF
transformers' `VitsTokenizer` tokenization logic (read from the actual
library source, not guessed) + raw `ort` inference. Captured the real
Python `input_ids` for the notebook's KO test sentences first and
hardcoded them as the Rust test's expected values — the port matches
exactly, not approximately.

Result: all four languages (VI/EN/ZH via Piper, KO via MMS) produce
real, round-trip-ASR-verified audio through the actual Rust path, and
the full pipeline runs capture→denoise→VAD→ASR→MT→TTS→playback for the
first time — measured at ~550-600ms button-release→audio on CPU, well
under AGENTS.md's ≤1.2s target. Re-packed and re-uploaded the
`models-v1` release bundle to include the new `espeak-ng-data`;
verified the real `gh release download` path end-to-end with a
checksum match, not just the packing step. Full writeup:
`docs/decisions/tts-rust-port-2026-07.md`.
