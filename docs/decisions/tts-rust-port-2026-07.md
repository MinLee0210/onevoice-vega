# TTS Rust port: what actually shipped, and the real bugs along the way

`src/stages/tts.rs` went from a stub to real, tested synthesis for all
four target languages (VI/EN/ZH via Piper, KO via MMS-TTS) in one pass.
This doc captures the real findings — a third-party crate build bug, a
runtime-data sourcing problem it caused, and a from-scratch tokenizer
port verified byte-identical to Python — the kind of thing worth
knowing before touching this stage again.

## The crate landscape (checked before writing any phonemization code)

Before assuming this needed a from-scratch espeak-ng FFI binding (the
original plan when the stage was still a stub — see its earlier module
docs), a real search turned up crates that already do most of the work:

- **[`piper-rs`](https://github.com/thewh1teagle/piper-rs)** (MIT) —
  loads a Piper `.onnx` + `.onnx.json` directly, does espeak-ng
  phonemization internally (via its own `espeak-rs`/`espeak-rs-sys`
  dependencies), and runs VITS inference via `ort` — the *same* `ort`
  version this project already depends on (`2.0.0-rc.12`), so it composes
  without a version conflict. API: `Piper::new(onnx_path, config_path)`,
  `.create(text, is_phonemes, speaker_id, length_scale, noise_scale,
  noise_w) -> (Vec<f32>, u32)`.
- **[`uroman` (Rust)](https://github.com/o24s/uroman-rs)** (Apache 2.0)
  — a from-scratch Rust reimplementation of the Python `uroman`
  package, self-contained (no Perl, no external data downloads).
  Verified to produce **byte-identical** romanized output to Python's
  `uroman` on both KO test sentences before being trusted.
- **Nothing wraps MMS-TTS.** No crate does the VITS-family model
  *without* espeak (MMS's tokenizer is a plain char-level lookup +
  romanization front-end, architecturally different from Piper's
  phoneme-based one). This half was always going to be hand-rolled;
  the crate search just confirmed there was no shortcut to skip.

## Real bug #1: `piper-rs`'s default build fails on the crates.io package

`cargo build` with `piper-rs = "0.2"` (default features) fails inside
`espeak-rs-sys`'s CMake step:

```
Error processing file 'target/debug/../phsource/intonation': No such file or directory.
```

This looks like a missing source file. **It isn't** — `phsource/intonation`
genuinely exists in the packaged crate
(`~/.cargo/registry/.../espeak-rs-sys-0.2.0/espeak-ng/phsource/intonation`,
confirmed by direct inspection). The error path itself is the tell:
`target/debug/../phsource/intonation` is a broken relative path (CMake
resolving `..` from somewhere inside the build tree rather than the
actual vendored source root) — a real bug in the crate's vendored
CMakeLists for at least this platform/version, not an artifact issue.

**Fix**: `piper-rs = { version = "0.2", default-features = false }`
(see `Cargo.toml`) skips the `compile-espeak-intonations` feature,
which is the only thing gated behind that broken CMake step. Cost:
lose synthesized prosody *intonation curves* specifically — phonemization
itself is unaffected. Confirmed via a throwaway scratch crate before
touching this project's real `Cargo.toml` (methodology consistent with
this repo's other crate/library validations).

## Real bug #2 (a side effect of the fix): missing espeak-ng runtime data

Disabling `compile-espeak-intonations` also disables the *entire* CMake
data-compilation step, not just intonations — the crate's `espeak-ng-data/`
that ships in the package only has `lang/` and `voices/` (static files,
copied as-is); `phontab`, `phondata`, `phonindex` are all compiled from
`phsource/` by the same disabled build target. Without them,
`espeak_Initialize` hard-aborts at runtime (`SIGTRAP`, no Rust panic —
signal 5 on aarch64, since `std::intrinsics::abort()` maps to `brk`
there) with zero output, which is a much worse failure mode than the
build-time error above — worth knowing if this ever regresses silently.

**Fix**: source a *complete* `espeak-ng-data/` from elsewhere and point
`espeak-rs` at it via the `PIPER_ESPEAKNG_DATA_DIRECTORY` env var
(`espeak-rs`'s own `locate_espeak_data()` reads this — see its source,
not documented in its README). The `piper-tts` PyPI package (already a
project dependency for the Python-side TTS validation) bundles a
complete, correct copy at `site-packages/piper/espeak-ng-data/` — reused
directly rather than requiring a new system dependency (e.g.
`brew install espeak-ng`) for something we already have a working
source for. This data is pure lookup tables, not compiled code, so the
macOS-sourced copy is expected to work unchanged on the eventual aarch64
Linux target — not yet verified on real hardware, flagged as an
assumption, not a confirmed fact.

Shipped as `models/tts/espeak-ng-data/` (gitignored like every other
model artifact, ~19MB, sourced via `scripts/setup_models.sh tts`) —
shared across all three Piper voices, not per-language.
`Pipeline::new` sets the env var once via `TtsEngine::load` before any
phonemization call (espeak-ng initializes lazily, once per process, via
a `OnceLock` inside `espeak-rs`).

## MMS Korean: verified byte-identical before trusting it

No crate exists for this path, so `src/stages/tts.rs::mms_tokenize` is a
direct port of `transformers.VitsTokenizer.prepare_for_tokenization` +
`_tokenize` (read from the actual HF `transformers` source, not
guessed): lowercase → `uroman` romanize (no explicit language code,
matching the Python call exactly) → drop any character outside the
model's `vocab.json` → intersperse the blank token (id 0) between every
remaining character. Before writing the Rust version, the *exact*
Python `input_ids` for both `experiments/04_tts.ipynb` KO probe
sentences were captured and hard-coded as the test reference (see
`mms_tokenize_matches_the_python_tokenizer_exactly` in `tts.rs`) — the
Rust port matches them exactly, not approximately.

The ONNX export itself (`torch.onnx.export` with `noise_scale=0` and
`noise_scale_duration=0`, done in `experiments/04_tts.ipynb` section 5
and mirrored in `scripts/setup_models.sh`) makes MMS synthesis fully
deterministic, unlike Piper's VITS (whose `.onnx.json` configs default
`noise_scale=0.667` — genuinely stochastic, sampled inside the graph).
This is why the Rust test suite can assert an *exact* sample count for
MMS (`synthesizes_ko_speech_via_mms_matching_expected_sample_count`)
but only structural sanity (non-silent, plausible duration) for Piper.

## Result

All four languages produce real, round-trip-ASR-verified audio through
the actual Rust inference path (not just the Python notebook). Full
pipeline (capture → denoise → VAD → ASR → MT → TTS → playback) measured
at ~550-600ms button-release → synthesized-audio on a CPU dev machine —
comfortably under AGENTS.md's ≤1.2s target, though not yet measured on
the Hexagon NPU target hardware. See `docs/TODO.md`'s TTS section for
the day-to-day summary and `src/stages/tts.rs` module docs for the
in-code version of this write-up.
