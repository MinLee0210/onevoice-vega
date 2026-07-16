# `mel_spec` crate: two bugs found porting ASR to Rust, both fixed

Found while implementing `src/stages/asr.rs`. The crate advertises
itself as producing "Mel spectrograms aligned to the results from the
whisper.cpp, pytorch and librosa reference implementations" — that
claim doesn't hold for the actual code path used
(`BatchLogMelSpectrogram::compute`), at least in v0.4.0. Recorded here
because it's exactly the kind of bug that doesn't fail loudly: the code
compiles and runs, and produces syntactically valid but semantically
wrong mel features, which downstream just reads as "the ASR model got
worse for no reason" if it's ever hit again (e.g. after a crate
upgrade).

## How this was caught

Before wiring anything into the real ASR engine: dumped a reference
mel-spectrogram from `transformers.WhisperFeatureExtractor` in Python
for a real audio sample, computed the same thing in a scratch Rust
crate, and diffed them numerically. First attempt: mean abs diff 0.35,
max diff 1.66 — way too large to be floating-point noise, and the first
several Rust-computed frames were suspiciously identical to each other
in a way the Python reference wasn't. That pattern was the signal
something was structurally wrong, not just imprecise.

## Bug 1 — natural log instead of log10

`BatchLogMelSpectrogram::compute_flat_with_scratch` (what
`.compute()` actually calls) does:
```rust
features[...] = (scratch.mel_energy[mel_idx] + self.config.log_zero_guard).ln();
```
Whisper's log-mel formula is base-10 (`torch.clamp(mel, min=1e-10).log10()`
in `openai/whisper`). `ln(x) = log10(x) * ln(10)` — a roughly 2.3x scale
error on every value. This alone accounted for most of the discrepancy.
(There's a *different* code path in the same crate,
`SparseMelFilterbank::project_power_f64`, that does use `.log10()` —
easy to assume that's what `.compute()` calls too if you only skim the
source. It isn't.)

**Fix:** divide the crate's raw output by `std::f32::consts::LN_10`.

## Bug 2 — zero-padding instead of reflect-padding for centered STFT

`BatchLogMelConfig { center: true }` pads the signal edges with
**zeros** (`prepare_padded_waveform`, `padded.resize(..., 0.0)`).
PyTorch's `torch.stft(center=True)` — what Whisper's own preprocessing
actually uses — pads by **reflecting** the signal at the edges instead
(`pad_mode='reflect'` is `torch.stft`'s default). This only affects
frames within `n_fft/2` samples of the start/end (12.5ms each side at
Whisper's settings), but that's still visibly wrong on the very first
frame, which is disproportionately important since VAD-trimmed
utterances start right at speech onset.

**Fix:** reflect-pad the audio manually (`reflect_pad` in
`src/stages/asr.rs`) and set `center: false` so the crate just consumes
what it's given without adding its own (wrong) padding.

## One more (not a bug, just a Whisper quirk to know about)

Centered STFT over a 30s (480,000-sample) clip at these settings
naturally produces 3001 frames, but Whisper's own code explicitly drops
the last one (`stft[..., :-1]` in `openai/whisper`) to land on exactly
3000. `mel_spec` doesn't do this trim — has to be done after the fact.

## After both fixes

Mean abs diff dropped to ~0.00035, max diff to ~0.035 (out of a value
range of roughly -1 to 1.5) — small enough to be ordinary FFT
implementation differences (rustfft vs. whatever numpy/torch use
internally), not a structural issue. Confirmed end-to-end, not just
numerically: fed the Rust-computed features through the real ONNX
encoder/decoder and got the exact expected transcription, byte-for-byte
identical to using the official Python `WhisperFeatureExtractor`. See
the `transcribes_vi_speech_matching_the_known_reference` test in
`src/stages/asr.rs`.

## Takeaway

A crate's marketing claim ("compatible with X") is not validation.
Before trusting any third-party reimplementation of a specific
reference algorithm — especially one as fussy about exact conventions
as Whisper's audio preprocessing — dump the reference implementation's
actual output and diff against it numerically, the same way MT's
tokenizer conversion and decode loop were validated against PyTorch
before being ported (see `docs/decisions/onnx-runtime-merged-decoder-gotchas.md`).
Skipping that step here would have shipped a Rust ASR stage that
"works" (runs without error) but transcribes garbage.
