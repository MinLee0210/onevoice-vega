# Pipeline improvement research (2026-07-15)

Deep-research pass across all stages, checking whether anything newer
or better exists since AGENTS.md's model stack was written. Bottom
line: **one genuinely actionable finding (VietASR), one confirmed dead
end (no fix for the Piper GPL phonemizer trap), everything else either
confirms the current plan or isn't practical for this device.** No
forced changes needed to what's already been validated.

## Worth prototyping: VietASR (CTC, 13M params) as a VI ASR alternative

[dangvansam/viet-asr](https://github.com/dangvansam/viet-asr) —
Apache-2.0, NeMo/QuartzNet-based, **13M parameters** (vs. PhoWhisper-base's
74M), trained on ~100h of diverse Vietnamese speech (YouTube, radio,
8kHz call-center audio, TTS data, VLSP/VIVOS/FPT).

Why this matters more than the parameter count alone suggests: it's
**CTC, not attention-based seq2seq**. AGENTS.md itself identifies the
autoregressive decoder as the latency bottleneck ("ASR decoder
(autoregressive) is the bottleneck — keep models small"). A CTC model
has no autoregressive decode loop at all — one forward pass over the
audio produces per-frame character probabilities, decoded with a cheap
greedy/beam CTC decode, not token-by-token generation with a KV cache.
That's a structurally different latency profile than everything else
in the current ASR stage, not just a smaller version of the same thing.

Caveats found, not yet resolved:
- Could not confirm WER numbers vs. PhoWhisper/Whisper from available
  sources — CTC models are generally somewhat less accurate than
  attention-based ones on noisy/accented speech, so this is a real
  tradeoff to measure, not a free win.
- Could not confirm ONNX export availability — would need the same
  export treatment MT got (`experiments/03_mt.ipynb` pattern), not a
  ready-made ONNX like PhoWhisper/Whisper were.
- Trained partly on 8kHz call-center audio and diverse noisy sources,
  which is actually a good sign for the "75-85dB factory noise" target
  in AGENTS.md's performance targets — PhoWhisper's 844h dataset
  composition isn't confirmed to be as noise-diverse.

**Recommendation:** worth a `experiments/`-style validation pass (WER
on the same VIVOS slice used in `02_asr.ipynb`, check ONNX
exportability) before committing either way. Not urgent, but higher
expected value than most other things researched here.

## Confirmed dead end: no license-clean phonemizer for Vietnamese Piper TTS

Directly chasing a fix for `docs/decisions/piper-vi-voice-license-trap.md`.
Checked three angles, none pan out:

- **OpenPhonemizer** (BSD-3-Clause-Clear, marketed as an espeak-ng
  drop-in replacement) — English only, and the repo was **archived in
  March 2026** with the maintainer saying they're building a different
  replacement elsewhere. Not usable for Vietnamese now or going forward
  on this codebase.
- **Kyutai Pocket TTS** (already AGENTS.md's listed TTS upgrade path) —
  checked its actual language support directly: English-only at the
  base model, and the May 2026 multilingual expansion
  (`kyutai/tts-voices`) added English/French/general sources but **no
  Vietnamese**. Architecturally interesting for later though — it
  appears to take raw text directly (no separate phonemizer step
  mentioned), MIT-licensed code, and its voice licenses are CC0/CC-BY/CC-NC
  per-voice (same "check the specific asset" pattern as Piper, but no
  GPL dependency at all). Worth re-checking if/when Vietnamese support
  lands.
- **sherpa-onnx** (k2-fsa) hit this exact problem and documented their
  fix: [issue #3731](https://github.com/k2-fsa/sherpa-onnx/issues/3731)
  explicitly removed espeak-ng "to remain fully compatible with the
  Apache-2.0 license," replacing it with either (a) a hand-built
  lexicon file (word→phoneme dictionary — a real project on its own for
  Vietnamese, not a drop-in fix) or (b) running `piper-phonemize` as a
  separate external process rather than a linked library. Confirms our
  decision doc's "separate process" idea is a real pattern serious
  projects use, but it's an architectural mitigation, not a legal
  resolution — still doesn't answer whether that isolation is
  sufficient, that's still a legal question, not an engineering one.

**Recommendation: no code change from this.** The risk documented in
`piper-vi-voice-license-trap.md` stands as written. Don't spend more
time searching for a silver bullet — get an actual legal read on the
external-process isolation pattern, since that's the only mitigation
that's actually real right now.

## Confirmed still-blocked: QCS6490 Whisper/QNN deployment

Re-checked `docs/decisions/qcs6490-whisper-deployment-risk.md` against
newer sources. Same picture: AI Hub Workbench has generally improved
for QCS6490 (QNN upgraded to 2.27, RB3 Gen 2 Vision Kit board support
since mid-November), but the Whisper-specific quantization path is
still blocked on AIMET-ONNX having no aarch64 Linux wheel. No new
information changes that risk assessment.

## Checked, no action needed (current plan already sound)

- **MT**: Qwen-MT and LLM-based translation (GPT-4.1/Claude-class)
  reportedly beat classical NMT on VI-EN quality now, but those are
  LLM-scale models — not viable within the latency/memory budget for
  an on-device beam=1 greedy decoder. AGENTS.md's listed upgrade path
  (VinAI-Translate/envit5, still on-device-sized) remains the right
  call; nothing found supersedes it at our size class.
- **Qwen3-ASR-0.6B/1.7B** (Apache-2.0, 52 languages, released Jan 2026,
  reportedly SOTA open ASR) — interesting for a *future* consolidation
  play (one model instead of separate PhoWhisper + Whisper-small for
  EN/ZH/KO), but 0.6B is already ~8x PhoWhisper-base's size; not
  practical for the current latency/memory budget. Worth bookmarking,
  not acting on.
- **Denoise/VAD**: nothing found suggests RNNoise/DeepFilterNet3/Silero
  VAD are outdated choices — no action.
