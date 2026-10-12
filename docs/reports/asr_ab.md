# ASR A/B (VIVOS test slice, CPU)

Workstream 2 of `docs/decisions/improvement-roadmap-2026-10.md`. 20 VIVOS
utterances, AWGN, greedy decode, WER case/punctuation-normalized
(`scripts/evaluate_asr.py`).

| model | arch | clean WER | 5 dB WER | latency/utt | RTF |
|---|---|---|---|---|---|
| vinai/PhoWhisper-base | Whisper seq2seq | **2.19%** | **9.84%** | 527–562 ms | 0.18 |
| Qwen/Qwen3-ASR-0.6B-hf | speech-LLM | 4.92% | 10.38% | ~2250 ms | 0.72 |
| dragonSwing/wav2vec2-base-vietnamese | CTC | 10.38% | 22.95% | 63–102 ms | **0.02–0.03** |

RTF = wall-clock / audio-seconds (<1.0 = faster than real time).

## Analysis

- **Keep PhoWhisper-base for VI accuracy.** It beats Qwen3-ASR-0.6B on clean
  WER (2.19% vs 4.92%), ties at 5 dB (9.84% vs 10.38%), and is ~4× faster.
  Qwen3-ASR's value is breadth (one model, 30+ languages), not VI accuracy —
  a swap is **not** justified for VI, and at 0.72 RTF it would be near/over
  real-time on the much weaker QCS6490 CPU.
- **CTC is the latency play, not the accuracy play.** `wav2vec2-base-vi` is
  ~5× faster (RTF 0.02) but much worse (10.38% / 22.95%). Same trade-off as
  AGENTS.md predicts: no autoregressive loop, lower accuracy.
- **5 dB without denoising:** PhoWhisper 9.84% already clears the AGENTS
  "≤15%" bar. Note the existing `asr_noise_robustness.md` says 12.32% at 5 dB
  (different checkpoint/sample size) — reconcile before quoting either.
- **Not tested:** ZH/KO ASR (Whisper-small), real machinery noise (AWGN here),
  and other CTC checkpoints (a better VI CTC would change the calculus).

## Rust runtime — CTC backend is real

The CTC path is implemented in the runtime (`src/stages/ctc.rs`,
`CtcAsrEngine`; `Asr::load` auto-selects it when the model dir has
`model.onnx`). Full Rust pipeline on the same WAV
(`asr_model_dir = "asr/vi-ctc"`):

| stage | Whisper (PhoWhisper) | CTC (wav2vec2-base-vi) |
|---|---|---|
| ASR | ~360–420 ms | **57 ms** |
| through-MT | ~440 ms | 93 ms |
| full cycle (through TTS) | ~485 ms | **181 ms** |

CTC buys ~300 ms end-to-end (clears the 0.8 s stretch budget with room) at a
real WER cost. Whisper stays the default; CTC is the latency-fallback backend.

## Method

- VIVOS test slice (`quocanh34/viet_vivos`), soundfile-decoded; AWGN identical
  to `evaluate_noise_robustness.py`; `jiwer` WER.
- Qwen3-ASR via its native multimodal API (`transformers>=5.13`, forced
  `language="vi"`); CTC via the standard ASR pipeline; both compared against
  PhoWhisper in the same harness.
- The Rust CTC export: `scripts/edge/export_ctc_asr.py`
  (`dragonSwing/wav2vec2-base-vietnamese` → `models/asr/vi-ctc/`).
