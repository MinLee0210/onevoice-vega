# ASR noise robustness — real industrial noise (vinai/PhoWhisper-base)

| condition | SNR | WER |
|---|---|---|
| — | clean | 4.93% |
| awgn | 10 dB | 9.86% |
| awgn | 5 dB | 11.97% |
| awgn | 0 dB | 17.25% |
| — | clean | 4.93% |
| industrial | 10 dB | 8.45% |
| industrial | 5 dB | 11.62% |
| industrial | 0 dB | 17.25% |

ESC-50 is CC-BY-NC (internal eval only). Industrial categories: chainsaw, engine, hand_saw, helicopter, siren, train, vacuum_cleaner, washing_machine.

## Analysis

- **The ≤15%-at-5 dB claim holds on real noise:** PhoWhisper hits **11.62%**
  at 5 dB against ESC-50 machinery sounds (and 11.97% against AWGN) — under
  the 15% bar, and this is *without any denoising*.
- **AWGN is a fair/harder proxy, not a soft one.** Real industrial noise is
  slightly *easier* at 10 dB (8.45% vs 9.86%) and equal at 5/0 dB. So the
  earlier AWGN-only report wasn't cheating; it was conservative. (Still worth
  re-running on recorded factory audio, which has different spectral tilt and
  non-stationarity than ESC-50.)
- **Clean WER moved 2.19% → 4.93%** just by going 20 → 30 VIVOS utterances —
  a reminder that these are small-sample numbers; report error bars or use
  ≥200 utts before quoting.
- **Not done here:** the denoise ablation (raw vs RNNoise vs RNNoise+VAD).
  RNNoise/DeepFilterNet Python bindings are broken in this env (see
  `experiments/01_denoise_vad.ipynb`); that arm belongs in the Rust pipeline
  where `nnnoiseless` actually runs.

## Method

VIVOS slice (`quocanh34/viet_vivos`), 30 utterances; ESC-50 industrial clips
mixed at target SNR (re-sampled to 16 kHz, tiled to the utterance length);
same greedy decode + `jiwer` WER as `asr_ab.md`; `scripts/evaluate_asr_noise.py`.
