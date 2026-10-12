# VI CTC ASR checkpoint sweep (VIVOS slice, CPU)

Following `asr_ab.md`: is there a CTC checkpoint that beats the Apache-2.0
`dragonSwing` baseline (clean 10.4% / 5 dB 23.0%)? 20 VIVOS utterances,
same harness (`scripts/evaluate_asr.py`). WER case/punctuation-normalized.

| model | arch | license | clean WER | 5 dB WER | latency | RTF |
|---|---|---|---|---|---|---|
| vinai/PhoWhisper-base | Whisper | BSD-3 | **2.19%** | **9.84%** | 527 ms | 0.18 |
| nguyenvulebinh/wav2vec2-base-vietnamese-250h | CTC | **CC-BY-NC-4.0** | 6.56% | **10.38%** | 54 ms | 0.02 |
| khanhld/wav2vec2-base-vietnamese-160h | CTC | CC-BY-NC-4.0 | 6.01% | 20.22% | 53 ms | 0.02 |
| dragonSwing/wav2vec2-base-vietnamese | CTC | Apache-2.0 | 10.38% | 22.95% | 79 ms | 0.03 |
| ginpham614/wav2vec2-base-vietnamese-…-v3.0 | CTC | Apache-2.0 | 100% | 100% | 54 ms | 0.02 |
| giangndm/parakeet-ctc-0.6b-vietnamese-bf16 | FastConformer-CTC | NVIDIA (other) | 2.22% | 10.00% | ~1065 ms | 0.33 |

## Analysis

- **The best CTC is CC-BY-NC.** `nguyenvulebinh/…-250h` is remarkable: at
  **5 dB it matches PhoWhisper (10.38%)** at **~10× the speed** (54 ms vs
  527 ms). That is exactly the industrial condition that matters. But it is
  **CC-BY-NC-4.0 → non-commercial**, so it cannot ship — the same trap as
  MMS-TTS-KO and NLLB.
- **The only Apache-2.0 VI CTC is materially worse** (`dragonSwing` 10.4% /
  23.0%). The other Apache-2.0 one tested is broken (100% WER).
- **Net:** CTC is a real latency lever, but as of this sweep there is **no
  commercially-licensed VI CTC checkpoint** that is good enough to replace
  PhoWhisper. Options: (a) fine-tune `dragonSwing` (Apache-2.0) on VIVOS/
  in-domain data to close the gap; (b) negotiate a CC-BY-NC exception /
  re-license; (c) keep Whisper as the shipped backend and use CTC only where
  a capable checkpoint is available.
- **Parakeet-CTC-0.6B matches Whisper accuracy** (clean 2.22% / 5 dB 10.00%)
  but is **2× slower than PhoWhisper on CPU** (1065 ms, RTF 0.33) — 0.6B is
  too big for the edge latency budget, and it's NVIDIA-licensed (not a clean
  commercial license). So accuracy parity exists in CTC-land, but not at a
  size/license that helps this device.

## Method

VIVOS test slice (`quocanh34/viet_vivos`), 20 utterances, AWGN, greedy;
standard HF ASR pipeline. Licenses from the model cards.

## Recommendation update

Do **not** switch the runtime default to CTC yet. Add a fine-tune of an
Apache-2.0 VI CTC (dragonSwing) on VIVOS + factory-noise-augmented audio as
Workstream 2 follow-up; only if it reaches ~≤7% clean does CTC become the
default and unlock the 57 ms ASR path on the QCS6490 CPU.
