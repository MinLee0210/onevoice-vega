# Pipeline improvement research (2026-10-12)

Online pass (arXiv + Hugging Face + model cards) for: (a) what to **add** to the
Vega Hand / Vega Air solutions, and (b) **models/techniques** to improve each
pipeline stage. Complements `pipeline-improvement-research-2026-07.md` and
`pipeline-performance-research-2026-08.md`. Everything below is a *candidate*;
nothing is validated on our data yet. Licenses checked on the card where noted —
verify before commercialization.

## A. Additions to the two solutions

1. **Quality-Estimation gate + one-shot self-correction (Hand).** A fully
   offline edge S2T pipeline (Jetson Nano, Whisper-tiny → Opus-MT) gates on a
   QE score and runs a **second pass only when confidence is low**; Minimum
   Bayes-Risk reranking (N=3) gave significant BLEU/ChrF/COMET gains, and the
   QE model is a *gate, not a ranker* — removing it from selection frees
   680 MB. Directly maps onto our "cheap unbiased pass, only re-decode if
   needed" glossary design and our safety-critical framing.
   arXiv 2610.07545.
2. **ASR confidence / no-speech gate** (already scoped): feed mean token
   log-prob + Whisper no-speech prob into the LED/haptic "didn't catch that"
   state instead of translating a hallucination.
3. **Real 2-mic beamforming front-end** (differentiator #3, currently
   unimplemented). On-device references: a **6-mic neural MVDR** beamformer at
   **15 ms/frame, 45.9 mW**, STOI 97.65% on an MCU with a three-stage
   mixed-precision pipeline (float32 MVDR + int8 CNN + float16 transformer,
   arXiv 2609.33755); a **dual-mic** dereverb + TCN beamformer + MetricGAN
   denoiser for commodity phones (RDSE, arXiv 2610.12476).
4. **Air: multi-channel / spatial cues** measurably help noisy outdoor audio and
   device-directed vs background speech (WearVox, arXiv 2601.02391) — supports
   keeping the near-rotor reference mic for the ego-noise stretch.
5. **Phrasebook fast-path + pre-rendered broadcast audio** (from the repo's own
   earlier review): instant, 100% term-accurate for the ~20–50 safety phrases;
   natural fit for Air (broadcast is mostly fixed phrases).

## B. Stage-by-stage candidates

| Stage | Candidate | Why | Size / license |
|---|---|---|---|
| **ASR (all langs)** | **Qwen3-ASR-0.6B / 1.7B** | one model, 30+ langs **incl. vi**, strong code-switching (VI–EN) — could replace PhoWhisper-base + whisper-small with a single model | 0.6B/1.7B, **Apache-2.0**; too big for QCS6490 baseline today, watch for distills |
| **ASR (VI)** | **VietASR** (13M CTC), **ViP-VL / ChunkFormer** (17k-h VI SSL, SOTA), **PhoASR** (Qualcomm 500-h VI dataset) | CTC/transducer removes the AR decode loop (our latency bottleneck); ViP-VL is new VI SOTA | 13M … ~100M; check licenses |
| **ASR (EN)** | **NVIDIA Nemotron Speech Streaming** | int4 k-quant + op-fusion: 2.47 GB → **0.67 GB**, WER within 1% abs, 0.56 s latency, CPU streaming (arXiv 2604.14493) | English only; NVIDIA license |
| **Quantization** | int4 k-quant + operator fusion; **layer-wise PTQ (FADE)**; QAT | Whisper int4/int8 cuts latency ~19% / size ~45% with retained accuracy (arXiv 2503.09905); FADE improves 3–4-bit WER (arXiv 2601.02455) | — |
| **ASR robustness** | **PiDA** phonetic data-augmentation for VI + LoRA | fine-tune on ASR-like phonetic corruptions: +2.04 BLEU downstream (INTERSPEECH 2026, arXiv 2606.12911) | training-time |
| **MT terminology** | **Hard-example selection** (keep only glossary-contradicting pairs) | term accuracy 78.7% → **89.9%** at fixed data (SalamandraTA, WMT26, arXiv 2609.09999); glossary injection removes genuine errors | training-time, matches our LoRA plan |
| **MT compression** | distillation + int8 | **69% smaller, ~98% faster** with no quality loss (biomedical NMT, arXiv 2610.07032) | — |
| **MT model** | Hunyuan-MT-1.8B/7B (vi supported) as **distillation teacher**, not edge baseline | 7B too big for 12 TOPS; teacher for the Opus-MT/LoRA student | check Tencent license |
| **TTS (license-clean VI)** | **parler-tts-vietnamese** (Apache-2.0), **spark-tts-vietnamese** (Apache-2.0), VITS-OpenBible (CC-BY-SA) | replace the CC-BY-NC MMS-KO trap and espeak-GPL path; alternatives to current Piper VI | verify per card; **avoid** F5-TTS-Vietnamese (CC-BY-NC-SA) |
| **TTS (arch)** | **Kokoro-82M** | Apache-2.0, tiny, very fast — but **en (+zh)** only, no VI as of this pass | 82M |
| **TTS quantization** | per-channel 4-bit + staged ablation | same bit-width behaves very differently per architecture (Kokoro robust, Supertonic −2.8 UTMOS); **must validate on target** (arXiv 2609.28974) | — |
| **Vocoder** | **PhaseGAN** | ~500K params, ~1 GMAC, edge-friendly high-fidelity (arXiv 2609.12918) | — |
| **Denoise** | **DeepFilterNet3**, **GTCRN** (tiny), ZipEnhancer, DuSpaR sparse-recurrent | better than RNNoise on machinery noise; ~50% less compute than GRU at equal quality (arXiv 2609.39237) | DFN3 ONNX MIT; check others |
| **VAD** | **TEN-VAD** (ONNX, real-time), Silero v5, FireRedVAD | TEN-VAD is a lightweight Silero alternative with ONNX packaging | verify license (Apache-2.0 variants exist) |
| **QE / reranking** | mBERT QE gate + MBR (N=3) | see A.1; MBR beat QE-reranking and constrained beam search | ~small |

## C. Caution / anti-recommendations

- **Denoising can *raise* ASR WER** (artifacts) — keep evaluating a no-denoise
  arm; a stuttering-detection study found a denoise+normalise+Opus+VAD chain
  dropped block-F1 0.638 → 0.465 (arXiv 2609.32285). Don't assume the current
  denoise→VAD→ASR order is optimal.
- **End-to-end S2ST** (Hibiki, SeamlessM4T) is tempting but degrades on long
  technical sentences (our content) — the repo's cascaded choice stands.
- **License traps:** MMS-TTS-KO (CC-BY-NC), F5-TTS-VI (CC-BY-NC-SA),
  MADLAD/NLLB (CC-BY-NC) — all non-commercial. Prefer Apache-2.0/MIT/CC-BY.
- **Big MT teachers** (Hunyuan-MT-7B, Seed-X, TranslateGemma) won't fit the
  QCS6490 latency/memory budget — use only as distillation teachers.

## D. Highest-leverage, lowest-risk next steps

1. **ASR** — prototype VietASR (CTC, kills the AR loop) and/or Qwen3-ASR-0.6B
   (one model, 4 languages) on VI; compare WER + latency vs PhoWhisper.
2. **MT** — hard-example glossary LoRA (proven +11% term accuracy) + int8
   distillation; test viability of split decoder on NPU if MT ever moves there.
3. **TTS** — swap KO off CC-BY-NC and VI off GPL espeak; evaluate
   parler/spark-tts-vietnamese + Kokoro (en/zh); validate 4-bit per architecture.
4. **Front-end** — implement the 2-mic beamformer; re-run noise eval with
   denoise / no-denoise / +VAD arms on real machinery noise.
5. **Hand** — wire the QE-gate + MBR self-correction (safety + BLEU/COMET win).

## Sources

- QE + self-correcting edge ST: https://arxiv.org/abs/2610.07545
- On-device streaming ASR (Nemotron, int4): https://arxiv.org/abs/2604.14493
- Whisper quantization study: https://arxiv.org/abs/2503.09905
- Layer-wise PTQ (FADE): https://arxiv.org/abs/2601.02455
- Whisper compression fairness (int4 loops): https://arxiv.org/abs/2609.28739
- Vietnamese ASR phonemic modeling: https://arxiv.org/abs/2605.27874
- ViP-VL (VI SSL, ChunkFormer): https://arxiv.org/abs/2606.10360
- PiDA (VI ST data augmentation): https://arxiv.org/abs/2606.12911
- PhoASR (Qualcomm VI dataset): https://arxiv.org/abs/2603.14779
- Terminology hard-example selection (WMT26): https://arxiv.org/abs/2609.09999
- NMT distillation+quantization: https://arxiv.org/abs/2610.07032
- TTS PTQ across architectures: https://arxiv.org/abs/2609.28974
- TTS speculative decoding (RVQ): https://arxiv.org/abs/2609.37007
- PhaseGAN vocoder: https://arxiv.org/abs/2609.12918
- Neural MVDR beamformer on MCU: https://arxiv.org/abs/2609.33755
- Dual-mic RDSE: https://arxiv.org/abs/2610.12476
- DuSpaR sparse recurrent: https://arxiv.org/abs/2609.39237
- WearVox wearables benchmark: https://arxiv.org/abs/2601.02391
- Models: Qwen/Qwen3-ASR-0.6B (Apache-2.0), TEN-framework/ten-vad,
  hexgrad/Kokoro-82M (Apache-2.0), tencent/Hunyuan-MT-7B,
  parler-tts-vietnamese (Apache-2.0), spark-tts-vietnamese (Apache-2.0)
