# Deep research: challenge rules, edge-AI techniques, products (2026-10-07)

Three tracks, run with Firecrawl (web search, page scrape, paper search) and Tavily.
Each claim is tagged **checked** (read on the source page or paper abstract this session)
or **unverified** (a search snippet, or my recollection). Coverage is uneven: the
challenge and paper tracks are solid, the product track is thin (see section 4).

## 1. The challenge: real rules, and who else is in it

### 1.1 Scoring (checked: Luma event page, `luma.com/g1rwi7ag`)

| Pillar | Weight | What is measured |
|---|---|---|
| Technical Performance | **50 %** | BLEU and **COMET** on a proprietary benchmark; **MOS** (native-speaker panel) for TTS prosody and clarity; RTF < 1.0; turnaround from "End of Speech" to "Start of Audio Synthesis" |
| Innovation & Novelty | 25 % | Unique approach, added-value features |
| Business Viability | 25 % | Market readiness, hardware sustainability |

Hard rules: latency over **2.0 s gets significant deductions**; **any internet
dependency during testing = disqualification**. Language pairs: vi↔en, vi↔zh, vi↔ko.
Any portable, handheld, wearable or embedded form factor. Team size unlimited.

AGENTS.md currently lists the Phase 2 rubric (35/25/15/15/10). From the prototype
phase the three-pillar rubric above is the one on the public page.

### 1.2 Timeline (checked: Luma page and 24h.com.vn / Arttimes article)

Registration 24 May–24 Jun 2026 → spec submission Jun–Jul → **prototype submission
Sep 2026** → **field testing Oct 2026** → grand finale Nov 2026, VNG Campus, HCMC.
Total prize value 1.3 billion VND. Organisers: Saigon AI Hub with VNG and VNU-HCM;
Qualcomm is the technology partner. Teams may run on their own hardware or Qualcomm's,
with compatibility testing on Qualcomm AI Hub.

Note: `saigonaihub.com/OneVoiceAIChallenge` returned 404 when scraped (it may be
case- or route-sensitive); the Luma page links to it as the canonical source, so
re-check it directly for updates.

### 1.3 Other participants

- **No team list or shortlist is public** (not found in English or Vietnamese searches).
  The Luma page shows about 230 attendees/registrants (checked). I cannot name a
  competing team and will not guess.
- **Closest public benchmark is a different Qualcomm program**, QVIC 2026 (checked:
  press release, 3 Sep 2026). Winners: NAISCORP Robotics (champion, USD 100k), TENSOR,
  VEEP, with AIZ and OpenMind Robotics as Innovation Award winners. Two of its top 15 touch our space:
  **OLOGN "NAWI"** (AR for industrial and emergency safety) and **UDATA** (AI technical
  support in the factory). Neither is speech translation. Useful as a pitch yardstick:
  winners were robotics/AIoT products with a clear market, not model demos.
- Competing teams in this contest will most likely build the same cascade
  (Whisper-class ASR → small MT → VITS TTS). Where we can differ is in what the
  scorers measure: COMET quality on domain speech, TTS MOS, and the glossary.

### 1.4 What this changes for the repo

1. **COMET is unmeasured today.** Add a COMET + chrF + BLEU harness on a held-out set.
2. **TTS naturalness is scored (MOS).** Korean MMS-TTS is both non-commercial
   (CC-BY-NC 4.0, see `THIRD_PARTY_LICENSES.md`) and the voice most likely to sound worst.
   Replacing it earns points and removes a licence blocker.
3. **Our 1.2 s target is inside the 2.0 s line, and latency is measured to the
   *start* of synthesis.** Further latency work has low marginal value compared with quality.
4. **Scope is exactly six directions** (vi↔en/zh/ko). Do not spend effort elsewhere.
5. **The brief is "natural bilingual conversations".** Vega Hand is the scored demo;
   Vega Air is a broadcast extension and fits the Innovation pillar, not the core test.
6. **Offline is a disqualifier, not a feature.** Check for hidden network use: the
   `ort` build, model downloads at first run, and espeak data paths must all work in airplane mode.

## 2. Articles and techniques for edge AI

### 2.1 Front-end: denoising can hurt ASR

- *When De-noising Hurts* (arXiv 2512.17562, **checked abstract**): on 500 recordings
  across nine noise conditions, enhancement preprocessing worsened semantic WER for
  Whisper, Parakeet and others in all 40 configurations (1.1 to 46.6 points absolute).
  Their setting was medical speech with a MetricGAN enhancer, not RNNoise on factory noise, so
  it is a warning, not proof about our stage.
- Latent-level enhancement (FM-Refiner, arXiv 2601.04459, checked) refines the ASR
  encoder output instead of the waveform; it needs a CTC-style encoder.
- For rotating-machinery noise, a cyclostationary beamformer ahead of a small DNN beat a
  larger DNN alone (arXiv 2602.12986, checked). Relevant to our harmonic factory noise and
  to the Vega Air rotor noise.
- **Action:** the noise eval must include a *no-denoise* arm. If RNNoise hurts,
  options are bypassing it, noise-augmented fine-tuning of PhoWhisper, or a
  harmonic-aware pre-filter.

### 2.2 ASR

- **Quantization.** Whisper-small with dynamic INT8: −57 % size with no WER loss; static
  quantization was worse (arXiv 2511.08093, checked). Edge-ASR benchmarks eight PTQ methods on
  Whisper and Moonshine (arXiv 2507.07877, checked). An evolution-strategy calibration
  makes full INT8 near-lossless for speech models (arXiv 2603.08173, checked).
  Takeaway for the QCS6490: full INT8 is the only precision that compiles there, so
  calibration data matters; calibrate on noisy Vietnamese, not clean read speech.
- **Padding and the 30 s window.** Padding suppresses hallucination but encoder cost
  grows sharply toward the full window (arXiv 2506.15912, checked via snippet;
  NPUsper, arXiv 2607.01108, **unverified** beyond its title). The ADR's "trim the tail"
  idea needs a WER test before adoption.
- **Moonshine v2** (arXiv 2602.12241, checked): sliding-window streaming encoder,
  accuracy on par with models 6× larger. Language coverage for Vietnamese is **unverified**.
- **Qwen3-ASR** (0.6B / 1.7B, 52 languages): appears on the Vietnamese Open ASR
  Leaderboard (checked). 0.6B is large for the NPU budget; keep as a quality reference.
- **Streaming ASR + MT cascades** on device (arXiv 2508.13358, checked) and speech-LLM
  streaming translation (arXiv 2605.14766, checked) show the research direction;
  half-duplex PTT does not need them.

### 2.3 MT and terminology

- **Quality-estimation gate.** An offline Jetson Nano pipeline (Whisper-tiny → Opus-MT)
  runs a second pass only when a QE score falls below a threshold; QE worked as a gate
  but poorly as a ranker, MBR decoding helped, constrained beam search hurt
  (arXiv 2610.07545, checked). This matches our "cheap first pass, correct only when a
  term is missing" design and could also trigger a low-confidence haptic.
- **Terminology training.** Keeping only the training examples where the model's own
  translation contradicts the glossary raised term accuracy from 78.7 % to 89.9 % at fixed
  data (arXiv 2609.09999, WMT26, checked). An LLM post-editor beat strict constraint
  enforcement on fluency (arXiv 2511.07461, checked). Both support moving from logit-bias
  to LoRA with a hard-example set.
- **HY-MT1.5-1.8B / Hy-MT2** (Tencent): lists Vietnamese, Chinese and Korean, described as
  edge-deployable (GitHub README and search snippets, checked; speed on QCS6490 and licence terms
  **unverified**). A 1.8B autoregressive model is unlikely to meet the latency budget on
  the Hexagon. Use it as a **teacher**: generate synthetic industrial parallel data and
  distil or LoRA-tune Opus-MT. Check the licence for territory or use restrictions first.
- **TranslateGemma** (Google, 55 languages, built on Gemma 3): another teacher candidate;
  licence and size per variant **unverified**.
- **Evaluation:** WMT-style COMET is what the organisers use. Run it offline in Python
  (model licence **unverified**).

### 2.4 TTS (MOS is scored)

- **Kokoro-82M**: Apache-2.0, ONNX, tops the TTS Arena leaderboard (v1) per a vendor
  comparison. One source lists Korean and Mandarin support (**unverified**; no Vietnamese).
- **Supertonic 2 (MIT)** and **Pocket TTS (Apache-2.0)** are listed as permissive options
  (same vendor comparison, **unverified** for Vietnamese/Korean).
- **Piper's current repository is GPL-3.0**; the archived original was MIT (checked via two
  comparison articles). This adds to the existing espeak-ng GPL note.
- **Vietnamese:** VietTTS (`dangvansam/viet-tts`, voice cloning) and VieNeu-TTS (CPU,
  fine-tuned from neutts-air on 140 h of Vietnamese) are both candidates already in TODO.md; licences **unverified**.
- **Action:** run a small native-speaker MOS test on all four current voices before
  choosing replacements; Korean first.

### 2.5 System designs worth borrowing

- X-Translator (arXiv 2607.17544, checked): modular cascade with session-level
  controller, incremental segment commitment and speaker-prompted TTS. Not needed for PTT,
  but its speaker-prompt idea connects to the voice-cloning item.
- Tutorial-grade benchmark: a streaming cascade reached 755 ms time-to-first-audio with
  cloud STT/TTS (arXiv 2603.05413, checked). Our measured on-device figure (about 740 ms
  through TTS on CPU) is in the same range without any network.

## 3. Ranked recommendations (supersedes the priority order in the 2026-10 review note)

| # | Action | Why now | Scored by |
|---|---|---|---|
| 1 | COMET/chrF/BLEU harness + held-out vi↔en/zh/ko set | Organisers score COMET; we have no number | Technical |
| 2 | Replace Korean TTS; MOS-check all 4 voices | Scored MOS plus the CC-BY-NC blocker | Technical, Business |
| 3 | Noise eval with real machinery noise and a no-denoise arm | Literature says denoising can hurt; claim is unproven | Technical |
| 4 | EN-pivot MT for the other 5 directions | Device cannot run 5 of 6 scored directions | Technical |
| 5 | Safety-phrasebook fast path | Zero-latency, exact output on safety terms | Innovation |
| 6 | Distil a teacher MT (HY-MT / TranslateGemma) into Opus-MT or LoRA with a hard-example glossary set | Quality plan that fits the latency budget | Technical, Innovation |
| 7 | Airplane-mode audit; QNN smoke test; live PTT loop | Disqualification risk; field test is this month | All |
| 8 | Cheap product features (section 4) | Differentiation without new models | Innovation |

## 4. Products (thin coverage)

The Tavily research call for this track failed on the plan's usage limit, and two other
long Tavily runs were still pending when I wrote this. This section rests on direct scrapes
and search results only. The prior landscape work in
`docs/decisions/competitor-landscape-2026-07.md` and `vega-air-and-feature-research-2026-08.md` is still the base.

| Product | What it has that we lack (checked unless noted) | Connectivity |
|---|---|---|
| **Weavix Walt** smart radio | 40+ languages and dialects incl. Vietnamese and Korean; real-time voice and text translation; **automatic transcription of every message with a text-or-audio choice; replay/playback; dictation and text-to-speech; channels**; language preference held in the **user account, not the radio**, so shared devices work across shifts | Needs connectivity (degrades to plain radio, per earlier research) |
| **Timekettle T1** handheld | 43 languages, offline packs, touchscreen, about USD 300 (search snippets, unverified) | Offline for packs |
| **Vasco V4** | 5-inch touchscreen, noise-reduction mic, 99 dB speaker (vendor blog, unverified) | Mostly online |

Cheap features suggested by this table, none needing new models:
1. **Translation log with replay and USB export** (Weavix playback). Already in TODO.md; raise its priority.
2. **Per-worker language profile** (NFC tag or a short code on the PTT button) so one device serves a shift (Weavix account-held language).
3. **Group broadcast on Hand**, reusing Vega Air's one-to-many path (Weavix channels).
4. **Optional text transcript** to a phone or small screen over BLE for noisy spots where audio is not heard (Weavix transcription). Keep the device itself screenless.

A follow-up pass on vendors is in `docs/decisions/vendor-register-2026-10.md` (every
vendor, with specs and evidence level). Notable additions: Relay's translated panic and
mass-emergency alerts, iFLYTEK's offline Vietnamese–Chinese pack, the Vietnamese brand
ATalk with offline Vietnamese, and DJI/CZI drone speakers. The earlier Vbee/Qualcomm claim
in AGENTS.md remains unverified.

Status update (from you, 2026-10-07): the prototype submission in September was
documents only and was accepted. Field testing in October therefore needs the first
working demo, so items 7 (airplane-mode audit, live PTT loop, QNN smoke test) move up the
list ahead of the quality items.

## 5. Not found or unverified

- Any competing OneVoice team or shortlist.
- Kokoro/Supertonic Korean support and voice quality; Vietnamese TTS licences (VietTTS, VieNeu).
- HY-MT and TranslateGemma licences, size per variant and speed on QCS6490.
- COMET model licence; Moonshine v2 Vietnamese coverage; NPUsper's actual results.
- The hidden benchmark's domain (general conversation vs industrial). It decides how much the glossary is worth.
- Whether our prototype was already submitted in September.

## Sources

- OneVoice AI Challenge 2026 (Luma): https://luma.com/g1rwi7ag
- 24h.com.vn / Arttimes launch article: https://www.24h.com.vn/cong-nghe-thong-tin/treo-thuong-13-ty-dong-cho-nha-lap-trinh-giai-phap-dich-thuat-ai-thoi-gian-thuc-khong-can-internet-c55a1764627.html
- VNG press release: https://vng.com.vn/news/press-release/saigon-ai-hub-onevoice-ai-challenge-edge-ai-dich-thoi-gian-thuc-khong-can-internet.html
- QVIC 2026 winners: https://www.kitsapsun.com/press-release/story/793145/qualcomm-announces-winners-for-qualcomm-vietnam-innovation-challenge-2026/
- Weavix translation page: https://weavix.com/features/ai-language-translation/
- Hy-MT: https://github.com/Tencent-Hunyuan/Hy-MT ; TranslateGemma: https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/
- TTS comparisons: https://picovoice.ai/blog/on-device-tts/ ; https://www.promptquorum.com/power-local-llm/piper-vs-kokoro-tts
- VietTTS: https://huggingface.co/dangvansam/viet-tts
- Qualcomm Whisper QCS6490 builds: https://huggingface.co/qualcomm-ai-hub-community/whisper-small-multilingual-qcs6490-qnn-int8
- Papers (arXiv IDs): 2512.17562, 2601.04459, 2602.12986, 2511.08093, 2507.07877, 2603.08173, 2506.15912, 2607.01108, 2602.12241, 2508.13358, 2605.14766, 2610.07545, 2609.09999, 2511.07461, 2605.22064, 2607.17544, 2603.05413
