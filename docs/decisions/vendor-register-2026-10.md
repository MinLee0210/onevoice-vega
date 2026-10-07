# Vendor register (2026-10-07)

Every vendor met during the competitor, product and drone-payload research, one row each,
so nothing has to be searched for twice. "Checked" means read on the vendor's own page or a
direct scrape this session; "snippet" means a search-result excerpt only; "not researched"
means named but no data gathered. Prices are list or retail figures from the pages cited and
will drift. Earlier detail: `competitor-landscape-2026-07.md` and `vega-air-and-feature-research-2026-08.md`.

## 1. Industrial push-to-talk / smart radio (closest to Vega Hand)

| Vendor / product | What it has | Offline? | Evidence |
|---|---|---|---|
| **Weavix Walt** | 40+ languages and dialects incl. Vietnamese and Korean; voice+text translation; auto-transcription (text or audio); replay; dictation; channels; language preference held in the user account, not the radio | No (needs LTE/WiFi) | Checked: weavix.com/features/ai-language-translation |
| **Relay (RelayX)**, `relaypro.com` | **TeamTranslate**, 30+ languages, "native language" per device; 2900 mAh, up to 24 h, charges in under 3 h; IP66/IP68, MIL-STD-810H, Class 1 Div 2; 3 mics, "110 dB noise cancellation", dual 2 W speakers up to 110 dB; 1.9" OLED; **panic alert with live location, mass emergency alerts translated per language, message audit trail**; quad-SIM + WiFi + LoRa; **Fleet Shield** (upgrade guarantee, next-day replacement) | No (cellular/WiFi) | Checked: relaypro.com/lp/relay-vs-motorola-industrial-radio (vendor comparison, so Motorola claims are the vendor's) |
| **Motorola Solutions** WAVE PTX, TLK 25, XPR series | Broadband PTT over cellular; wearable TLK 25 with emergency features and voice assistant. Relay's page says the XPR line has no translation (vendor claim) | n/a | Snippet |
| **Hytera** PoC radios (HyTalk) | PoC radios: group calls, dispatch, patrol tasking, data | n/a | Snippet |
| **Sonim** | Rugged PTT phones for transport, fleet | n/a | Snippet |
| **Zebra** (TC series) | Rugged handhelds; translation features not examined | — | Not researched |

## 2. Handheld / wearable translators (consumer and travel)

| Vendor / product | What it has | Offline? | Evidence |
|---|---|---|---|
| **iFLYTEK** Smart Translator | 60 languages; 18 offline packs. **The offline list includes Vietnamese–Chinese (Mandarin)**, so it already covers one of our three pairs offline | Partly | Checked: iflytek.com FAQ "Languages supported by voice translation" |
| **Timekettle** T1 handheld (about USD 300), WT2 Edge, W3 | T1: 43 languages, offline packs, touchscreen; earbuds: 8 offline languages, no Vietnamese, 2-3 h battery (earlier research) | Yes (T1 packs) | Snippet (T1); earlier notes (earbuds) |
| **Vasco Translator** V4 / Q1 / E1 | 5" touchscreen, noise-reduction mic, 99 dB speaker; cloud via embedded SIM | No | Snippet; earlier notes |
| **Pocketalk** (Classic/S/S2) | 92+ languages; needs WiFi/eSIM; 30 s utterance cap | No | Earlier notes |
| **Travis** (Touch Go, One; Dutch brand, sold in Vietnam) | Handheld translators; Vietnamese retail about 6M VND on promotion for Touch Go | Some | Snippet (retailer listing) |
| **ATalk** (Vietnamese brand; Plus+, Plus, One, Go) | Plus+ claims 137 languages; Vietnamese→16 languages two-way; offline voice translation (Plus: 13 offline languages; One: 19 offline incl. Vietnamese; Go: 14); group translation; vSIM on Go. Retail around 2-3.5M VND; a reseller claims about 19,800 units sold (unverified) | Yes (limited packs) | Checked: baolaocai.vn article (reseller-written) |
| **ili (Logbar)**, **Waverly Labs Pilot**, **Muama Enence**, **Langogo** | ili: one-way canned phrases; Pilot: service shut down April 2024 and bricked units; Muama: needs app/WiFi; Langogo not researched | mixed | Earlier notes; Langogo not researched |
| Meta Ray-Ban glasses | Live translation on glasses | No | Snippet (KUDO) |

## 3. Wearables / AR for industry

| Vendor / product | What it has | Evidence |
|---|---|---|
| **RealWear** Navigator Z1 / 520 | Intrinsically safe hands-free headset; voice recognition up to 100 dBA; multiple noise-cancelling mics; inspection logging. Translation not examined | Snippet |
| **OLOGN** "NAWI" (QVIC 2026 top 15) | Universal AR for industrial and emergency safety (Vietnam) | Checked: QVIC press release |

## 4. Drone loudspeaker payloads (Vega Air)

| Vendor / product | Specs | Evidence |
|---|---|---|
| **DJI Zenmuse V1** | 690 g, 134×119×140 mm, 30 W rated, 25 W speaker, **129 dB @ 1 m, 700 m on Matrice 400**, 800 Hz-8 kHz, text-to-speech and storage, directional audio recording; fits Matrice 400/350 RTK/300 RTK | Checked: enterprise.dji.com/mobile/zenmuse-v1/specs |
| **DJI Payload SDK speaker** | Voice files uploaded from Pilot for playback | Snippet |
| **CZI MP130 V2** | LRAD-class, up to 500 m over the PSDK link | Snippet |
| **CZI MP140** | 120 W, 140 dB, 225×272×221 mm, 2.5 kg, TTS | Snippet |
| **CZI LP12** | 270 g, up to 350 m, loudspeaker + spotlight, **multi-language TTS**, live voice and uploaded audio, for Matrice 30 | Snippet |
| Puyu, Ramoton (Shenzhen PSDK aftermarket) | Named in AGENTS.md from earlier research; not re-verified | Earlier notes |

Reading: these payloads do text-to-speech of typed or pre-recorded messages and live voice.
None of the pages shows speech translation. Our pod target (about 300 g) is lighter than
the Zenmuse V1 (690 g) and in the CZI LP12 class (270 g).

## 5. Vietnamese voice / AI vendors (suppliers or benchmarks, not device competitors)

| Vendor | What it has | Evidence |
|---|---|---|
| **Vbee** (AIVoice) | Cloud TTS: 50+ languages, 1000+ voices, voice cloning, dubbing. No edge-device or Qualcomm product found; the AGENTS.md precedent claim stays unverified | Snippet |
| **Viettel AI** | Vietnamese TTS and speech services; its guide lists Zalo AI | Snippet |
| **FPT.AI** | Vietnamese ASR and TTS platform, conversational AI | Snippet |
| **Zalo AI** | Four Vietnamese TTS voices (two Northern, two Southern) per the Viettel guide | Snippet |
| **VinAI** | PhoWhisper / PhoMT lineage (already in the stack) | Earlier notes |

## 6. Platform, hardware and ecosystem

| Party | Role | Evidence |
|---|---|---|
| **Qualcomm** (AI Hub, Dragonwing QCS6490) | Technology partner; pre-built QCS6490 Whisper QNN binaries | Checked |
| **Thundercomm** (Rubik Pi 3) | Board maker; limited global release; 12 TOPS | Snippet |
| **Edge Impulse**, **Canonical** | Rubik Pi 3 tooling (PPE detection demo) and optimised Ubuntu image | Snippet |
| **Saigon AI Hub, VNG, VNU-HCM** | Challenge organisers | Checked |
| **QVIC 2026 teams** (NAISCORP Robotics, TENSOR, VEEP, AIZ, OpenMind Robotics, UDATA, others) | Other Qualcomm program; pitch benchmarks, not rivals | Checked |

## 7. Features seen here that we do not have

1. Panic alert with live location, and translated mass emergency broadcast (Relay). We have no SOS yet.
2. Message history and audit trail (Relay), replay and transcription (Weavix).
3. Per-person language setting that follows the worker, not the device (Weavix account, Relay "native language").
4. A hardware service model: upgrade guarantee and next-day replacement (Relay Fleet Shield). A business-pillar answer for an offline device with no subscription.
5. Group translation (ATalk Plus+).
6. Battery: Relay quotes up to 24 h; our target is 8 h or more. Worth re-checking the power budget.
7. Hazardous-area certification (Class 1 Div 2) at Relay and Weavix; our target is IP54+, which is fine for the initial market.

## 8. Competitive notes that change the pitch

- **iFLYTEK lists a Vietnamese–Chinese offline pack** and ATalk markets offline Vietnamese, so
  "nobody has offline Vietnamese" is wrong. The defensible claim is industrial vocabulary
  plus noise-first audio plus a ruggedised PTT form, not offline Vietnamese alone.
  (AGENTS.md already says this for iFlytek; ATalk is new here.)
- **Local price anchor:** consumer Vietnamese-market translators sell at roughly 2-6M VND, so
  an industrial device must justify its price by noise robustness, ruggedness and glossary.
- **No found product** combines offline operation, a screenless rugged PTT body and a safety
  glossary. That remains an unrefuted gap, but only as far as this search reached.

## 9. Still to research

Zebra, Langogo, Motorola's own translation offering, Hytera/Sonim translation features,
RealWear translation, Vietnamese industrial-radio resellers, other drone-speaker makers
(Puyu, Ramoton re-check), and any Vietnamese hardware startups building on Qualcomm
(Vbee edge claim). Tavily research for this track had hit its usage limit; Firecrawl results only.
