# Vega Air competitor landscape + copyable feature research (2026-08)

Follow-up research pass after the Phase 2 technical submission (the
`QUAVIO/Vega` two-carrier proposal: **Vega Hand** = PTT handheld, one-to-one;
**Vega Air** = drone-mounted multilingual broadcast, one-to-many). The earlier
`competitor-landscape-2026-07.md` covered the handheld translator market; this
doc covers (1) the drone-PA/broadcast market Vega Air enters, (2) the
ego-noise/rotor-noise cancellation technique Vega Air's two-way stretch needs,
and (3) a distilled, actionable feature list for both carriers.

Bottom line: **no drone-mounted loudspeaker does speech translation or
simultaneous multilingual broadcast, and no speech-translation device does
drone broadcast — the Vega Air position is uncontested.** The differentiation
is entirely the on-device ASR→MT→TTS engine, not the acoustic hardware (which
is commodity).

---

## 1. Drone-mounted PA / loudspeaker broadcast (Vega Air)

### Market structure

Three layers, none of which owns the category:

- **One first-party DJI accessory** — the *DJI Mavic 3 Enterprise Speaker*
  (85 g, USB-C top-mount, 3 W, **110 dB @1 m, 100 m @70 dB**, US$215.99).
  No DJI loudspeaker exists for the Matrice 30/350/400 class.
- **A Shenzhen aftermarket of DJI-SDK payloads** (Puyu, Drone-Payload/Ramoton,
  CZI, UAVForDrone, …) targeting M30/M350/M400 via **SkyPort V2/V3** or
  **PSDK E-Port**, controlled from DJI Pilot 2 / WeChat / API / a 4G handheld
  mic.
- **2-in-1 speaker + spotlight/strobe combos** (CZI LP12, etc.).

### Spec tiers (vendor-published; Chinese aftermarket range claims are marketing)

| Tier | SPL @1 m | Weight | Power | Example |
|---|---|---|---|---|
| Ultralight | 110 dB | 85 g | 3 W | DJI M3E Speaker |
| Compact | ~120 dB | 265–288 g | 25 W | CZI LP12, RT-H10 |
| Pro quad-driver | 125–130 dB | 390 g | 50 W | Puyu H150A (claims 500 m–1 km) |

### Key finding: nobody translates

"Multilingual" everywhere means **TTS playback of pre-typed text** (Puyu's
"Text Shout"), never translation. Targeted search for "drone loudspeaker
real-time translation multilingual broadcast" returned only cloud event-SaaS
(EventCAT, XSpeak), iFlytek's cloud translation, and **Axon Real-Time
Translation** (a body-worn-camera feature, 50+ langs, cloud) — nothing
airborne, nothing offline.

### What this means for Vega Air

- **The 20–30 m intelligibility requirement is trivial** — even DJI's weakest
  speaker over-delivers (>3×). The acoustic hardware is not the moat.
- **Specs to match (parity):** ≥120 dB @1 m (compact dual-driver), ≥200 m
  effective range, ≤300 g to stay M30-class, PSDK/SkyPort integration,
  live-mic + stored-file + loop playback.
- **Specs to beat (the differentiation surface, all unclaimed):**
  1. on-device ASR→MT→TTS (no cloud),
  2. simultaneous 3-language broadcast (one instruction → VI/EN/ZH/KO),
  3. offline / link-isolated operation,
  4. industrial glossary-constrained MT.
- **Unverified:** aftermarket MSRPs are quote-only; no independent acoustic
  test of any vendor's dB-at-range under factory noise; whether M30 has a
  genuine integrated speaker (DJI's own spec page lists none).

---

## 2. Ego-noise / rotor-noise cancellation (Vega Air two-way stretch)

### The proposed technique is not published

Vega Air's plan — deterministic **RPM-harmonic subtraction from flight-controller
telemetry, then RNNoise residual cleanup** — appears novel: no paper found uses
FC motor telemetry to synthesize the rotor harmonic comb before neural denoising.
It is a synthesis of three individually-proven ideas (harmonic/cyclostationary
noise modeling, reference-based cancellation, neural residual denoising), and is
directionally aligned with the 2025–2026 literature, but "viable" is unproven.

### Five established technique families (proven in literature)

1. **Multichannel blind source separation** on a drone mic array (Wang &
   Cavallaro, IEEE/ACM TASLP 2020 — peer-reviewed).
2. **Multichannel Wiener filter with dedicated ego-noise reference mics placed
   near the propellers** (Tengan et al. 2022, `arXiv:2211.02690`) — improves
   STOI/PESQ over a plain MWF.
3. **Beamforming + deep autoencoder** (UGent).
4. **Neural monaural speech enhancement** (DroFiT `arXiv:2509.16945`,
   FRCRN harmonic adapter `arXiv:2405.10022`, DRONEAUDIONET `arXiv:2608.00875`).
5. **Cyclostationary / cyclic processing** that exploits rotor harmonic
   structure — cyclic-MPDR (`arXiv:2510.18391`) gains **up to +5 dB SI-SDR
   with a single mic** but *estimates* the fundamental from audio, not telemetry.

### The important caution

"Normalise or condition?" (`arXiv:2609.31699`) found **naive deterministic
subtraction worsens streaming false-alarm rate ~10×** — subtractive noise-floor
front-ends fire far more often than plain log-mel at the same threshold. Passing
the RPM-derived harmonic estimate as a **conditioning/side input** to the neural
denoiser is safer than hard subtraction. The broadband turbulent rotor-noise
component is invisible to telemetry and must be handled by the neural stage
(or shielding/placement).

### Physics is on our side for the tonal part

Blade-passing frequency = `N_blades × RPM/60` is a standard aeroacoustics
identity and exactly what FC telemetry provides. But each rotor runs at a
slightly different RPM (almost-cyclostationary, not a stable comb), and the
broadband component cannot be synthesized from telemetry — so telemetry is a
*fundamental-frequency hint*, not a full cancellation source.

### Reusable assets

- **DroneAudioSet** (23.5 h, MIT, real flight, mic-placement/throttle
  variants) on HuggingFace — the natural training/eval backbone.
- **cyclic-MPDR** and **EchoHawk** release code (EchoHawk also documents a
  session-level data-leakage pitfall when splitting drone audio).
- Tengan et al. released semi-anechoic UAV speech+ego-noise recordings
  (3 fixed rotor speeds).

### Recommendation for Vega Air

Treat the two-way capture as a **Phase 4 stretch**, not Phase 3 baseline. When
it starts: (a) place a physical reference mic near the rotor (proven) to
cross-check the telemetry synthesis, (b) prefer *conditioning* over *subtraction*,
(c) train/eval on DroneAudioSet. Keep the RPM-harmonic idea as a documented
differentiator but don't promise it in the Phase 3 demo.

---

## 3. Copyable features (Vega Hand) — distilled from competitors

### Corrections to the earlier landscape note

- **iFlytek DOES list Vietnamese** among its 18 offline/"Chat Translation"
  languages — the earlier note said "Vietnamese not confirmed." Weaken the
  "nobody supports VI" claim to "nobody *prioritizes* VI, and no one does
  Vietnamese industrial-safety translation."
- **Timekettle now ships a handheld offline translator (NEW T1, 44 offline
  packs, 0.2 s claimed, dual speakers/mics, EDC noise reduction)** — but
  **no Vietnamese offline pack** anywhere in their line (VI is online/photo
  only). Our fully-offline VI↔EN/ZH/KO set is still a gap they don't fill.
- **RelayX hard specs captured:** MIL-STD-810H, IP68+IP69K, 110 dB SPL dual
  speakers, 2900 mAh "two shifts," 3 mics, circular LED, **Class 1 Div 2**,
  SOS = 5× rapid-press → <1 s alert.

### Master feature list → verdict

| Feature | Has it | Verdict for Vega Hand |
|---|---|---|
| Offline language-pack download/swap per pair | Timekettle T1 (44), iFlytek (18) | BOTH — we lazy-load per pair; copy the *framing* in the pitch |
| 4-mic array + AI noise cancel | iFlytek | EVALUATE — benchmark that 2 mics may be optimistic |
| High-SPL speaker + playback EQ | RelayX 110 dB, Weavix 105 dB, iFlytek EQ | COPY — target ≥100 dB + EQ |
| Glove-friendly large buttons | Weavix | COPY (cheap) |
| Dedicated SOS trigger (5× press, <1 s) | Relay, Weavix | COPY — long/triple-press on PTT |
| Multi-state LED status | RelayX, Weavix | ALREADY PLANNED; formalize states |
| **Vibration/haptic feedback** | (none found) | **BUILD — unclaimed, cheap, glove-friendly** |
| One-click / double-click activation | Timekettle T1 | COPY (cheap) |
| Translation history log + USB export | iFlytek | COPY — industrial audit trail |
| Camera/photo OCR translation | iFlytek, Pocketalk, T1 | SKIP — needs a display |
| Auto language detection | iFlytek | SKIP — fixed pairs simpler |
| Ruggedization MIL-STD-810H / IP68+IP69K / C1D2 | RelayX, Weavix | NOTE CEILING — we're IP54, fine now |
| Wearability (lanyard / belt clip) | Timekettle, RelayX | COPY (cheap accessory) |
| Utterance cap | Pocketalk 30 s | ALREADY — our ≤10 s is conservative |
| Industrial safety-glossary enforcement | (nobody) | **BUILD — our differentiator** |
| 100% offline industrial translation | (nobody) | **OWN — lead the pitch** |
| Vietnamese-first (incl. offline) | (nobody) | **OWN — now nuanced by iFlytek** |

### The three buckets

**(a) Cheap to copy — do in Phase 3 prototype:** multi-state LED; glove-friendly
buttons; high-SPL speaker + EQ; **haptic feedback** (unclaimed); SOS trigger on
PTT; lanyard/belt-clip; translation-log export.

**(b) Build beyond competitors (differentiators):** fully-offline industrial
translation; Vietnamese-first; industrial safety-glossary enforcement; measured
WER-vs-SNR noise curve (not marketing accuracy); un-brickable hardware; latency
HUD.

**(c) Already handled — don't spend time:** PTT, LED, denoise mics, offline
packs, utterance cap, auto-detect, ≥8 h battery, eSIM monetization (that's a
business-model gap, not a feature).

---

## Sources

Full per-claim citations are in the three research note files under
`research_notes/vega-air-and-feature-research-2026-08/` (drone-pa-broadcast.md,
ego-noise-cancellation.md, translation-device-features.md). Chinese aftermarket
SPL/range claims and any item flagged "unverified" above should be treated as
marketing until independently measured.
