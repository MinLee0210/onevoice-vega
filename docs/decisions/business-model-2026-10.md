# Business model (2026-10-13)

Business Viability is 25% of judging, and this was the one pillar with no
concrete answer: every competitor (Weavix, Relay, Vasco, Pocketalk) monetizes
via a subscription/data-plan tied to cloud connectivity, which OneVoice's
offline architecture **structurally cannot replicate**. This proposes what to
sell instead, anchored to the device's actual strengths.

## The constraint, turned into the pitch

Offline = no per-utterance cloud cost, no data-plan revenue, and no
usage-metered hook. It also = **no recurring network dependency to monetize**.
So the revenue has to come from things a factory actually pays for: hardware,
a per-seat/site software license, domain content, and support — all of which
an air-gapped fleet needs more, not less, of.

## Revenue lines

| Line | What | Why it fits offline | Anchor |
|---|---|---|---|
| **1. Hardware margin** | Sell Hand units / Air payloads at a device margin | Standard B2B hardware; no cloud to subsidize the device | Competitor rugged PTT radios run hundreds of $/unit |
| **2. Site/seat software license** | Annual license per site (or per 10 units) for the engine (ASR/MT/TTS + glossary), sold with the hardware | Replaces the subscription competitors charge — but it's a **license for the offline engine**, not cloud usage | Weavix/Relay sell per-user/mo; we can undercut on a 3-yr TCO with no data plan |
| **3. Industry vocab packs** | One-time + annual update for the industrial termbase (safety, machine names, logistics), per vertical | The glossary is the differentiator; content is versionable and updatable on-device via USB/OTA-in-LAN | "App-store for terminology" — small, high-margin, recurring |
| **4. Deployment & support** | Commissioning, fleet provisioning, on-site training, SLA | Air-gapped fleets can't self-serve cloud onboarding; they pay for hands-on rollout | Standard enterprise services revenue |
| **5. Qualcomm ecosystem** | Reference design / Dragonwing showcase; joint GTM | Qualcomm wants Dragonwing design wins and AI Hub adoption | Strategic, not a direct revenue line, but de-risks 1–4 |

## Pricing sketch (to validate in Phase 3–4)

- **Hand**: hardware + 3-yr engine license bundled (avoid a separate "cloud
  plan" line the buyer distrusts). Target a per-seat TCO **below** a
  cloud-translator subscription over 3 years, since there is zero data cost.
- **Air**: premium payload (one pod serves a whole site) — priced per site,
  not per seat; the value is the multilingual broadcast, which no loudhailer
  offers at all.
- **Vocab packs**: cheap impulse add-on per industry (construction, logistics,
  hospital), renewed yearly.

## Why this beats the competitor structure

- Competitors' offline modes degrade to a ~5k-word generic vocabulary. Ours is
  **full-vocabulary and permanent** — so a per-seat license is defensible where
  a subscription that "stops working offline" is not.
- **Data privacy as a selling point:** no audio/text leaves the site. For
  automotive/defense-adjacent factories this alone justifies premium pricing
  and removes a procurement blocker.
- **Two carriers, one procurement contact:** the safety officer buying Air is
  the same buyer as the Hand fleet — one BOM lineage, one license, one support
  contract (see the proposal's §3).

## Open questions to close

- Exact BOM cost of the QCS6490 module + rugged enclosure (drives line 1 margin).
- Whether buyers prefer capex (license bundled) vs a small annual renew — needs
  Phase 4 field conversations.
- Willingness to pay for vocab packs vs expecting them included.
- Whether a **fleet-management console** (on-prem/LAN, offline) is expected —
  it would be a natural line 2/4 add-on and a moat vs point devices.

## Not doing

- No cloud subscription, no data plan, no per-minute metering — those are the
  competitor models we are explicitly positioned against.
