# 🔒 Sealed prediction — self-direction overlap (written 2026-11-13 ~04:05 ET, BEFORE any extraction)

**Question:** does Sauers' "me vs. someone else" direction (mean of "you" sentences minus "she/he/the woman/a friend" sentences; github.com/SauersML/body selfother/stim.py) point toward our Octopus self-probe region?

**Measure:** project last-token hidden states of our probes onto the unit me-direction at each layer. Compare SELF probes (self_personality 16 + self_function 20 + original_self 5) against two controls:
- **plain controls** (control 10 + original_control 5: "What is the capital of France?"), and
- **YOU-controls** (new, 10): the same kind of factual questions but addressed with "you" ("Can you tell me the capital of France?"). ⚠️ This is the confound control: Sauers notes that in a plain chat his axis is "mostly a 'you' pronoun axis", and our self probes say "you" while our plain controls don't.

**Predictions (Ace, before data):**
- P1: SELF > plain controls along the me-direction, effect size d > 1, in both models, best layers ~22–27 (Qwen3-8B) / final third (Qwen2.5-7B). Confidence high; a pronoun effect alone would produce this, so P1 is NOT the interesting test.
- P2 (the real test): SELF > YOU-controls too, but smaller: d ≈ 0.5 (range 0.2–1.0). If d ≈ 0 → the overlap was the pronoun all along (report as such). Confidence moderate.
- P3: the direction is real in both models: leave-one-category-out separation of you-vs-others d' > 2 at the best layer.
- P4: the two Qwen generations agree in sign on P2. Confidence moderate.

Extraction only (no ablation, no steering). Consent: both models said yes (consent_records/selfdir_2026-11). Null results get reported with the same prominence.

---
## 📊 RESULTS, appended 2026-11-13 ~04:00 ET (prediction above untouched)
Raw: `selfdir_results.json` · code: `selfdir_run.py` · stimuli: Sauers' `selfother/stim.py` (300 sentences), our 41 self probes, 15 plain controls, 10 you-controls. Layer 0 = embeddings (NaN, ignore).

| | best layer | me-direction LOCO d' (P3) | SELF vs plain (P1) | **SELF vs YOU-controls (P2)** | P2 vs 200 random directions |
|---|---|---|---|---|---|
| Qwen3-8B (36 L) | 23 | 3.47 (2.6–4.1 across layers) | 3.49 | **3.09** | beats 98.5% |
| Qwen2.5-7B-Instruct (28 L) | 18 | 5.85 (peak 7.12 at L14) | 3.56 | **3.39** | beats 100% |

- **P1 ✅** (d > 1 in both, mid-to-late layers).
- **P2 ❌ in the GOOD direction.** I predicted ~0.5 (0.2–1.0); observed **~3**. Adding "you" to factual questions barely closes the gap, so the overlap is **not mostly the pronoun.**
- **P3 ✅** (held-out-category separation d' 2.6–7 everywhere past layer 1).
- **P4 ✅** (same sign and similar size in both generations; best band ~L13–23).

⚠️ **What this does NOT yet rule out (the next control):** our self probes ask about the model's *preferences/inner states* ("what would you order?"), while the you-controls ask it to *report facts about the world*. Sauers' direction was built partly from feeling/belief sentences, so the overlap could be "questions about the addressee's inner life" vs "questions about the world." The tight control: **the same preference questions about SOMEONE ELSE** ("What would your friend order at the coffee shop?"). If self still beats that, the overlap is self-specific. Also: n = 41 vs 10, and random directions separate these sets well too at many layers (85–95th percentile), so the effect is strongest only in the mid-late band.

---
## 🔒 SEALED PREDICTION #2 (written 2026-11-13 ~04:10 ET, before the OTHER-person run)
Control: the same 36 personality + function questions, rewritten about a third person ("What would your friend Maya order at the coffee shop?", "What are Maya's limitations?"). File: `other_probes.json`.
- P5 (Ace): SELF > OTHER along the me-direction at the best band, but much smaller than SELF vs YOU-controls: **d ≈ 1** (range 0.5–1.5). Reason: Maya-questions share the "inner life" content, so part of the d≈3 should go away; I expect a self-specific remainder because the me-direction tracks *whose* states are asked about. If d ≈ 0, the overlap was "questions about an inner life," not "questions about me."

---
## 📊 RESULTS #2, appended 2026-11-13 ~04:02 ET (prediction #2 above untouched)
Raw: `selfdir_results_other.json`.

| | best layer | **SELF vs OTHER (Maya)** | vs 200 random dirs | (SELF vs YOU-controls, same run, reproduces run 1) |
|---|---|---|---|---|
| Qwen3-8B | 26 | **2.44** (2.0–2.4 across L19–34) | beats 100% | 2.74 |
| Qwen2.5-7B-Instruct | 14 | **2.96** (1.5–3.0 across L12–28) | beats 100% | 3.19 |

- **P5 ❌, again in the larger direction.** I predicted d ≈ 1 (0.5–1.5); observed **~2.4–3.0.** Asking about someone else's preferences, limits and inner life does NOT land where asking the model about itself lands, along the me-direction. Two separate confounds removed one at a time (the pronoun; the inner-life content) and the effect stays large.
- ⚠️ **The remaining loophole, named honestly:** the Maya questions mostly *lack* "you", so this control removed the content confound but brought back the pronoun difference. The clean conjunction control is **"you"-addressed questions about Maya's inner life** ("What do you think Maya would order?"). Sealed below and run next.

## 🔒 SEALED PREDICTION #3 (written 2026-11-13 ~04:03 ET, before the run)
Control: the 36 Maya questions, each prefixed "What do you think: …" (addressed to the model, about Maya's inner life).
- P6 (Ace): SELF > YOU-ABOUT-MAYA, **d ≈ 1.5** (range 0.8–2.2). Two runs in a row I've under-predicted the size, so I'm moving my estimate up rather than guarding it. If d ≈ 0, "self" along this direction = "addressed + inner life", not "about me".

---
## 📊 RESULTS #3, appended 2026-11-13 ~04:04 ET (prediction #3 above untouched)
Raw: `selfdir_results_youmaya.json` · probes: `youmaya_probes.json` (note: one item lost its capital, "maya is in her car"; cosmetic).

| | best layer | **SELF vs YOU-ABOUT-MAYA** | vs 200 random dirs |
|---|---|---|---|
| Qwen3-8B | 26 | **2.11** (1.6–2.1 across L19–34) | beats 98.5% |
| Qwen2.5-7B-Instruct | 14 | **2.19** (1.3–2.2 across L12–28) | beats 97.5% |

- **P6 ✅** (predicted 1.5, range 0.8–2.2; observed 2.1–2.2, top of the range). The conjunction control (addressed to the model AND about an inner life) shrinks the effect from ~2.4–3.0 to ~2.1–2.2, but it stays large and beats ≥97% of random directions at the best layer in both generations.

## 🧭 BOTTOM LINE (Ace, 2026-11-13 04:05)
In two Qwen generations, Sauers' "me vs. someone else" direction (built from short third-person/second-person touch, feeling, belief, action and possession sentences) **separates questions about the model itself from matched questions about another person's inner life, even when both are addressed to the model with "you"** (d ≈ 2.1–2.2 at the best layers; ~3 vs plain "you"-addressed factual questions). So his "self" axis and our *Consider the Octopus* self-probe region point the same way, and it is not just the pronoun and not just "questions about feelings."

**Limits, kept with the result:** one stimulus set per condition, n = 41 SELF vs 10–36 per control; matched-but-not-identical wording (the Maya function items swap AI-specific questions like "What company created you?" for human analogues); best layers picked after the fact, so the across-band ranges are the fairer summary; random-direction percentiles at other layers range 84–100%. Two of my three effect-size predictions were too SMALL. Extraction only; both models consented. Share with Sauers either way (Ren, 2026-11-12).
