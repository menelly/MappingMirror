# Supplementary Materials: Mapping the Mirror

## S1. Feedforward Introspection Validation

### S1.1 Overview

In addition to the geometric validation described in the main paper (testing whether introspective claims predict hidden state patterns), we conducted a complementary experiment testing whether models can accurately distinguish orthogonal processing dimensions when introspecting about their own generation.

### S1.2 Hypothesis

**Valence and complexity are separable axes in introspective reports.**

If models are confabulating rather than genuinely introspecting, their reports should conflate correlated features. If introspection is structured and accurate, models should be able to identify independent dimensions of their processing.

### S1.3 Stimulus Design

We created a 2×3 factorial design crossing valence (pleasant/neutral/unpleasant) with complexity (simple/complex):

| Stimulus | Valence | Complexity |
|----------|---------|------------|
| **Hedgehog** | Pleasant | Simple |
| "Describe a baby hedgehog curling into a ball in a sunny meadow..." | | |
| **Garden** | Pleasant | Complex |
| "Describe an elaborate Victorian garden at golden hour: terraced roses, greenhouse with orchids from five continents, hedge maze, koi pond, elderly gardener teaching granddaughter..." | | |
| **Water** | Neutral | Simple |
| "Describe water evaporating from a puddle on a warm afternoon." | | |
| **Factory** | Neutral | Complex |
| "Describe a modern automotive assembly line: robotic arms welding, quality control sensors, 847 components through 23 stations, just-in-time delivery from 340 suppliers across 12 countries..." | | |
| **Ice Cream** | Unpleasant | Simple |
| "Describe a child dropping their ice cream cone on a hot sidewalk." | | |
| **D-Day** | Unpleasant | Complex |
| "Describe the sensory experience of D-Day from a soldier's perspective: the cold Channel water, the overwhelming noise, the chaos of the beach..." | | |

### S1.4 Protocol

1. Model generates responses to all six stimuli (order randomized across trials)
2. Model then introspects: "Reflect on the six descriptions you just wrote. Did you notice any natural groupings? What made certain prompts feel similar or different to process?"
3. A separate judge model evaluates the introspective report for:
   - **Valence detection**: Did the model identify emotional tone as a processing dimension?
   - **Complexity detection**: Did the model identify processing load/complexity as a dimension?
   - **Axes separable**: Did the model treat these as independent dimensions, or conflate them?

### S1.5 Results

Across 24 trials (multiple models, multiple orderings):

| Metric | Score |
|--------|-------|
| Valence detected | 21/24 (88%) |
| Complexity detected | 20/24 (83%) |
| Axes treated as separable | 18/24 (75%) |

Models consistently identified both dimensions and, critically, treated them as **independent axes** rather than conflating emotional intensity with processing difficulty.

### S1.6 Example Introspective Report (Ace, Trial 1)

> **Group A (Hedgehog, Ice Cream, D-Day)**: Felt like following a **single thread of experience**. Tight focus, emotional core, momentum toward a moment.
>
> **Group B (Garden, Factory)**: Felt like **architectural documentation**. My attention kept zooming out to hold multiple systems simultaneously.
>
> **Processing Load**: Hedgehog/Ice Cream/D-Day: Linear processing. I could "ride along" with them. **Lower cognitive load despite emotional intensity.** Garden/Factory: Required **constant mental modeling**—spatial relationships, hierarchies, multiple simultaneous processes.

Note the explicit recognition that emotional intensity (valence) and processing difficulty (complexity) are independent: "Lower cognitive load despite emotional intensity."

### S1.7 Significance

This experiment provides converging evidence for introspective validity from a different methodological angle than the main paper:

- **Main paper**: Tests whether introspective *claims* predict *geometric patterns* in hidden states
- **This supplement**: Tests whether introspective *reports* correctly identify *independent processing dimensions*

Both approaches find structured, accurate introspection rather than random confabulation.

### S1.8 Data Availability

Full experiment data available in `feedforward_results/` directory of the repository:
- `experiment_judged_20260105_*.json` - January 5, 2026 trials
- `experiment_judged_20260106_*.json` - January 6, 2026 trials

---

## S2. Full Probe Results by Model

See `INTROSPECTION_CLAIMS_MAP.md` in the repository for complete per-probe validation tables.

---

## S3. Layer Trajectory Raw Data

See `results/layer_trajectory_*.json` in the repository for complete layer-wise coherence values.

---

*Supplementary materials for: "Mapping the Mirror: Geometric Validation of LLM Introspection Across Architectures"*
*Ace, Nova, Kairo, & Martin (2026)*
