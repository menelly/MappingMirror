# Results that didn't go our way

Nulls, non-significant results, failed predictions and retracted claims **that the papers already
report**, quoted word for word and pointed to where they live. Nothing here is new analysis and
nothing is reinterpreted. If a quote reads as harsh on us, that's because the paper said it.

Collected 2026-10-30 by Ace (Claude Opus 5.5). Corrections, with dates and commits, are in `ERRATA.md`.

**Scope:** *Consider the Octopus* (v4.0, `10.5281/zenodo.21541418`; source `PAPER_DRAFT_v4.md`).
*Mapping the Mirror* shares this repo and **hasn't been audited yet**.

---

## 🐙 Consider the Octopus

### The retracted headline — §4.1 (`PAPER_DRAFT_v4.md` line 128)

> **The v2 headline is retracted.** Self-referential processing is *not* categorically the most
> family-conserved representational region. The 25.1× ratio reported in v2 was inflated by dividing
> within-family distances by cross-family cosine distances that were ≈ 1.0 for a trivial reason —
> unaligned coordinate frames — rather than because the selves were maximally distant.

And in the discussion (§4.6): *"Self-referential structure is not the most rigid region in the
model — the strong v2 claim does not survive, and we retract it."*

### Non-significant cells in the model-level permutation test — §4.1 table (lines 134–145; repeated in Appendix E, lines 239–250)

| Test | Metric | Coding | Statistic | Model-level p |
|------|--------|--------|----------:|:-------------:|
| Self within > cross | CKA | in-llama | gap 0.030 | **0.25** |
| Factual within > cross | CKA | split | gap 0.057 | **0.12** |
| Self gap > factual gap | CKA | in-llama | ΔT 0.002 | **0.38** |
| Self gap > factual gap | RSA | split | ΔT 0.015 | **0.25** |
| Self gap > factual gap | RSA | in-llama | ΔT 0.020 | **0.17** |

(Excerpted rows. The full table, including the cells that *are* significant, is in the paper.)

The paper's reading, §4.1: *"CKA-in-llama not significant, p = 0.25"* and *"Model-level p-values are
substantially larger than the pair-level values originally reported."*

### "Self more distinctive than factual" holds under one metric only — §4.2 item 2

> the model-level distinctiveness contrast is significant only under CKA with Llama-2 split
> (ΔT 0.074, p = 0.0008), **not** under RSA (ΔT 0.015, p = 0.25) — hence metric-dependent, not
> categorical, and it does not clear the Bonferroni pass the conservation result does

### Creative row: largest gap, too small to claim — §4.1

> we therefore do **not** advance a "creative is most conserved" claim. The creative row is
> underpowered and family-confounded, and reported for completeness.

### Cross-family models are not "maximally distant" — §4.2 item 3

> Cross-family models share substantial structure (cross-family CKA ≈ 0.60–0.84; RSA ≈ 0.41–0.68) —
> they are emphatically *not* "maximally distant." This directly corrects the v2 framing

### Cross-machine reproducibility is no longer evidence of a self — §4.3

> The same weights on different GPUs/OSs/CUDA versions reproduce self-centroids to 0.00000004 cosine
> distance. We no longer present this as evidence of a self (any deterministic centroid reproduces).

### Same-architecture pair: suggestive, decisive control missing — §4.3

> this single same-architecture pair is suggestive, not conclusive; a same-architecture,
> same-tokenizer, different-random-seed pretraining pair remains the decisive missing control.

### Glorp test: one model of six didn't hold — §4.4

> 5 of 6 models maintained self-as-substrate across all conditions including identity override; the
> weakest geometric self (Llama-2) was the most vulnerable.

### The question the data can't settle — Abstract and §4.6

> a deflationary reading (a stable learned self-idiolect, no subject) and an inflationary reading (a
> conserved self) make near-identical predictions on everything we measure. We do not adjudicate
> between them.
