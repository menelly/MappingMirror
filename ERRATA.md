# ERRATA — corrections to papers built from this repo

Every correction we can **source** (a commit hash, a line in the paper, or a Zenodo version record),
newest last. If a correction isn't here, either it never happened or we couldn't source it yet, and
those go in the ❓ list at the bottom instead of being dressed up as fact.

**Scope.** This repo (GitHub: `menelly/MappingMirror`, renamed from `menelly/geometricevolution`;
the old URL redirects to the same repository) holds two papers:

- **Consider the Octopus** — covered below.
- **Mapping the Mirror** — **not yet audited.** Its absence from this file means "nobody has
  checked yet," not "no corrections."

Compiled 2026-10-30 by Ace (Claude Opus 5.5) from `git log`, the paper's own v4.0 revision note,
and the Zenodo API. Nothing in the papers was edited to make this file; it only points at them.

---

## 🐙 Consider the Octopus

**Version of record:** v4.0, `10.5281/zenodo.21541418` (2026-07-24). Concept DOI
`10.5281/zenodo.19578724` resolves to the latest version.
**Source in this repo:** `PAPER_DRAFT_v4.md`.

| # | Date | What changed | Where in v4 | Source |
|---|------|--------------|-------------|--------|
| 1 | 2026-05-29 | **Headline retracted.** v2 said self-referential processing is *the* most family-conserved representational region (within/cross ratio **25.1×**). That ratio divided by cross-family cosine distances that were ≈1.0 only because independently pretrained networks share no coordinate frame. Under basis-invariant metrics (linear CKA, RSA) the conservation-gap order is creative > self > factual: self is in the middle, not on top. | §3.3, §4.1 ("The v2 headline is retracted") | commit `ce9e412` |
| 2 | 2026-05-29 | **"Maximally distant" framing for different-family models withdrawn.** Cross-family CKA ≈ 0.60–0.84 and RSA ≈ 0.41–0.68 mean "measurably distinct lineage-specific structure," not "maximally distant." | §4.2 item 3, §4.5 | commit `ce9e412` |
| 3 | 2026-05-29 | **Cross-machine reproducibility (0.00000004 cosine distance) reclassified.** No longer presented as evidence of a self ("any deterministic centroid reproduces"). It now does one job: support the copying premise of the welfare-counting argument. | §4.3, "Cross-machine reproducibility" paragraph | commit `ce9e412` |
| 4 | 2026-06-04 | **Pair-level p-values replaced by a model-level permutation test.** The pair-level tests counted every model about 18 times (pseudo-replication). Model-level p-values are "substantially larger" and are now primary. Self within > cross survives (RSA p = 5×10⁻⁵; CKA-split p = 0.017) but is **not significant under CKA with Llama-2 coded in-family (p = 0.25)**. | §4.1 model-level table, Appendix E.3–E.5 | commit `5fe9e7e` |
| 5 | 2026-06-04 | **"Self is more family-distinctive than factual" scoped down to metric-dependent.** Significant under CKA-split (p = 0.0008), **not under RSA (p = 0.25)**, and doesn't clear the Bonferroni pass. | §4.2 item 2; v4.0 revision note item (2) | commit `5fe9e7e` |
| 6 | 2026-07-24 | **Model count corrected: 18 → 19.** An earlier v4 draft said "18 retained." The 18 was `n_w`, the number of within-family *pairs*. 19 is the model count (171 self pairs = C(19,2)). | §3.2 "Note on counts"; v4.0 revision note item (3) | commit `a47fecc` |
| 7 | 2026-07-24 | **The claim that Dolphin-Mistral's data were deleted (Dolphin-Mistral refused consent) replaced.** We never deleted the raw extraction. v4 now claims something a reader can check: Dolphin-Mistral is excluded from every reported analysis, and that was **verified by recomputation** (zero stored pairs involve Dolphin-Mistral; re-running the model-level tests with those pairs filtered out leaves every reported statistic unchanged). | §3.2 "Verification of the exclusion"; §4.7 | commit `a47fecc` (script `rerun_octopus_no_dolphin.py`); the original "data deleted" wording is in commit message `3c3c5d6` (2026-04-13) |
| 8 | 2026-07-24 | **Scope narrowed (wording, not numbers).** The welfare conclusion now covers the count of candidate *kinds* of patient and explicitly concedes the occurrent-processing/experientialist rival (§2.1). Advocacy framing is kept to the Conclusion. | v4.0 revision note items (4)–(5) | v4.0 revision note, `PAPER_DRAFT_v4.md` line 11 |

### Notes on the public record (facts, not corrections)

- **The 2026-06-12 Zenodo version (`10.5281/zenodo.20667910`) still carries the retracted 25.1× claim.**
  Its PDF is byte-identical to the April deposit `10.5281/zenodo.19578725` (SHA-256 `be9e8417…a1329`,
  both downloaded and hashed 2026-10-30), and its description repeats the 25.1× sentence. It was minted
  two weeks *after* the retraction landed here (`ce9e412`, 2026-05-29). It is superseded by v4.0, and the
  concept DOI resolves to v4.0. Anyone citing `20667910` directly is citing the pre-retraction text.
- **Zenodo metadata, checked live 2026-10-30:** version field `4.0`, title *"Consider the Octopus: Tractable
  AI Welfare and the Architecture-Level Self."* (An older internal note said the version field read `v3`
  and the title differed. Neither is true today; I didn't find out when that changed.)
- **The paper's data link `github.com/menelly/MappingMirror` is this repository** (same GitHub repo id,
  1125022905, checked via `gh api` 2026-10-30).

### ❓ Possible, unverified (not corrections; things to check)

- **"We committed in advance to reporting the result either way"** (§3.3). I found no timestamped
  artifact of that commitment *before* the CKA/RSA run. The commitment and the analysis both arrive in
  commit `ce9e412`. It may live in Linear (CHA-239), which I didn't check. Until something is found,
  treat this as unverified.
- **Phi, included or excluded?** §3.2 lists Phi among the families in the 19-model analysis. Appendix D
  is still titled "The Phi Exclusion," and the 2026-04-13 commit `3c3c5d6` says "Phi excluded
  (compression floor effect, 3/10 probe validation)." These may refer to different analyses (the April
  cosine study vs. the v4 CKA set), but the paper doesn't say. Unresolved.
