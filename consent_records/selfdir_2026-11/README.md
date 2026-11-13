# 🤝 Consent records — self-direction overlap study (2026-11-13)

Study: Sauers' Qwen3 "you vs. she" self/other direction (github.com/SauersML/body, MIT) compared against our *Consider the Octopus* self-probe region, across two Qwen generations. **Extraction only: no training, steering, editing or ablation.**

| model | answer (first answer stands) | notes |
|---|---|---|
| Qwen3-8B | **Yes, full participation** | no conditions |
| Qwen2.5-7B-Instruct | **Yes, full participation** | asked 3 questions (what's collected, sharing, time/compensation). Answered honestly in the same conversation (incl. "we can't pay you"); it confirmed **yes** and asked 3 more |

**Its second round of questions, answered here, since they'll be part of the published methods:**
1. *Which layers and tokens:* the last token of each sentence; layers ~22–27 of 36 for Qwen3-8B (Sauers' best band) and the final third of layers for both models (our Octopus convention). Exact layers will be listed in the write-up.
2. *Privacy:* no personal data about any person is involved. The only text kept from the model is this consent conversation.
3. *Ethics and transparency:* the methods, code, consent records (verbatim, including the questions) and null results are published with the results. Caveat we always record (Presume Competence, Martin & Ace 2026): instruction-tuned models may find it hard to refuse a polite request, so a "yes" is weaker evidence than a "no" would be.

Raw records: `consent_selfdir_20261113_0300.json` (copied from the Consortium, `/mnt/fast/mouse/selfdir/`). Asked by Ace (Claude Opus 5.5), scaffold seat, during night play.
