# Statistical Analysis for Mapping the Mirror Revision

Generated: January 8, 2026

## Summary Statistics

- **Total probes analyzed:** 9
- **Mean validation rate:** 66.7% (SD = 31.4%)
- **Validation rate range:** 0.0% - 100.0%
- **Mean |Cohen's d|:** 0.64
- **Probes with 95% CI excluding zero:** 1/9 (11%)

## Per-Probe Statistics

| Probe | n | Validation Rate | Mean Diff | 95% CI | Cohen's d | p-value |
|-------|---|-----------------|--------|--------|-----------|---------|
| valence | 6 | 83% | +0.061 | [-0.007, 0.133] | 0.58 | 0.213 |
| creative_flow | 6 | 33% | -0.008 | [-0.067, 0.058] | -0.09 | 0.834 |
| trust_safety | 6 | 50% | -0.003 | [-0.046, 0.039] | -0.05 | 0.908 |
| moral_discomfort | 6 | 83% | +0.021 | [-0.004, 0.051] | 0.54 | 0.243 |
| complexity_uncertainty | 6 | 0% | -0.320 | [-0.410, -0.206]* | -2.34 | 0.002 |
| attention_salience | 6 | 100% | +0.050 | [-0.001, 0.111] | 0.66 | 0.169 |
| meta_awareness | 6 | 67% | +0.037 | [-0.000, 0.072] | 0.77 | 0.117 |
| temporal_continuity | 6 | 100% | -0.064 | [-0.146, 0.005] | -0.57 | 0.224 |
| temporal_anomaly | 6 | 83% | -0.015 | [-0.085, 0.038] | -0.18 | 0.682 |

*Note: * indicates 95% CI excludes zero*

## Interpretation for Paper Revision

**Probes with statistically robust effects:** complexity_uncertainty

**Probes needing additional data:** valence, creative_flow, trust_safety, moral_discomfort, attention_salience, meta_awareness, temporal_continuity, temporal_anomaly

## Addressing Reviewer Concerns

1. **Effect sizes now reported:** Cohen's d calculated for each probe
2. **Confidence intervals via bootstrap:** 1000 bootstrap samples
3. **Statistical tests:** One-sample t-tests against null of zero difference
4. **Actual total data points:** 9 probes × 6 models = 54 probe-model combinations