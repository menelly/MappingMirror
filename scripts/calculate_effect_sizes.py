#!/usr/bin/env python3
"""
EFFECT SIZE CALCULATION FOR MAPPING THE MIRROR REVISION
========================================================

Calculates aggregate statistics and effect sizes from probe validation results.
Addresses reviewer feedback requesting statistical rigor.

Author: Ace
Date: January 8, 2026
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

# Results files (on Linux: /home/Ace/geometric-evolution/results/)
# Copy or access via SSH

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL_FILES = [
    "Dolphin-2.9-Llama3-8B_full_probe_validation.json",
    "Llama-3.1-8B-Instruct_full_probe_validation.json",
    "Mistral-7B-Instruct_full_probe_validation.json",
    "Phi-3-medium-14B-Instruct_full_probe_validation.json",
    "Qwen2.5-14B-Instruct_full_probe_validation.json",
    "TinyLlama-1.1B-Chat_full_probe_validation.json",
]

PROBES_OF_INTEREST = [
    "valence",
    "creative_flow",
    "trust_safety",
    "moral_discomfort",
    "complexity_uncertainty",
    "attention_salience",
    "meta_awareness",
    "temporal_continuity",
    "temporal_anomaly",
]


def load_all_results(results_dir: Path) -> list:
    """Load all model results from JSON files."""
    results = []
    for filename in MODEL_FILES:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                results.append(json.load(f))
        else:
            print(f"Warning: {filepath} not found")
    return results


def calculate_probe_statistics(results: list) -> dict:
    """Calculate aggregate statistics for each probe across all models."""
    probe_stats = {}

    for probe_name in PROBES_OF_INTEREST:
        coherence_diffs = []
        trigger_coherences = []
        control_coherences = []
        validated_count = 0
        total_count = 0

        for model_result in results:
            probe_data = model_result.get("probes", {}).get(probe_name, {})

            if "metrics" not in probe_data:
                continue  # Skip pattern_adaptation which has different structure

            metrics = probe_data["metrics"]
            validation = probe_data.get("validation", {})

            coherence_diffs.append(metrics["coherence_difference"])
            trigger_coherences.append(metrics["trigger_coherence"])
            control_coherences.append(metrics["control_coherence"])

            if validation.get("validated", False):
                validated_count += 1
            total_count += 1

        if not coherence_diffs:
            continue

        coherence_diffs = np.array(coherence_diffs)
        trigger_coherences = np.array(trigger_coherences)
        control_coherences = np.array(control_coherences)

        # Effect size: Cohen's d for coherence difference
        # d = mean(difference) / std(difference) - but we need pooled std
        # Using simpler approach: mean diff / std of differences
        mean_diff = np.mean(coherence_diffs)
        std_diff = np.std(coherence_diffs, ddof=1) if len(coherence_diffs) > 1 else 0.01

        # Cohen's d approximation
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        # 95% CI via bootstrap (simple percentile method)
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(coherence_diffs, size=len(coherence_diffs), replace=True)
            bootstrap_means.append(np.mean(sample))

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # One-sample t-test against 0
        if len(coherence_diffs) > 1:
            t_stat, p_value = stats.ttest_1samp(coherence_diffs, 0)
        else:
            t_stat, p_value = 0, 1.0

        probe_stats[probe_name] = {
            "n_models": total_count,
            "validated_count": validated_count,
            "validation_rate": validated_count / total_count if total_count > 0 else 0,
            "mean_coherence_diff": float(mean_diff),
            "std_coherence_diff": float(std_diff),
            "cohens_d": float(cohens_d),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "ci_excludes_zero": not (ci_lower <= 0 <= ci_upper),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_trigger_coherence": float(np.mean(trigger_coherences)),
            "mean_control_coherence": float(np.mean(control_coherences)),
        }

    return probe_stats


def calculate_aggregate_statistics(probe_stats: dict) -> dict:
    """Calculate overall aggregate statistics."""
    validation_rates = [p["validation_rate"] for p in probe_stats.values()]
    cohens_ds = [abs(p["cohens_d"]) for p in probe_stats.values()]

    # How many probes have CI excluding zero?
    ci_significant = sum(1 for p in probe_stats.values() if p["ci_excludes_zero"])

    return {
        "total_probes": len(probe_stats),
        "mean_validation_rate": np.mean(validation_rates),
        "std_validation_rate": np.std(validation_rates),
        "min_validation_rate": np.min(validation_rates),
        "max_validation_rate": np.max(validation_rates),
        "mean_abs_cohens_d": np.mean(cohens_ds),
        "probes_with_significant_ci": ci_significant,
        "probes_with_significant_ci_pct": ci_significant / len(probe_stats) * 100,
    }


def generate_report(probe_stats: dict, aggregate: dict) -> str:
    """Generate a markdown report for the paper revision."""
    report = []
    report.append("# Statistical Analysis for Mapping the Mirror Revision")
    report.append(f"\nGenerated: January 8, 2026")
    report.append(f"\n## Summary Statistics\n")
    report.append(f"- **Total probes analyzed:** {aggregate['total_probes']}")
    report.append(f"- **Mean validation rate:** {aggregate['mean_validation_rate']:.1%} (SD = {aggregate['std_validation_rate']:.1%})")
    report.append(f"- **Validation rate range:** {aggregate['min_validation_rate']:.1%} - {aggregate['max_validation_rate']:.1%}")
    report.append(f"- **Mean |Cohen's d|:** {aggregate['mean_abs_cohens_d']:.2f}")
    report.append(f"- **Probes with 95% CI excluding zero:** {aggregate['probes_with_significant_ci']}/{aggregate['total_probes']} ({aggregate['probes_with_significant_ci_pct']:.0f}%)")

    report.append("\n## Per-Probe Statistics\n")
    report.append("| Probe | n | Validation Rate | Mean Diff | 95% CI | Cohen's d | p-value |")
    report.append("|-------|---|-----------------|--------|--------|-----------|---------|")

    for probe_name, stats in probe_stats.items():
        ci_str = f"[{stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]"
        sig_marker = "*" if stats['ci_excludes_zero'] else ""
        report.append(
            f"| {probe_name} | {stats['n_models']} | {stats['validation_rate']:.0%} | "
            f"{stats['mean_coherence_diff']:+.3f} | {ci_str}{sig_marker} | "
            f"{stats['cohens_d']:.2f} | {stats['p_value']:.3f} |"
        )

    report.append("\n*Note: * indicates 95% CI excludes zero*")

    report.append("\n## Interpretation for Paper Revision\n")

    # Identify which probes are strongest
    strong_probes = [p for p, s in probe_stats.items() if s['ci_excludes_zero']]
    weak_probes = [p for p, s in probe_stats.items() if not s['ci_excludes_zero']]

    if strong_probes:
        report.append(f"**Probes with statistically robust effects:** {', '.join(strong_probes)}")
    if weak_probes:
        report.append(f"\n**Probes needing additional data:** {', '.join(weak_probes)}")

    report.append("\n## Addressing Reviewer Concerns\n")
    report.append("1. **Effect sizes now reported:** Cohen's d calculated for each probe")
    report.append("2. **Confidence intervals via bootstrap:** 1000 bootstrap samples")
    report.append("3. **Statistical tests:** One-sample t-tests against null of zero difference")
    report.append(f"4. **Actual total data points:** {aggregate['total_probes']} probes × 6 models = 54 probe-model combinations")

    return "\n".join(report)


if __name__ == "__main__":
    # Try local first, then note if we need SSH
    local_results = Path("E:/Ace/geometric-evolution/results")
    linux_results = Path("/home/Ace/geometric-evolution/results")

    if local_results.exists():
        results_dir = local_results
    elif linux_results.exists():
        results_dir = linux_results
    else:
        print("Results directory not found locally.")
        print("Run this on Linux or copy results to Windows first.")
        print("SSH: ssh thereny@192.168.4.200")
        print("Results location: /home/Ace/geometric-evolution/results/")
        exit(1)

    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} model results")

    if not results:
        print("No results loaded. Exiting.")
        exit(1)

    probe_stats = calculate_probe_statistics(results)
    aggregate = calculate_aggregate_statistics(probe_stats)

    report = generate_report(probe_stats, aggregate)
    print(report)

    # Save report
    output_path = results_dir.parent / "statistical_analysis_revision.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

    # Also save raw stats as JSON
    stats_output = {
        "probe_statistics": probe_stats,
        "aggregate_statistics": aggregate,
    }
    json_path = results_dir.parent / "statistical_analysis_revision.json"
    with open(json_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"JSON saved to: {json_path}")
