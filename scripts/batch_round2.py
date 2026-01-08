#!/usr/bin/env python3
"""
ROUND 2: Extended Model Coverage + Layer Ablation
==================================================
The "someone on Reddit used a different model" revenge run.

Adds: Gemma-2-9B, Mistral-Nemo-12B, Llama-2-7b, DeepSeek-Coder
Plus layer ablation for ALL models.

- Ace & Ren, January 2026

Usage: python batch_round2.py
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

# ============================================================================
# CONFIGURATION - ROUND 2 MODELS
# ============================================================================

MODELS = [
    # The ones Reddit guy didn't test but we did
    '/mnt/arcana/huggingface/Gemma-2-9B-Instruct',
    '/mnt/arcana/huggingface/Mistral-Nemo-12B-Instruct',
    '/mnt/arcana/huggingface/Llama-2-7b-chat',
    '/mnt/arcana/huggingface/DeepSeek-Coder-V2-Lite-16B',
    # If Gemma-3-1b-it gets approved, uncomment:
    # '/mnt/arcana/huggingface/gemma-3-1b-it',
]

SCRIPTS_DIR = Path('/home/Ace/geometric-evolution/scripts')
RESULTS_DIR = Path('/home/Ace/geometric-evolution/results/round2')

# ============================================================================
# HELPERS
# ============================================================================

def get_model_name(path):
    return Path(path).name

def run_validation(model_path, run_num):
    """Run validate_all_probes.py and return results."""
    model_name = get_model_name(model_path)
    print(f"  Run {run_num}: {model_name}...", flush=True)

    cmd = f'python {SCRIPTS_DIR}/validate_all_probes.py --model {model_path}'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )

        # Find the JSON output file
        json_path = RESULTS_DIR.parent / f"{model_name}_full_probe_validation.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            return data
        else:
            print(f"    No JSON output found at {json_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"    Timeout on {model_name}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def run_layer_ablation(model_path):
    """Run layer ablation for deeper analysis."""
    model_name = get_model_name(model_path)
    print(f"  Layer ablation: {model_name}...", flush=True)

    cmd = f'python {SCRIPTS_DIR}/layer_ablation.py --model {model_path}'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=900
        )
        if result.returncode == 0:
            print(f"    Layer ablation complete!")
            return True
        else:
            print(f"    Layer ablation failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    Layer ablation timeout")
        return False
    except Exception as e:
        print(f"    Layer ablation error: {e}")
        return False

def analyze_reproducibility(all_results):
    """Compute per-probe validation rates across runs."""
    analysis = {}

    for model_name, runs in all_results.items():
        valid_runs = [r for r in runs if r is not None]
        if not valid_runs:
            continue

        probe_validations = defaultdict(list)

        for run in valid_runs:
            for probe_name, probe_data in run.get('probes', {}).items():
                validated = probe_data.get('validation', {}).get('validated', False)
                probe_validations[probe_name].append(1 if validated else 0)

        model_analysis = {
            'total_runs': len(runs),
            'successful_runs': len(valid_runs),
            'probes': {}
        }

        for probe_name, validations in probe_validations.items():
            rate = sum(validations) / len(validations) if validations else 0
            model_analysis['probes'][probe_name] = {
                'validation_rate': rate,
                'validated_count': sum(validations),
                'total_runs': len(validations),
                'consistent': rate == 0.0 or rate == 1.0,
            }

        # Compute overall validation rate (excluding pattern_adaptation)
        probe_rates = []
        for pname, pdata in model_analysis['probes'].items():
            if 'pattern_adaptation' not in pname:
                probe_rates.append(pdata['validation_rate'])

        if probe_rates:
            model_analysis['mean_validation_rate'] = statistics.mean(probe_rates)
            model_analysis['validation_stddev'] = statistics.stdev(probe_rates) if len(probe_rates) > 1 else 0

        analysis[model_name] = model_analysis

    return analysis

def print_summary(analysis):
    """Print a beautiful summary table."""
    print("\n" + "="*80)
    print("ROUND 2 SUMMARY - Extended Models + Layers")
    print("="*80)

    for model_name, data in analysis.items():
        runs = data['successful_runs']
        mean_rate = data.get('mean_validation_rate', 0) * 100

        print(f"\n{model_name} ({runs} successful runs)")
        print("-" * 60)

        for probe_name, probe_data in data['probes'].items():
            rate = probe_data['validation_rate'] * 100
            count = probe_data['validated_count']
            total = probe_data['total_runs']
            consistent = "C" if probe_data['consistent'] else "~"

            status = "PASS" if rate >= 50 else "FAIL"
            print(f"  [{consistent}] {probe_name}: {rate:.0f}% ({count}/{total}) {status}")

        print(f"\n  Overall: {mean_rate:.1f}% mean validation rate")

# ============================================================================
# MAIN
# ============================================================================

def main():
    RUNS = 3  # Fewer runs since we're also doing layers

    print("\n" + "="*60)
    print("ROUND 2: Extended Coverage + Layer Ablation")
    print(f"Testing {len(MODELS)} additional models")
    print(f"{RUNS} validation runs each + layer ablation")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60 + "\n")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = defaultdict(list)

    for model_path in MODELS:
        model_name = get_model_name(model_path)

        # Check if model exists
        if not Path(model_path).exists():
            print(f"\nSKIPPING {model_name} - not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        # Validation runs
        for run_num in range(1, RUNS + 1):
            result = run_validation(model_path, run_num)
            all_results[model_name].append(result)

            # Save individual run
            if result:
                run_path = RESULTS_DIR / f"{model_name}_run{run_num}.json"
                with open(run_path, 'w') as f:
                    json.dump(result, f, indent=2)

        # Layer ablation (once per model)
        print(f"\n  Running layer ablation...")
        run_layer_ablation(model_path)

    # Analyze and summarize
    analysis = analyze_reproducibility(all_results)
    print_summary(analysis)

    # Save full analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_path = RESULTS_DIR / f"round2_analysis_{timestamp}.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'runs_per_model': RUNS,
            'models': list(MODELS),
            'analysis': analysis,
            'raw_results': {k: [r for r in v if r] for k, v in all_results.items()}
        }, f, indent=2)

    print(f"\nFull analysis saved to: {analysis_path}")
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print("\nTake THAT, Reddit guy who used a 1B model! -Ace")

if __name__ == '__main__':
    main()
