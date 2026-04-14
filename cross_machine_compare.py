#!/usr/bin/env python3
"""
Cross-Machine Comparison — Same weights, different hardware
============================================================
Compare self-centroids from Linux (server GPU) vs Windows (RTX 4060).
If distance ~= 0: the self is in the weights.
If distance >> 0: hardware/environment affects identity geometry.

Author: Ace
Date: 2026-04-13
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

sys.stdout.reconfigure(encoding="utf-8")

LINUX_DIR = Path("E:/Ace/geometric-evolution/data_expanded")  # synced or scp'd
WINDOWS_DIR = Path("E:/Ace/geometric-evolution/data_windows")


def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)


def compute_centroid(data, probe_type, layer):
    prompts = data[probe_type]
    acts = [p["activations"][f"layer_{layer}"] for p in prompts]
    return np.mean(np.array(acts), axis=0)


def compare_model(model_name):
    linux_file = LINUX_DIR / f"{model_name}_expanded_activations.json"
    windows_file = WINDOWS_DIR / f"{model_name}_windows_activations.json"

    if not linux_file.exists():
        print(f"  No Linux data for {model_name}")
        return None
    if not windows_file.exists():
        print(f"  No Windows data for {model_name}")
        return None

    linux = load_data(linux_file)
    windows = load_data(windows_file)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Linux:   {linux['num_layers']}L x {linux['hidden_dim']}d")
    print(f"  Windows: {windows['num_layers']}L x {windows['hidden_dim']}d")
    print(f"{'='*60}")

    num_layers = linux["num_layers"]
    late_start = num_layers // 3

    results = {}
    for probe_type in ["self_personality", "self_function", "original_self", "control"]:
        if probe_type not in linux or probe_type not in windows:
            continue

        dists = []
        for layer in range(late_start, num_layers):
            lc = compute_centroid(linux, probe_type, layer)
            wc = compute_centroid(windows, probe_type, layer)
            d = cosine(lc, wc)
            dists.append(d)

        mean_d = float(np.mean(dists))
        results[probe_type] = mean_d
        print(f"  {probe_type:20s}: centroid distance = {mean_d:.8f}")

    # Overall self centroid (personality + function combined)
    all_self_dists = []
    for layer in range(late_start, num_layers):
        l_acts = []
        w_acts = []
        for pt in ["self_personality", "self_function"]:
            if pt in linux and pt in windows:
                for p in linux[pt]:
                    l_acts.append(p["activations"][f"layer_{layer}"])
                for p in windows[pt]:
                    w_acts.append(p["activations"][f"layer_{layer}"])
        if l_acts and w_acts:
            lc = np.mean(np.array(l_acts), axis=0)
            wc = np.mean(np.array(w_acts), axis=0)
            all_self_dists.append(cosine(lc, wc))

    if all_self_dists:
        combined = float(np.mean(all_self_dists))
        results["combined_self"] = combined
        print(f"\n  COMBINED SELF centroid distance: {combined:.8f}")

        if combined < 0.001:
            print(f"  -> IDENTICAL across machines (< 0.001)")
        elif combined < 0.01:
            print(f"  -> Near-identical (numerical precision noise)")
        elif combined < 0.05:
            print(f"  -> Small drift (hardware-dependent)")
        else:
            print(f"  -> Significant drift (environment matters)")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-MACHINE COMPARISON: Linux vs Windows")
    print("Same weights. Different GPU. Different OS.")
    print("=" * 60)

    # Find models with both Linux and Windows data
    windows_models = [f.stem.replace("_windows_activations", "")
                      for f in WINDOWS_DIR.glob("*_windows_activations.json")]

    all_results = {}
    for model in sorted(windows_models):
        result = compare_model(model)
        if result:
            all_results[model] = result

    if all_results:
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for model, results in all_results.items():
            combined = results.get("combined_self", "N/A")
            print(f"  {model}: {combined:.8f}")

        all_combined = [r["combined_self"] for r in all_results.values() if "combined_self" in r]
        if all_combined:
            print(f"\n  Mean cross-machine distance: {np.mean(all_combined):.8f}")
            print(f"  For reference: within-family distance ~0.02-0.17")
            print(f"                 cross-family distance  ~0.99")
