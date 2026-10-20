"""Plot EMD and IME results from SNR sweep .json outputs.

Reads sweep_part*.json from results/patternlocal-repro-{xor-corr,xor-white}-sweep-*,
merges alphas, and produces two figures:
  - sweep_emd.png   : EMD vs alpha for all methods (2 panels: XOR CORR, XOR WHITE)
  - sweep_ime.png   : IME vs alpha for all methods (2 panels: XOR CORR, XOR WHITE)
Error bars = std deviation across evaluated samples.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
OUT_DIR = RESULTS_DIR / "figures"

METHODS = [
    ("lime",               "LIME",                 "tab:blue",   "o",  "-"),
    ("patternlocal",       "PatternLocal",         "tab:orange", "s",  "-"),
    ("analytical_grad",    "Analytical-grad",      "tab:green",  "^",  "-"),
    ("analytical_ig",      "Analytical-IG",        "tab:red",    "D",  "-"),
    ("plain_grad",         "Plain Grad",           "tab:purple", "v",  "--"),
    ("plain_ig",           "Plain IG",             "tab:brown",  "P",  "--"),
    ("plain_smoothgrad",   "Plain SmoothGrad",     "tab:gray",   "*",  "--"),
]

SCENARIOS = [
    ("xor_corr",  "XOR CORR"),
    ("xor_white", "XOR WHITE"),
]


def load_sweep(scenario_prefix):
    """Merge all sweep_part*.json for a scenario into {alpha: {method: {emd_mean,...}}}."""
    pattern = str(RESULTS_DIR / f"patternlocal-repro-{scenario_prefix}-sweep-*" / "sweep_part*_results.json")
    merged = {}
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            d = json.load(fh)
        for alpha_key, entry in d.items():
            if not isinstance(entry, dict) or "alpha" not in entry:
                continue
            alpha = float(entry["alpha"])
            merged[alpha] = entry
    return dict(sorted(merged.items()))


def extract_metric(merged, metric):
    """Return (alphas, {method: (mean_list, std_list)})."""
    alphas = sorted(merged.keys())
    results = {}
    for method_key, *_ in METHODS:
        means, stds = [], []
        for a in alphas:
            entry = merged[a]
            if method_key in entry and f"{metric}_mean" in entry[method_key]:
                means.append(entry[method_key][f"{metric}_mean"])
                stds.append(entry[method_key][f"{metric}_std"])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        results[method_key] = (np.array(means), np.array(stds))
    return alphas, results


def plot_metric(metric, data_by_scenario, out_path, scenarios):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, (scenario_key, scenario_label) in zip(axes, scenarios):
        alphas, results = data_by_scenario[scenario_key]
        for method_key, label, color, marker, ls in METHODS:
            means, stds = results[method_key]
            ax.errorbar(alphas, means, yerr=stds, label=label,
                        color=color, marker=marker, linestyle=ls,
                        capsize=3, markersize=6, linewidth=1.8, alpha=0.9)

        ax.set_xlabel("α (SNR)", fontsize=13)
        ax.set_title(scenario_label, fontsize=14, fontweight="bold")
        ax.set_xticks(alphas)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel(metric, fontsize=13)

    # Single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{metric} across SNR levels", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load merged results per scenario
    loaded = {}
    for scenario_key, _ in SCENARIOS:
        prefix = scenario_key.replace("_", "-")  # xor-corr, xor-white
        merged = load_sweep(prefix)
        if not merged:
            print(f"Warning: no results for {scenario_key}")
            continue
        loaded[scenario_key] = merged
        print(f"{scenario_key}: {len(merged)} alphas ({sorted(merged.keys())})")

    active_scenarios = [(k, l) for k, l in SCENARIOS if k in loaded]

    for metric in ["emd", "ime"]:
        data_by_scenario = {}
        for scenario_key, _ in active_scenarios:
            alphas, results = extract_metric(loaded[scenario_key], metric)
            data_by_scenario[scenario_key] = (alphas, results)

        plot_metric(metric.upper(), data_by_scenario,
                    OUT_DIR / f"sweep_{metric}.png", active_scenarios)


if __name__ == "__main__":
    main()
