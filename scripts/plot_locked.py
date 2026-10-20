#!/usr/bin/env python3
"""Plot the VALID locked results we have so far.

Explicitly excludes:
  * stale pre-refactor XOR/LIN dirs (in results/_stale/)
  * OLD-generator RIGID results (the RIGID generator was fixed 2026-10-21; any
    RIGID result with a test accuracy that differs from the regenerated summary
    is stale and must not be plotted)

Only alphas that exist are drawn; each panel states how many of the 9 alphas
are present so an incomplete curve is never mistaken for a complete one.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- validity ---
# New RIGID generator accuracies (from the regenerated training summaries).
NEW_RIGID_ACC = {
    ("RIGID_CORR", 0.1): 0.9930, ("RIGID_CORR", 0.2): 0.9985,
    ("RIGID_CORR", 0.3): 1.0000, ("RIGID_CORR", 0.4): 0.9985,
    ("RIGID_CORR", 0.5): 0.9990, ("RIGID_CORR", 0.6): 1.0000,
    ("RIGID_CORR", 0.7): 1.0000, ("RIGID_CORR", 0.8): 1.0000,
    ("RIGID_CORR", 0.9): 1.0000,
    ("RIGID_WHITE", 0.1): 0.9985, ("RIGID_WHITE", 0.2): 0.9995,
    ("RIGID_WHITE", 0.3): 1.0000, ("RIGID_WHITE", 0.4): 1.0000,
    ("RIGID_WHITE", 0.5): 1.0000, ("RIGID_WHITE", 0.6): 1.0000,
    ("RIGID_WHITE", 0.7): 1.0000, ("RIGID_WHITE", 0.8): 1.0000,
    ("RIGID_WHITE", 0.9): 1.0000,
}
TOL = 1e-6  # must match the regenerated summary exactly (same seeds/deterministic)


def scenario_of(cfg):
    m = re.match(r"([A-Z]+_[A-Z]+)_alpha", cfg)
    return m.group(1) if m else None


def alpha_of(cfg):
    m = re.search(r"alpha([0-9.]+?)\.npz", cfg)
    return float(m.group(1)) if m else None


def is_valid(cfg, acc):
    sc, a = scenario_of(cfg), alpha_of(cfg)
    if sc is None or a is None:
        return False
    if sc.startswith("RIGID"):
        ref = NEW_RIGID_ACC.get((sc, a))
        # RIGID results only count if they match the regenerated generator.
        return ref is not None and abs(acc - ref) < TOL
    return True


METHODS = [
    ("patternlocal",    "PatternLocal",    "#ff7f0e", "s", "-"),
    ("lime",            "LIME",            "#1f77b4", "o", "-"),
    ("analytical_ig",   "Analytical-IG",   "#d62728", "D", "-"),
    ("analytical_grad", "Analytical-grad", "#2ca02c", "^", "-"),
    ("plain_grad",      "Saliency",        "#9467bd", "v", "--"),
    ("plain_ig",        "Plain IG",        "#8c564b", "P", "--"),
    ("deeplift",        "DeepLift",        "#e377c2", "X", "--"),
    ("gradient_shap",   "GradientShap",    "#7f7f7f", "*", "--"),
    ("guided_backprop", "GuidedBackprop",  "#bcbd22", "1", "--"),
    ("sobel",           "Sobel",           "#17becf", "+", ":"),
    ("laplace",         "Laplace",         "#000000", "x", ":"),
]

SCENARIOS = [
    ("LINEAR WHITE", "LIN_WHITE"), ("LINEAR CORR", "LIN_CORR"),
    ("XOR WHITE", "XOR_WHITE"), ("XOR CORR", "XOR_CORR"),
    ("RIGID WHITE", "RIGID_WHITE"), ("RIGID CORR", "RIGID_CORR"),
]


def load():
    rows = {}
    for f in glob.glob(str(RESULTS / "patternlocal-repro-*-sweep-*"
                           / "sweep_part*_results.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for _, r in data.items():
            if not (isinstance(r, dict) and "config" in r):
                continue
            if r.get("skipped") or "lime" not in r:
                continue
            if is_valid(r["config"], r.get("test_accuracy", 0.0)):
                rows[r["config"]] = r
    return rows


def plot(rows, metric, out_path):
    grouped = {k: {} for _, k in SCENARIOS}
    for cfg, r in rows.items():
        sc, a = scenario_of(cfg), alpha_of(cfg)
        if sc in grouped:
            grouped[sc][a] = r

    panels = [(t, k) for t, k in SCENARIOS if grouped[k]]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                             squeeze=False)

    for idx, (title, key) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        pts = grouped[key]
        alphas = sorted(pts)
        for mkey, label, color, marker, ls in METHODS:
            ys, es = [], []
            for a in alphas:
                d = pts[a].get(mkey, {})
                ys.append(d.get(f"{metric}_mean", np.nan))
                n = pts[a].get("n_eval", 1) or 1
                es.append(d.get(f"{metric}_std", 0.0) / np.sqrt(n))
            if np.all(np.isnan(ys)):
                continue
            lw = 3.0 if mkey == "patternlocal" else 1.5
            ms = 9 if mkey == "patternlocal" else 6
            ax.errorbar(alphas, ys, yerr=es, label=label, color=color,
                        marker=marker, linestyle=ls, linewidth=lw,
                        markersize=ms, capsize=3, alpha=0.95,
                        zorder=4 if mkey == "patternlocal" else 2)
        n_present = len(alphas)
        ax.set_title(f"{title}   [{n_present}/9 alphas]",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("alpha (SNR)", fontsize=10)
        ax.set_xlim(0.05, 0.95)
        ax.set_xticks([round(0.1 * i, 1) for i in range(1, 10)])
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel(metric.upper(), fontsize=12, fontweight="bold")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{metric.upper()} vs SNR -- LOCKED results so far "
                 f"(band = standard error)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    rows = load()
    print(f"valid configs: {len(rows)}")
    for cfg in sorted(rows):
        print("   ", cfg, "acc=%.4f" % rows[cfg]["test_accuracy"])
    plot(rows, "emd", OUT / "locked_emd.png")
    plot(rows, "ime", OUT / "locked_ime.png")


if __name__ == "__main__":
    main()
