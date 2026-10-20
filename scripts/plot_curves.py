#!/usr/bin/env python3
"""Plot EMD / IME vs SNR curves for the XAI-TRIS reproduction.

Produces paper-style Figure-4 analogues: one panel per scenario, one line per
method, x = alpha (SNR), y = metric.

Key choices, to match the paper's presentation:
  * Error bands show the **standard error of the mean** (std / sqrt(n)) by
    default.  The paper's Figure 8 caption says "shaded regions indicate
    standard deviation as the standard error is too small to be visible" --
    i.e. their bands are visually tiny.  Plotting raw std over ~570 samples
    makes our bands look ~24x wider than the paper's for the same data, so
    std is available via --band std but SE is the default.
  * Only results fetched into results/patternlocal-repro-<scenario>-sweep-*
    are read (stale pre-refactor dirs live in results/_stale/ and are skipped).

Usage:
    python3 scripts/plot_curves.py                 # EMD + IME, SE bands
    python3 scripts/plot_curves.py --band std      # raw std bands
    python3 scripts/plot_curves.py --metric emd
"""

import argparse
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

# (key, label, color, marker, linestyle)
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

# scenario key -> (panel title, substring matched against the config name)
SCENARIOS = [
    ("LINEAR WHITE", "LIN_WHITE"),
    ("LINEAR CORR",  "LIN_CORR"),
    ("XOR WHITE",    "XOR_WHITE"),
    ("XOR CORR",     "XOR_CORR"),
    ("RIGID WHITE",  "RIGID_WHITE"),
    ("RIGID CORR",   "RIGID_CORR"),
]

# The paper's own Figure 4 axis limits (extracted from the PDF text layer), used
# to make our panels directly comparable.  None => autoscale.
PAPER_EMD_YLIM = {
    "LIN_WHITE": (0.23, 0.30), "LIN_CORR": (0.22, 0.29),
    "XOR_WHITE": (0.01, 0.19), "XOR_CORR": (0.03, 0.15),
    "RIGID_WHITE": (0.06, 0.31), "RIGID_CORR": (0.05, 0.30),
}


def load_rows():
    """Return {config: result_dict} from results/patternlocal-repro-*-sweep-*."""
    rows = {}
    pattern = str(RESULTS / "patternlocal-repro-*-sweep-*"
                  / "sweep_part*_results.json")
    for f in glob.glob(pattern):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for _, r in data.items():
            if isinstance(r, dict) and "config" in r:
                rows[r["config"]] = r
    return rows


def _alpha_of(cfg):
    m = re.search(r"alpha([0-9.]+?)\.npz", cfg)
    return float(m.group(1)) if m else None


def group_by_scenario(rows):
    """{scenario_key: {alpha: result}}."""
    out = {k: {} for _, k in SCENARIOS}
    for cfg, r in rows.items():
        a = _alpha_of(cfg)
        if a is None:
            continue
        for _, key in SCENARIOS:
            if cfg.startswith(key + "_"):
                out[key][a] = r
    return out


def plot(rows, metric, band_kind, out_path, use_paper_ylim):
    grouped = group_by_scenario(rows)
    panels = [(title, key) for title, key in SCENARIOS if grouped[key]]
    if not panels:
        print("no results to plot")
        return
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows),
                             squeeze=False)

    for idx, (title, key) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        pts = grouped[key]
        alphas = sorted(pts)
        for mkey, label, color, marker, ls in METHODS:
            means, errs = [], []
            for a in alphas:
                d = pts[a].get(mkey, {})
                means.append(d.get(f"{metric}_mean", np.nan))
                err = d.get(f"{metric}_std", 0.0)
                n = pts[a].get("n_eval", 1) or 1
                errs.append(err / np.sqrt(n) if band_kind == "se" else err)
            if np.all(np.isnan(means)):
                continue
            lw = 2.6 if mkey == "patternlocal" else 1.5
            ms = 8 if mkey == "patternlocal" else 6
            ax.errorbar(alphas, means, yerr=errs, label=label, color=color,
                        marker=marker, linestyle=ls, linewidth=lw,
                        markersize=ms, capsize=3, alpha=0.9,
                        zorder=4 if mkey == "patternlocal" else 2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("alpha (SNR)", fontsize=11)
        ax.set_xticks(sorted({a for a in alphas}))
        ax.grid(True, alpha=0.3)
        if use_paper_ylim and metric == "emd" and key in PAPER_EMD_YLIM:
            lo, hi = PAPER_EMD_YLIM[key]
            vals = [pts[a][mk]["emd_mean"] for a in alphas
                    for mk, *_ in METHODS
                    if isinstance(pts[a].get(mk), dict)
                    and "emd_mean" in pts[a][mk]]
            if vals:
                ax.set_ylim(min(lo, float(np.nanmin(vals)) * 0.9), hi)
            else:
                ax.set_ylim(lo, hi)

    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel(metric.upper(), fontsize=12, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.03))
    band_txt = "standard error" if band_kind == "se" else "standard deviation"
    fig.suptitle(f"{metric.upper()} vs SNR -- band = {band_txt}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["se", "std"], default="se")
    ap.add_argument("--metric", choices=["emd", "ime", "both"], default="both")
    ap.add_argument("--paper-ylim", action="store_true",
                    help="clamp EMD panels to the paper's axis ranges")
    args = ap.parse_args()

    rows = load_rows()
    print(f"loaded {len(rows)} config(s)")
    metrics = ["emd", "ime"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        plot(rows, metric, args.band, OUT / f"curves_{metric}_{args.band}.png",
             args.paper_ylim)


if __name__ == "__main__":
    main()
