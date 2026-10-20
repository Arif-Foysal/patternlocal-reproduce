"""Plot batch-1 results as EMD / IME vs alpha line curves.

One column per scenario (LIN WHITE, RIGID WHITE, RIGID CORR, XOR CORR), one
line per method, x = alpha (log-ish spacing as in the paper), y = metric.
Only the alphas we have are drawn; lines show what exists so far.
"""

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/tmp/b1")
OUT = ROOT / "results" / "figures"
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

SCENARIOS = [
    ("LIN WHITE",  "LIN_WHITE"),
    ("RIGID WHITE", "RIGID_WHITE"),
    ("RIGID CORR", "RIGID_CORR"),
    ("XOR WHITE",  "XOR_WHITE"),
    ("XOR CORR",   "XOR_CORR"),
]


def load():
    rows = {}
    for f in glob.glob(f"{SRC}/*/sweep_part*_results.json"):
        for _, r in json.load(open(f)).items():
            rows[r["config"]] = r
    return rows


def extract(rows, prefix, metric):
    """Return (alphas, {method: (means, stds)})."""
    pts = {}
    for cfg, r in rows.items():
        if not cfg.startswith(prefix + "_"):
            continue
        pts[r["alpha"]] = r
    alphas = sorted(pts)
    data = {}
    for key, *_ in METHODS:
        m, s = [], []
        for a in alphas:
            d = pts[a].get(key, {})
            m.append(d.get(f"{metric}_mean", np.nan))
            s.append(d.get(f"{metric}_std", 0.0))
        data[key] = (np.array(m), np.array(s))
    return alphas, data


def plot(rows, metric, out_path):
    panels = [(lab, pre) for lab, pre in SCENARIOS
              if any(c.startswith(pre + "_") for c in rows)]
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows),
                             squeeze=False)

    for idx, (lab, pre) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        alphas, data = extract(rows, pre, metric)
        for key, label, color, marker, ls in METHODS:
            means, stds = data[key]
            if np.all(np.isnan(means)):
                continue
            lw = 2.6 if key == "patternlocal" else 1.6
            ms = 8 if key == "patternlocal" else 6
            ax.errorbar(alphas, means, yerr=stds, label=label, color=color,
                        marker=marker, linestyle=ls, linewidth=lw,
                        markersize=ms, capsize=3, alpha=0.9, zorder=3
                        if key == "patternlocal" else 2)
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.set_xlabel("α (SNR)", fontsize=11)
        ax.set_xticks(sorted({a for a in alphas}))
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    for r in range(nrows):
        axes[r][0].set_ylabel(metric.upper(), fontsize=12, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{metric.upper()} vs SNR — batch 1 (post-refactor core)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    rows = load()
    print(f"loaded {len(rows)} configs")
    plot(rows, "emd", OUT / "batch1_emd_curve.png")
    plot(rows, "ime", OUT / "batch1_ime_curve.png")


if __name__ == "__main__":
    main()
