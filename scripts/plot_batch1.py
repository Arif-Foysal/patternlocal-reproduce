"""Plot the batch-1 snapshot (6 completed sweeps) as EMD and IME bar/point charts.

With only 1-2 alphas per scenario a line "curve" is not yet meaningful, so this
renders grouped bars per scenario (one bar per method), which is the honest
representation of what we currently have.  It also drops the merged JSONs into
results/<kernel>/ so the existing plot_sweep_results.py can pick them up later.
"""

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/tmp/b1")
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# method -> (label, color)
METHODS = [
    ("lime",             "LIME",             "#1f77b4"),
    ("patternlocal",     "PatternLocal",     "#ff7f0e"),
    ("analytical_ig",    "Analytical-IG",    "#d62728"),
    ("analytical_grad",  "Analytical-grad",  "#2ca02c"),
    ("plain_grad",       "Saliency/Grad",    "#9467bd"),
    ("plain_ig",         "Plain IG",         "#8c564b"),
    ("deeplift",         "DeepLift",         "#e377c2"),
    ("gradient_shap",    "GradientShap",     "#7f7f7f"),
    ("guided_backprop",  "GuidedBackprop",   "#bcbd22"),
    ("sobel",            "Sobel",            "#17becf"),
    ("laplace",          "Laplace",          "#aec7e8"),
]

LABELS = {k: lab for k, lab, _ in METHODS}
COLORS = {k: col for k, _, col in METHODS}
ORDER = [k for k, _, _ in METHODS]


def load():
    rows = {}
    for f in glob.glob(f"{SRC}/*/sweep_part*_results.json"):
        for _, r in json.load(open(f)).items():
            rows[r["config"]] = r
    return rows


def plot(rows, metric, out_path):
    items = sorted(rows.items(),
                   key=lambda kv: kv[1]["alpha"])
    n = len(items)
    fig, ax = plt.subplots(figsize=(max(12, n * 2.0), 6))
    x = np.arange(n)
    width = 0.8 / len(ORDER)

    for i, m in enumerate(ORDER):
        vals, errs = [], []
        for _, r in items:
            d = r.get(m, {})
            vals.append(d.get(f"{metric}_mean", np.nan))
            errs.append(d.get(f"{metric}_std", 0.0))
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      yerr=errs, capsize=2, label=LABELS[m],
                      color=COLORS[m], edgecolor="black", linewidth=0.3)
        # mark the proposed method
        if m == "patternlocal":
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.4)

    titles = []
    for _, r in items:
        cfg = r["config"].replace(".npz", "")
        titles.append(f"{cfg}\nacc={r['test_accuracy']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=9)
    ax.set_ylabel(metric.upper(), fontsize=13, fontweight="bold")
    ax.set_title(f"{metric.upper()} by scenario — batch 1 snapshot "
                 f"({n} configs, post-refactor core)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def mirror_for_existing_plotter():
    """Copy fetched JSONs into results/<kernel>/ so plot_sweep_results.py finds them."""
    for d in glob.glob(f"{SRC}/*/"):
        name = Path(d).name
        dst = RESULTS / f"patternlocal-repro-{name}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(f"{d}*.json") + glob.glob(f"{d}*.log"):
            shutil.copy2(f, dst)
    print("mirrored fetched results into results/")


def main():
    rows = load()
    print(f"loaded {len(rows)} configs")
    mirror_for_existing_plotter()
    plot(rows, "emd", OUT / "batch1_emd.png")
    plot(rows, "ime", OUT / "batch1_ime.png")


if __name__ == "__main__":
    main()
