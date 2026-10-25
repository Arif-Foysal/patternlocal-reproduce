#!/usr/bin/env python3
r"""Quantitative SNR sweep curves: 2x3 grid (WHITE row, CORR row) x
(LINEAR, XOR, RIGID), one panel per metric, in the PatternLocal paper's
reference style.

Data source: the fetched sweep results
  results/patternlocal-repro-<scen>-sweep-<n>/sweep_part<n>_results.json
(one file per sweep part; methods at top level with
emd_mean / emd_std / ime_mean / ime_std).

Panels plot only the alphas that exist for that scenario (gated points are
absent: RIGID CORR has 0.2-0.9, RIGID WHITE has 0.7-0.9, all others 0.1-0.9).

Usage:
  .venv/bin/python scripts/plot_quantitative_curves.py                # both metrics
  .venv/bin/python scripts/plot_quantitative_curves.py --band-scale 1.0
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
})

# method key -> (Display Label, Color, Marker, LineWidth, ZOrder)
METHOD_STYLES = [
    ("analytical_ig",    "PatternGrad (Ours)",  "#c0392b", "D", 1.8, 10),
    ("patternlocal",     "PatternLocal",        "#d98b5f", "o", 1.4, 9),
    ("lime",             "LIME",                "#7b52ab", "s", 1.1, 6),
    ("sobel",            "Sobel",               "#2b6cb0", "x", 1.1, 7),
    ("laplace",          "Laplace",             "#7fb3d5", "v", 1.1, 7),
    ("plain_ig",         "IntegratedGradients", "#2d6a4f", "^", 1.1, 5),
    ("plain_grad",       "Saliency",            "#40916c", ">", 1.1, 5),
    ("plain_smoothgrad", "SmoothGrad",          "#52b788", "<", 1.1, 5),
    ("guided_backprop",  "GuidedBackprop",      "#74c69d", "2", 1.1, 4),
    ("gradient_shap",    "GradientShap",        "#95d5b2", "1", 1.1, 4),
]

SCENARIO_GRID = [
    # Row 0: WHITE
    [("lin_white", "LINEAR", "WHITE"),
     ("xor_white", "XOR",    None),
     ("rigid_white", "RIGID", None)],
    # Row 1: CORR
    [("lin_corr",  None,     "CORR"),
     ("xor_corr",  None,     None),
     ("rigid_corr", None,    None)],
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "figures")


def load_scenario_series(results_dir, scenario_key, metric_name="emd"):
    """method key -> (alphas, means, stds) from sweep_part*_results.json."""
    pattern = os.path.join(results_dir,
                           f"patternlocal-repro-{scenario_key.replace('_', '-')}-sweep-*",
                           "sweep_part*_results.json")
    series = {}
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            payload = json.load(f)
        # Files are keyed "alpha0.2" -> results dict; take the single entry.
        inner = payload[next(iter(payload))] if payload else {}
        if inner.get("skipped") or inner.get("gated_out"):
            continue
        alpha = inner.get("alpha")
        if alpha is None:  # fallback: parse the "alpha0.2" dict key
            m = re.match(r"alpha([\d.]+)", next(iter(payload)))
            alpha = float(m.group(1)) if m else None
        if alpha is None:
            continue
        for m_key, _, _, _, _, _ in METHOD_STYLES:
            m_entry = inner.get(m_key)
            if not isinstance(m_entry, dict):
                continue
            mean_val = m_entry.get(f"{metric_name}_mean")
            std_val = m_entry.get(f"{metric_name}_std", 0.0)
            if mean_val is not None:
                series.setdefault(m_key, []).append(
                    (float(alpha), float(mean_val), float(std_val)))
    out = {}
    for m_key, pts in series.items():
        pts.sort(key=lambda x: x[0])
        arr = np.array(pts)
        out[m_key] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def plot_metric_grid(results_dir, metric_name="emd", band_scale=0.5,
                     out_prefix=None):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 4.2))
    fig.subplots_adjust(wspace=0.22, hspace=0.28, bottom=0.22)

    legend_handles = {}

    for r_idx in range(2):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            scen_key, col_title, row_title = SCENARIO_GRID[r_idx][c_idx]
            data = load_scenario_series(results_dir, scen_key,
                                        metric_name=metric_name)

            all_alphas = set()
            ymin, ymax = np.inf, -np.inf

            for m_key, label, color, marker, lw, zorder in METHOD_STYLES:
                if m_key not in data:
                    continue
                alphas, means, stds = data[m_key]
                all_alphas.update(alphas.tolist())

                line, = ax.plot(
                    alphas, means,
                    label=label, color=color, marker=marker,
                    markersize=5.0, linewidth=lw, zorder=zorder,
                )
                ax.fill_between(
                    alphas,
                    means - band_scale * stds,
                    means + band_scale * stds,
                    color=color, alpha=0.16, linewidth=0,
                    zorder=zorder - 1,
                )

                if label not in legend_handles:
                    legend_handles[label] = line

                ymin = min(ymin, np.min(means - band_scale * stds))
                ymax = max(ymax, np.max(means + band_scale * stds))

            # x ticks from the valid (non-gated) alphas of this panel.
            # RIGID panels span few points: tighter pad so 0.7-0.9 gets
            # xlim 0.68-0.92 with clean ticks at the data points.
            sorted_alphas = sorted(all_alphas)
            if sorted_alphas:
                ax.set_xticks(sorted_alphas)
                pad_x = 0.02 if scen_key.startswith("rigid") else 0.025
                ax.set_xlim(min(sorted_alphas) - pad_x,
                            max(sorted_alphas) + pad_x)

            # y axis: 4 evenly spaced 2-decimal ticks, tight around data
            if np.isfinite(ymin) and np.isfinite(ymax):
                span = max(ymax - ymin, 0.02)
                y_low = max(0.0, ymin - 0.05 * span)
                y_high = ymax + 0.05 * span
                yticks = np.round(np.linspace(y_low, y_high, 4), 2)
                ax.set_ylim(yticks[0], yticks[-1])
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.1)
            ax.spines["bottom"].set_linewidth(1.1)
            ax.tick_params(direction="in", length=3.5, width=1.0,
                           labelsize=8.5)
            ax.grid(True, linestyle=":", linewidth=0.6,
                    color="#bbbbbb", alpha=0.7)

            if col_title:
                ax.set_title(col_title, fontfamily="monospace",
                             fontsize=12, pad=8)
            if row_title:
                ax.set_ylabel(row_title, fontfamily="monospace",
                              fontsize=12, labelpad=8)
            if r_idx == 1:
                ax.set_xlabel(r"$\alpha$ (SNR)", fontsize=9.5, labelpad=4)

    # Shared bottom legend
    handles = [legend_handles[label]
               for _, label, _, _, _, _ in METHOD_STYLES
               if label in legend_handles]
    labels = [label for _, label, _, _, _, _ in METHOD_STYLES
              if label in legend_handles]
    leg = fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.08),
        ncol=5, frameon=True, edgecolor="#cccccc", fontsize=9.0,
        handlelength=2.2, columnspacing=1.5,
    )

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    fig.savefig(f"{out_prefix}.pdf", format="pdf", bbox_inches="tight",
                bbox_extra_artists=[leg], dpi=300)
    fig.savefig(f"{out_prefix}.png", format="png", bbox_inches="tight",
                bbox_extra_artists=[leg], dpi=300)
    plt.close(fig)
    print(f"wrote {out_prefix}.pdf / .png "
          f"({len(labels)} methods, band_scale={band_scale})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--metric", choices=["emd", "ime", "all"], default="all")
    ap.add_argument("--band-scale", type=float, default=0.5,
                    help="std multiplier for the shaded band (0.5 default, "
                         "1.0 for full std)")
    args = ap.parse_args()

    metrics = ["emd", "ime"] if args.metric == "all" else [args.metric]
    for m in metrics:
        plot_metric_grid(args.results_dir, metric_name=m,
                         band_scale=args.band_scale,
                         out_prefix=os.path.join(args.outdir,
                                                 f"fig_curves_{m}"))


if __name__ == "__main__":
    main()
