#!/usr/bin/env python3
"""Qualitative attribution grid (guide Script 2).

Rows: 6 benchmark rows at the guide's signal-to-noise levels.
Columns: Instance, Ground Truth, then attribution methods. Data columns use
coolwarm; attribution columns use seaborn icefire with a shared colorbar.

Data source: figure_data_export kernel output
  results/patternlocal-repro-figure-data-export/figure_bundle/qual_<ROW>.npz
Keys per file: X, mask (raw), y, tags, plus one (5, 64, 64) array per method.

Variants:
  full     8 columns: PatternGrad, PatternLocal, Analytical-Grad (No Path),
           Integrated Gradients, SmoothGrad, LIME  (guide's recommended set)
  compact  6 columns: PatternGrad, PatternLocal, Analytical-Grad (No Path), LIME
  wide    10 columns: full + Sobel + Laplace (paper filter baselines)
  all      render all three

--use-legacy falls back to the per-sweep alpha*_sample_images.npz files
(lime, patternlocal, analytical_ig only) for a quick preview before the
bundle is fetched.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

DEFAULT_BUNDLE = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "patternlocal-repro-figure-data-export", "figure_bundle")

ROWS = [
    ("LIN_WHITE_0.2",   "LIN (0.2)\nWHITE"),
    ("LIN_CORR_0.2",    "LIN (0.2)\nCORR"),
    ("XOR_WHITE_0.2",   "XOR (0.2)\nWHITE"),
    ("XOR_CORR_0.2",    "XOR (0.2)\nCORR"),
    ("RIGID_WHITE_0.7", "RIGID (0.7)\nWHITE"),
    ("RIGID_CORR_0.3",  "RIGID (0.3)\nCORR"),
]

VARIANTS = {
    "final": [
        ("analytical_ig", "PatternGrad\n(Ours)"),
        ("patternlocal", "PatternLocal"),
        ("plain_ig", "Integrated\nGradients"),
        ("plain_smoothgrad", "SmoothGrad"),
        ("lime", "LIME"),
    ],
    "full": [
        ("analytical_ig", "PatternGrad\n(Ours)"),
        ("patternlocal", "PatternLocal"),
        ("analytical_grad", "Analytical-Grad\n(No Path)"),
        ("plain_ig", "Integrated\nGradients"),
        ("plain_smoothgrad", "SmoothGrad"),
        ("lime", "LIME"),
    ],
    "compact": [
        ("analytical_ig", "PatternGrad\n(Ours)"),
        ("patternlocal", "PatternLocal"),
        ("analytical_grad", "Analytical-Grad\n(No Path)"),
        ("lime", "LIME"),
    ],
    "wide": [
        ("analytical_ig", "PatternGrad\n(Ours)"),
        ("patternlocal", "PatternLocal"),
        ("analytical_grad", "Analytical-Grad\n(No Path)"),
        ("plain_ig", "Integrated\nGradients"),
        ("plain_smoothgrad", "SmoothGrad"),
        ("lime", "LIME"),
        ("sobel", "Sobel"),
        ("laplace", "Laplace"),
    ],
}


def max_abs_scale(arr):
    denom = np.max(np.abs(arr))
    return arr / denom if denom > 1e-12 else arr


def _load_row(bundle, row_key, sample, use_legacy):
    if use_legacy:
        slug = row_key.lower()  # e.g. lin_white_0.2
        scen, a = row_key.rsplit("_", 1)[0], row_key.rsplit("_", 1)[1]
        sweep_n = a.split(".")[1]  # alpha 0.2 lives in sweep-2
        path = os.path.join(os.path.dirname(__file__), "..", "results",
                            f"patternlocal-repro-{scen.lower().replace('_', '-')}"
                            f"-sweep-{sweep_n}",
                            f"alpha{float(a):.1f}_sample_images.npz")
        if not os.path.exists(path):
            sys.exit(f"legacy npz not found: {path}")
        z = np.load(path)
        p = min(sample, len(z["X"]) - 1)
        return {
            "X": z["X"][p], "mask": z["mask"][p],
            "lime": z["lime"][p], "patternlocal": z["patternlocal"][p],
            "analytical_ig": z["analytical_ig"][p],
        }
    path = os.path.join(bundle, f"qual_{row_key}.npz")
    if not os.path.exists(path):
        sys.exit(f"{path} not found; fetch the figure-data-export output first")
    z = np.load(path)
    p = min(sample, len(z["X"]) - 1)
    return {k: z[k][p] for k in z.files if k != "tags"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                                                     "..", "results", "figures"))
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    ap.add_argument("--sample", type=int, default=0,
                    help="which of the 5 saved eval samples per row")
    ap.add_argument("--per-row-sample", default=None,
                    help="comma list of sample indices, one per row "
                         "(e.g. '0,0,1,1,0,0' shows y=1 for both XOR rows)")
    ap.add_argument("--use-legacy", action="store_true",
                    help="render from per-sweep sample npz (3 methods only)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cmap_data = plt.get_cmap("coolwarm")
    cmap_attr = sns.color_palette("icefire", as_cmap=True)

    to_render = list(VARIANTS) if args.variant == "all" else [args.variant]

    for variant in to_render:
        method_cols = VARIANTS[variant]
        if args.use_legacy:
            method_cols = [(mk, lbl) for mk, lbl in method_cols
                           if mk in ("lime", "patternlocal", "analytical_ig")]
        all_cols = ["Instance", "Ground\nTruth"] + [lbl for _, lbl in method_cols]
        n_rows, n_cols = len(ROWS), len(all_cols)

        fig = plt.figure(figsize=(1.15 * n_cols + 0.5, 1.15 * n_rows))
        gs = gridspec.GridSpec(
            n_rows, n_cols + 1,
            width_ratios=[1] * n_cols + [0.12],
            wspace=0.05,
            hspace=0.05,
        )

        attr_im_handle = None
        per_row = None
        if args.per_row_sample:
            per_row = [int(v) for v in args.per_row_sample.split(",")]
            if len(per_row) != len(ROWS):
                sys.exit(f"--per-row-sample needs {len(ROWS)} comma-separated values")
        for r_idx, (row_key, row_label) in enumerate(ROWS):
            row = _load_row(args.bundle, row_key,
                            per_row[r_idx] if per_row else args.sample,
                            args.use_legacy)
            # Ground Truth column: signed continuous signal when available,
            # else the binary mask (legacy preview npz lack the signal).
            gt_key = "signal" if "signal" in row else "mask"
            cols = [("X", "data"), (gt_key, "data")] + \
                   [(mk, "attr") for mk, _ in method_cols]
            for c_idx, (arr_key, kind) in enumerate(cols):
                ax = fig.add_subplot(gs[r_idx, c_idx])
                arr = max_abs_scale(np.asarray(row[arr_key], dtype=np.float64))
                if kind == "data":
                    ax.imshow(arr, cmap=cmap_data, vmin=-1.0, vmax=1.0,
                              interpolation="nearest")
                else:
                    attr_im_handle = ax.imshow(arr, cmap=cmap_attr,
                                               vmin=-1.0, vmax=1.0,
                                               interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_edgecolor("#444444")

                if r_idx == 0:
                    label = all_cols[c_idx]
                    is_bold = "PatternGrad" in label
                    ax.set_title(label, fontsize=8.5,
                                 fontweight="bold" if is_bold else "normal",
                                 pad=5)
                if c_idx == 0:
                    ax.set_ylabel(row_label, fontsize=8, rotation=90,
                                  labelpad=6, va="center", ha="center")

        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(attr_im_handle, cax=cbar_ax,
                            ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.set_label("Feature Importance", fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)

        suffix = ""
        if per_row:
            suffix = "_mixed" if per_row != [args.sample] * len(ROWS) else ""
        out = os.path.join(args.outdir, f"fig_qualitative_{variant}{suffix}.pdf")
        plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
