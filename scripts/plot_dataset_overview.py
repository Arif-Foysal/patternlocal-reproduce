#!/usr/bin/env python3
"""Dataset overview figure (guide Script 1).

Renders the XAI-TRIS dataset panels: for each scenario (LIN, XOR, RIGID) the
first y=0 and y=1 test instance at alpha=0.2 with its ground-truth mask.

Data source: figure_data_export kernel output
  results/patternlocal-repro-figure-data-export/figure_bundle/dataset_overview.npz

Variants:
  corr      fig3a_corr.pdf      3x4 grid, CORR scenarios only (LaTeX panel a)
  white     fig3b_white.pdf     3x4 grid, WHITE scenarios only (LaTeX panel b)
  combined  fig3_combined.pdf   single figure, all 6 scenario rows
  all       render all three

Subcaptions (a)/(b) are native LaTeX: use \subcaptionbox or subfigure with
the two PDFs side by side.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

DEFAULT_BUNDLE = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "patternlocal-repro-figure-data-export", "figure_bundle")


def max_abs_scale(arr):
    """Scale a 64x64 array to [-1, 1] while preserving zero."""
    denom = np.max(np.abs(arr))
    return arr / denom if denom > 1e-12 else arr


def _load(bundle):
    path = os.path.join(bundle, "dataset_overview.npz")
    if not os.path.exists(path):
        sys.exit(f"dataset_overview.npz not found at {path}; fetch the "
                 "figure-data-export kernel output first")
    return np.load(path, allow_pickle=True)


def _render(rows, data, noise_type, out_path):
    """rows: list of (scenario_key, row_label). One shared colorbar."""
    cols = ["y0_mask", "y0_x", "y1_mask", "y1_x"]
    col_headers = ["MASK", "INST.", "MASK", "INST."]

    fig = plt.figure(figsize=(4.8, 1.05 * len(rows) + 0.35))
    gs = gridspec.GridSpec(
        len(rows), 5,
        width_ratios=[1, 1, 1, 1, 0.16],
        wspace=0.10,
        hspace=0.10,
    )

    im_handle = None
    axes_grid = np.empty((len(rows), 4), dtype=object)

    for r_idx, (scen_key, row_label) in enumerate(rows):
        for c_idx, col_key in enumerate(cols):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            axes_grid[r_idx, c_idx] = ax

            img = max_abs_scale(data[scen_key][col_key].item()
                                if isinstance(data[scen_key], np.ndarray)
                                and data[scen_key].dtype == object
                                else data[scen_key][col_key])
            im_handle = ax.imshow(img, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                                  interpolation="nearest")

            ax.set_xticks([])
            ax.set_yticks([])

            if r_idx == 0:
                ax.set_title(col_headers[c_idx], fontsize=8.5, pad=4)

            if c_idx == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=8.5,
                    rotation=0,
                    labelpad=26,
                    va="center",
                    ha="center",
                )

    pos_c0 = axes_grid[0, 0].get_position()
    pos_c1 = axes_grid[0, 1].get_position()
    pos_c2 = axes_grid[0, 2].get_position()
    pos_c3 = axes_grid[0, 3].get_position()

    y_top = pos_c0.y1 + 0.08
    fig.text((pos_c0.x0 + pos_c1.x1) / 2, y_top, r"$y = 0$",
             ha="center", va="bottom", fontsize=11)
    fig.text((pos_c2.x0 + pos_c3.x1) / 2, y_top, r"$y = 1$",
             ha="center", va="bottom", fontsize=11)

    cbar_ax = fig.add_subplot(gs[:, 4])
    cbar = fig.colorbar(im_handle, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                                                     "..", "results", "figures"))
    ap.add_argument("--variant", choices=["corr", "white", "combined", "all"],
                    default="all")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = _load(args.bundle)

    corr_rows = [
        ("LIN_CORR", "LIN\nCORR"),
        ("XOR_CORR", "XOR\nCORR"),
        ("RIGID_CORR", "RIGID\nCORR"),
    ]
    white_rows = [
        ("LIN_WHITE", "LIN\nWHITE"),
        ("XOR_WHITE", "XOR\nWHITE"),
        ("RIGID_WHITE", "RIGID\nWHITE"),
    ]
    combined_rows = corr_rows + white_rows

    if args.variant in ("corr", "all"):
        _render(corr_rows, data, "CORR",
                os.path.join(args.outdir, "fig3a_corr.pdf"))
    if args.variant in ("white", "all"):
        _render(white_rows, data, "WHITE",
                os.path.join(args.outdir, "fig3b_white.pdf"))
    if args.variant in ("combined", "all"):
        _render(combined_rows, data, "ALL",
                os.path.join(args.outdir, "fig3_combined.pdf"))


if __name__ == "__main__":
    main()
