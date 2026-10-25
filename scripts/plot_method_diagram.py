#!/usr/bin/env python3
r"""Conceptual methodological diagram for PatternGrad (Figure 1).

A 1x4 horizontal pipeline, fully programmatic (no generated imagery, no
raster stitching): every visual element is drawn in code, panels A and D use
the real exported attribution data, so the figure is deterministic and
reproducible from this repo.

  (a) Surrogate  : baseline x' -> instance x* -> exact Integrated-Gradients
                   attribution w (real data: qual_LIN_WHITE_0.2.npz, sample 0)
  (b) Neighborhood: stylized 2D manifold scatter; target x* with its K nearest
                   training neighbors encircled; kernel weights fade with
                   distance (median-adaptive Gaussian bandwidth)
  (c) Transform  : closed-form, matrix-free kernel-weighted covariance
                   a = C_K w  (k x k work, no O(D^3) solve)
  (d) Output     : the real PatternGrad attribution a for the same instance
                   (crisp shape, zero background leakage)

Outputs: results/figures/fig1_method_diagram.pdf / .png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
})

HERE = os.path.dirname(__file__)
BUNDLE = os.path.join(HERE, "..", "results",
                      "patternlocal-repro-figure-data-export", "figure_bundle")
OUT = os.path.join(HERE, "..", "results", "figures")
SAMPLE = 0  # LIN WHITE alpha=0.2, sample 0 (y=0: crisp T, black background)


def max_abs_scale(arr):
    denom = np.max(np.abs(arr))
    return arr / denom if denom > 1e-12 else arr


def signed_sqrt_display(w):
    """Sign-preserving display stretch: spreads dim values, keeps zero.

    Purely cosmetic (rendering only); the exported data stays raw.
    """
    return np.sign(w) * np.sqrt(np.abs(w))


def load_real_arrays():
    z = np.load(os.path.join(BUNDLE, "qual_LIN_WHITE_0.2.npz"))
    x_inst = max_abs_scale(z["X"][SAMPLE])
    # signed-sqrt display stretch so the dim IG surrogate tile stays legible
    w_ig = signed_sqrt_display(max_abs_scale(
        np.asarray(z["plain_ig"][SAMPLE], dtype=np.float64)))
    a_out = max_abs_scale(np.asarray(z["analytical_ig"][SAMPLE],
                                     dtype=np.float64))
    return x_inst, w_ig, a_out


# ---------------------------------------------------------------------------
# Panel (a): baseline -> instance -> IG surrogate
# ---------------------------------------------------------------------------
def panel_a(fig, rect, x_inst, w_ig):
    x0, y0, w, h = rect
    tile_w = w / 3.0
    tiles = [
        (np.zeros_like(x_inst), r"$x'$", "baseline", "data"),
        (x_inst, r"$x_*$", "instance", "data"),
        (w_ig, r"$w$", "IG surrogate", "attr"),
    ]
    for i, (arr, sym, sub, kind) in enumerate(tiles):
        ax = fig.add_axes([x0 + i * tile_w, y0, tile_w, h])
        cmap = "coolwarm" if kind == "data" else CMAP_ATTR
        ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        ax.set_xlabel(f"{sym}\n{sub}", fontsize=7.5, labelpad=2)
        if i < 2:  # small arrow between the mini tiles
            add_arrow(fig, (x0 + (i + 1) * tile_w - 0.012, y0 + h / 2),
                      (x0 + (i + 1) * tile_w + 0.012, y0 + h / 2), lw=1.0)


# ---------------------------------------------------------------------------
# Panel (b): empirical neighborhood with kernel weights
# ---------------------------------------------------------------------------
def panel_b(fig, rect):
    rng = np.random.default_rng(7)
    ax = fig.add_axes(rect)
    ax.set_xticks([]), ax.set_yticks([])

    # stylized manifold: two lobes + sparse background
    n_lobes, n_bg, k = 140, 60, 14
    lobe1 = rng.normal([0.45, 0.45], 0.16, (n_lobes, 2))
    lobe2 = rng.normal([0.62, 0.62], 0.13, (n_lobes // 2, 2))
    bg = rng.uniform(0.02, 0.98, (n_bg, 2))
    pts = np.vstack([lobe1, lobe2, bg])
    x_star = np.array([0.5, 0.5])

    d = np.linalg.norm(pts - x_star, axis=1)
    order = np.argsort(d)
    idx_k = order[:k]
    dists_k = d[idx_k]
    bw = float(np.median(dists_k))                     # median-adaptive bw
    weights = np.exp(-(dists_k ** 2) / (2 * bw ** 2))  # Gaussian kernel
    weights = weights / weights.max()

    # background points (light), neighbors colored by kernel weight
    bg_idx = np.setdiff1d(np.arange(len(pts)), idx_k)
    ax.scatter(pts[bg_idx, 0], pts[bg_idx, 1], s=8, color="#c8ccd4",
               alpha=0.55, linewidths=0)
    sc = ax.scatter(pts[idx_k, 0], pts[idx_k, 1], s=14 + 70 * weights,
                    c=weights, cmap="YlOrRd", vmin=0, vmax=1,
                    edgecolors="#333333", linewidths=0.5, zorder=5)
    ax.scatter(*x_star, marker="*", s=210, color="#1a1a2e",
               edgecolors="white", linewidths=0.7, zorder=6)
    ax.add_patch(Circle(x_star, dists_k[-1] * 1.25, fill=False,
                        ls="--", lw=1.0, ec="#555555", zorder=4))

    ax.text(x_star[0] + 0.04, x_star[1] - 0.055, r"$x_*$", fontsize=10)
    ax.text(0.03, 0.955,
            r"$K$ nearest, $v_i = \mathrm{kernel}(d_i / \tilde d)$",
            fontsize=7.5, va="top")
    ax.text(0.03, 0.045, r"$\tilde d$ = median kNN distance",
            fontsize=7.5, va="bottom")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# Panel (c): matrix-free closed-form covariance transform
# ---------------------------------------------------------------------------
def panel_c(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")

    # equation (top zone)
    ax.text(0.5, 0.90, r"$a \;=\; C_K\, w$",
            ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.73,
            r"$C_K = \sum_{i \in N_K} v_i\,"
            r"(x_i - \bar x_K)(x_i - \bar x_K)^{\!\top}$",
            ha="center", va="center", fontsize=8.5)

    # schematic (middle zone): w -> [ C_K ] -> a  (C_K never formed)
    ay = 0.46
    ax.add_patch(FancyArrowPatch((0.14, ay), (0.36, ay),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.2, color="#333333"))
    ax.add_patch(FancyArrowPatch((0.64, ay), (0.86, ay),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.2, color="#333333"))
    ax.text(0.07, ay, r"$w$", ha="center", va="center", fontsize=12)
    ax.text(0.50, ay + 0.105, r"$C_K$", ha="center", va="center", fontsize=11)
    ax.add_patch(plt.Rectangle((0.37, ay - 0.07), 0.26, 0.14,
                               fill=False, lw=1.2, ec="#c0392b"))
    ax.text(0.50, ay, "never\nformed", ha="center", va="center",
            fontsize=6.8, color="#c0392b", linespacing=1.1)
    ax.text(0.93, ay, r"$a$", ha="center", va="center", fontsize=12)

    # notes (bottom zone)
    notes = [
        (0.26, "matrix-free: never forms $D \\times D$"),
        (0.165, r"$k \times k$ work, $k \ll D$: no $O(D^3)$ solve"),
        (0.07, "single pass, closed form, deterministic"),
    ]
    for yy, txt in notes:
        ax.text(0.5, yy, txt, ha="center", va="center", fontsize=7.8,
                color="#333333")


# ---------------------------------------------------------------------------
# Panel (d): real PatternGrad output
# ---------------------------------------------------------------------------
def panel_d(fig, rect, a_out):
    ax = fig.add_axes(rect)
    ax.imshow(a_out, cmap=CMAP_ATTR, vmin=-1.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xlabel(r"$a$" + "\nPatternGrad output", fontsize=7.5, labelpad=2)


# ---------------------------------------------------------------------------
# arrows between panels + step annotations
# ---------------------------------------------------------------------------
def add_arrow(fig, start, end, lw=2.2):
    ar = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                         lw=lw, color="#222222")
    fig.add_artist(ar)


def main():
    global CMAP_ATTR
    import seaborn as sns
    CMAP_ATTR = sns.color_palette("icefire", as_cmap=True)

    x_inst, w_ig, a_out = load_real_arrays()
    os.makedirs(OUT, exist_ok=True)

    fig = plt.figure(figsize=(12.5, 3.1))

    # 4 panel rects (figure coords), arrow gaps between
    pw, gap = 0.215, 0.048
    xs = [0.012, 0.012 + (pw + gap), 0.012 + 2 * (pw + gap),
          0.012 + 3 * (pw + gap)]
    y0, h = 0.40, 0.50

    panel_a(fig, [xs[0], y0, pw, h], x_inst, w_ig)
    panel_b(fig, [xs[1], y0, pw, h])
    panel_c(fig, [xs[2], y0, pw, h])
    panel_d(fig, [xs[3], y0, pw, h], a_out)

    # arrows + step numbers between panels
    steps = [
        "1  exact IG surrogate\n      (no perturbation,\n      no surrogate fit)",
        "2  k-NN neighborhood\n      median-adaptive\n      Gaussian bandwidth",
        "3  matrix-free kernel-\n      weighted covariance\n      (closed form)",
        "4  deterministic pattern\n      attribution $a$",
    ]
    titles = ["(a) Surrogate", "(b) Neighborhood",
              "(c) Transform", "(d) PatternGrad"]
    for i in range(3):
        ax_start = (xs[i] + pw + 0.004, y0 + h / 2)
        ax_end = (xs[i + 1] - 0.004, y0 + h / 2)
        add_arrow(fig, ax_start, ax_end)
        mid = (xs[i] + pw + xs[i + 1]) / 2
        fig.text(mid, y0 + h / 2 + 0.075, f"step {i + 1}",
                 ha="center", fontsize=7.5, color="#555555")

    # panel titles + step captions
    for i, (x, step, title) in enumerate(zip(xs, steps, titles)):
        fig.text(x + pw / 2, y0 + h + 0.055, title,
                 ha="center", fontsize=9.5, fontweight="bold")
        fig.text(x + pw / 2, y0 - 0.155, step,
                 ha="center", va="top", fontsize=7.6, linespacing=1.35)

    for ext, fmt in [("pdf", "pdf"), ("png", "png")]:
        out = os.path.join(OUT, f"fig1_method_diagram.{ext}")
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=300)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
