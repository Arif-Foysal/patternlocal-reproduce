#!/usr/bin/env python3
r"""PatternGrad conceptual diagram (Figure 1, publication style).

Wide left-to-right scientific workflow: Inputs -> (1) deterministic IG
surrogate -> (2) neighborhood retrieval -> (3) adaptive Gaussian locality ->
(4) covariance conversion (visually central) -> explanation comparison.
Real LIN-WHITE alpha=0.2 data (sample 0) wherever a real map exists;
everything else is schematic and drawn in code.

Notation follows sn-article.tex exactly:
  w^PG = s_hat^IG(x*)                       (M-step quadrature, x' = 0)
  N_K(x*)                                   (K nearest, Euclidean metric)
  pi_j = K_sigma(d_j) / sum_l K_sigma(d_l), K_sigma(d) = exp(-(d/sigma)^2/2)
  sigma = median neighbor distance          (no bandwidth hyper-parameter)
  a = C_K w^PG / ((w^PG)^T C_K w^PG + lambda)   (l2-ridge closed form)
  two-pass product X~_K^T (pi o (X~_K w))   (matrix-free, O(KD))

Outputs (repo organization: results/figures/):
  patterngrad_conceptual_diagram.pdf / .svg / .png
LaTeX snippet: results/figures/PATTERNGRAD_FIGURE_SNIPPET.tex
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
})

HERE = os.path.dirname(__file__)
BUNDLE = os.path.join(HERE, "..", "results",
                      "patternlocal-repro-figure-data-export", "figure_bundle")
OUT = os.path.join(HERE, "..", "results", "figures")

W, H = 16.5, 5.6   # canvas in inches (full-width paper figure)
SAMPLE = 0         # LIN WHITE alpha=0.2, sample 0 (y=0)


def R(x, y, w, h):
    """Inch rect -> figure-fraction rect."""
    return [x / W, y / H, w / W, h / H]


def signed_sqrt(w):
    """Sign-preserving display stretch for dim maps (rendering only)."""
    return np.sign(w) * np.sqrt(np.abs(w))


def maxabs(a):
    return a / max(np.max(np.abs(a)), 1e-12)


def load_real():
    z = np.load(os.path.join(BUNDLE, "qual_LIN_WHITE_0.2.npz"))
    x_inst = maxabs(np.asarray(z["X"][SAMPLE], dtype=np.float64))
    w_ig = maxabs(signed_sqrt(np.asarray(z["plain_ig"][SAMPLE],
                                         dtype=np.float64)))
    gt = maxabs(np.asarray(z["signal"][SAMPLE], dtype=np.float64))
    a_out = maxabs(np.asarray(z["analytical_ig"][SAMPLE], dtype=np.float64))
    return x_inst, w_ig, gt, a_out


def label(fig, x, y, text, ha="center", va="top", fontsize=7.5,
          color="#333333", **kw):
    fig.text(x / W, y / H, text, ha=ha, va=va, fontsize=fontsize,
             color=color, **kw)


def heading(fig, cx, text, color="#111111"):
    label(fig, cx, 5.32, text, va="center", fontsize=9.5, color=color,
          fontweight="bold")


def tile(fig, x, y, s, arr, cmap, lw=0.6, ec="#333333"):
    ax = fig.add_axes(R(x, y, s, s))
    ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(lw), sp.set_edgecolor(ec)
    return ax


def note(fig, cx, text, y, fontsize=7.0, color="#555555"):
    label(fig, cx, y, text, fontsize=fontsize, color=color, linespacing=1.3)


def main():
    import seaborn as sns
    cmap_data = "coolwarm"
    cmap_attr = sns.color_palette("icefire", as_cmap=True)
    x_inst, w_ig, gt, a_out = load_real()

    fig = plt.figure(figsize=(W, H))

    # single overlay for every schematic shape; coords = absolute inches
    ov = fig.add_axes([0, 0, 1, 1])
    ov.axis("off")
    ov.set_xlim(0, W), ov.set_ylim(0, H)
    ov.set_zorder(2)

    # stage columns (inch origin, width, center)
    c0, w0 = 0.35, 1.60
    c1, w1 = 2.45, 2.85
    c2, w2 = 5.80, 2.10
    c3, w3 = 8.35, 2.15
    c4, w4 = 10.95, 2.45
    c5, w5 = 13.85, 1.55
    ctr = lambda x, w: x + w / 2

    # ============================ INPUTS ================================
    heading(fig, ctr(c0, w0), "Inputs")
    tile(fig, ctr(c0, w0) - 0.5, 3.85, 1.0, x_inst, cmap_data)
    label(fig, ctr(c0, w0), 3.70,
          r"$\mathbf{x}_\star$" + "\ntarget instance", linespacing=1.25)
    ov.add_patch(Rectangle((ctr(c0, w0) - 0.5, 2.75), 1.0, 0.42,
                           fill=False, lw=1.0, ec="#2b6cb0"))
    label(fig, ctr(c0, w0), 2.96, r"$f$   differentiable classifier",
          va="center", fontsize=7.5)
    for dx, dy in [(0.0, 0.0), (0.09, 0.12), (0.18, 0.24)]:
        ov.add_patch(Rectangle((0.84 + dx, 1.30 + dy), 0.62, 0.62,
                               facecolor="#e8eaee", edgecolor="#8a8f98",
                               lw=0.7, zorder=3 - dy))
    label(fig, ctr(c0, w0), 1.10,
          r"$\mathcal{D}_{\mathrm{train}}$" + "\ntraining samples",
          linespacing=1.25)

    # ==================== STAGE 1: DETERMINISTIC IG =====================
    heading(fig, ctr(c1, w1), "1   Deterministic IG", color="#2d6a4f")
    tile(fig, 2.57, 3.90, 0.85, np.zeros((64, 64)), cmap_data)
    label(fig, 2.995, 3.75, r"$\mathbf{x}' = \mathbf{0}$", fontsize=7.5)
    ov.add_patch(FancyArrowPatch((3.47, 4.325), (3.80, 4.325),
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=1.1, color="#2d6a4f"))
    label(fig, 3.635, 4.47, r"$\gamma$", fontsize=7.5,
          ha="center", va="bottom", color="#2d6a4f")
    tile(fig, 3.85, 3.90, 0.85, x_inst, cmap_data)
    label(fig, 4.275, 3.75, r"$\mathbf{x}_\star$", fontsize=7.5)
    for i, gx in enumerate([2.95, 3.70, 4.45]):
        t = (i + 1) / 4
        tile(fig, gx, 2.70, 0.62, np.clip(t * x_inst, -1, 1), cmap_data,
             lw=0.45, ec="#888888")
        label(fig, gx + 0.31, 2.55, rf"$\gamma = {t:.2f}$", fontsize=6.5)
    ov.add_patch(FancyArrowPatch((3.875, 2.62), (3.875, 2.25),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#555555"))
    tile(fig, 3.375, 1.15, 1.0, w_ig, cmap_attr)
    label(fig, ctr(c1, w1), 1.00,
          r"$\mathbf{w}^{\mathrm{PG}} = \hat{\mathbf{s}}^{\mathrm{IG}}(\mathbf{x}_\star)$"
          "\npixel-level IG attribution", linespacing=1.3)
    note(fig, ctr(c1, w1),
         "deterministic: fixed $(f, \\mathbf{x}', M)$"
         "\nno random masks, no surrogate fit"
         "\n(M-step quadrature error only)", y=0.50)

    # ================ STAGE 2: NEIGHBORHOOD RETRIEVAL ====================
    heading(fig, ctr(c2, w2), "2   Neighborhood", color="#2b6cb0")
    label(fig, ctr(c2, w2), 4.95,
          r"$\mathcal{N}_K(\mathbf{x}_\star)$", fontsize=8.5)
    ov.add_patch(Circle((6.30, 3.35), 0.14, facecolor="#2b6cb0",
                        edgecolor="#1a3a66", lw=0.8, zorder=5))
    label(fig, 6.30, 3.08, r"$\mathbf{x}_\star$", fontsize=7.5)
    for gx, gy in [(6.85, 3.55), (7.35, 3.55), (6.85, 2.85), (7.35, 2.85)]:
        ov.plot([6.46, gx], [3.35, gy + 0.23], color="#9fb3cc", lw=0.7,
                zorder=1)
        ov.add_patch(Rectangle((gx, gy), 0.46, 0.46, facecolor="#eef0f3",
                               edgecolor="#8a8f98", lw=0.6, zorder=3))
    note(fig, ctr(c2, w2),
         r"$K$ nearest training samples"
         "\nunder the Euclidean metric;"
         "\nretrieved from $\\mathcal{D}_{\\mathrm{train}}$,"
         "\nnot synthesized perturbations", y=1.95)

    # ================= STAGE 3: ADAPTIVE LOCALITY =======================
    heading(fig, ctr(c3, w3), "3   Gaussian locality", color="#d98b5f")
    kd = fig.add_axes(R(8.45, 3.05, 1.95, 1.35))
    dd = np.linspace(0, 2.6, 200)
    kd.plot(dd, np.exp(-0.5 * dd ** 2), color="#d98b5f", lw=1.4)
    rng = np.random.default_rng(3)
    dn = np.sort(rng.gamma(2.2, 0.42, 14))
    dn = dn / np.median(dn)
    wts = np.exp(-0.5 * dn ** 2)
    kd.scatter(dn, wts, s=6 + 26 * wts, c=wts, cmap="YlOrRd", vmin=0,
               vmax=1, edgecolors="#555555", linewidths=0.4, zorder=5)
    kd.axvline(1.0, color="#555555", ls="--", lw=0.9)
    kd.text(1.06, 0.92, r"$\sigma$ = median", fontsize=7.0, va="top",
            color="#555555")
    kd.set_xlabel(r"neighbor distance $d_j$ (units of $\sigma$)",
                  fontsize=7.0, labelpad=1.5)
    kd.set_ylabel(r"$K_\sigma(d)$", fontsize=7.5)
    kd.set_xlim(0, 2.6), kd.set_ylim(0, 1.02)
    kd.tick_params(labelsize=6.2, direction="in", length=2.5, width=0.8)
    kd.spines["top"].set_visible(False)
    kd.spines["right"].set_visible(False)
    pis = np.sort(wts)[::-1][:6]
    for i, p in enumerate(pis):
        yy = 2.42 - i * 0.155
        ov.add_patch(Rectangle((ctr(c3, w3) - p * 0.95 / 2, yy), p * 0.95,
                               0.11, facecolor=plt.cm.YlOrRd(0.15 + 0.8 * p),
                               edgecolor="#777777", lw=0.4))
    label(fig, ctr(c3, w3), 1.45,
          r"weighted, centered neighborhood $\widetilde{\mathbf{X}}_K$",
          fontsize=7.5)
    note(fig, ctr(c3, w3),
         r"$\pi_j = K_\sigma(d_j)\, / \sum_{\ell \in \mathcal{N}_K} K_\sigma(d_\ell)$"
         "\n"
         r"adaptive bandwidth $\sigma = \mathrm{median}\, d_j$;"
         "\nno bandwidth hyper-parameter;"
         "\nnot a retraining step", y=1.18)

    # ================= STAGE 4: COVARIANCE CONVERSION ====================
    heading(fig, ctr(c4, w4), "4   Pattern conversion", color="#c0392b")
    ov.add_patch(Rectangle((c4 + 0.04, 0.40), w4 - 0.08, 4.55, fill=False,
                           lw=1.4, ec="#c0392b", zorder=0))
    label(fig, ctr(c4, w4), 4.98,
          r"$\mathbf{a} = \dfrac{\mathrm{Cov}_{\Pi^K}\!\left[\mathbf{h}_{\mathbf{x}_\star}(\mathbf{x}),\, \tilde{y}\right]}"
          r"{\mathrm{Var}_{\Pi^K}\!\left[\tilde{y}\right] + \lambda}$",
          fontsize=8.5, va="top")
    label(fig, ctr(c4, w4), 3.98,
          r"$\mathbf{a} = \dfrac{\mathbf{C}_K\, \mathbf{w}^{\mathrm{PG}}}"
          r"{(\mathbf{w}^{\mathrm{PG}})^{\!\top} \mathbf{C}_K\, \mathbf{w}^{\mathrm{PG}} + \lambda}$",
          fontsize=10.0, va="top", color="#c0392b")
    ay = 2.55
    ov.add_patch(FancyArrowPatch((11.30, ay), (11.78, ay), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#333333"))
    ov.add_patch(FancyArrowPatch((12.62, ay), (13.10, ay), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#333333"))
    label(fig, 11.10, ay, r"$\mathbf{w}^{\mathrm{PG}}$", va="center",
          fontsize=10)
    label(fig, ctr(c4, w4), ay + 0.34, r"$\mathbf{C}_K$", va="center",
          fontsize=10)
    ov.add_patch(Rectangle((11.82, ay - 0.25), 0.76, 0.50, fill=False,
                           lw=1.2, ec="#c0392b"))
    label(fig, ctr(c4, w4), ay + 0.03, "never\nmaterialized", va="center",
          fontsize=6.6, color="#c0392b", linespacing=1.15)
    label(fig, 13.24, ay, r"$\mathbf{a}$", va="center", fontsize=10)
    label(fig, ctr(c4, w4), 2.00,
          r"two-pass product:"
          "\n"
          r"$\widetilde{\mathbf{X}}_K^{\!\top}\!\left(\boldsymbol{\pi} \odot (\widetilde{\mathbf{X}}_K\, \mathbf{w}^{\mathrm{PG}})\right)$",
          fontsize=7.6, linespacing=1.4)
    note(fig, ctr(c4, w4),
         r"$\mathcal{O}(KD)$ time and space;"
         "\nno $D \\times D$ matrix; $\\lambda = 10^{-5}$", y=1.30)

    # ======================== COMPARISON PANEL ==========================
    heading(fig, ctr(c5, w5), "Comparison")
    label(fig, ctr(c5, w5), 5.12, "same instance, same scale",
          fontsize=6.6, color="#777777")
    cmp_x = ctr(c5, w5) - 0.5
    for y, arr, cm, lbl in [
        (3.75, gt, cmap_data, "ground-truth signal"),
        (2.25, w_ig, cmap_attr, "raw Integrated Gradients"),
        (0.75, a_out, cmap_attr, "PatternGrad"),
    ]:
        tile(fig, cmp_x, y, 1.0, arr, cm)
        label(fig, ctr(c5, w5), y - 0.13, lbl, fontsize=7.2)

    # ======================== inter-stage arrows ========================
    for xa, xb in [(2.02, 2.38), (5.38, 5.72), (7.98, 8.28),
                   (10.58, 10.88), (13.48, 13.78)]:
        fig.add_artist(FancyArrowPatch((xa / W, 2.60 / H), (xb / W, 2.60 / H),
                                       arrowstyle="-|>", mutation_scale=13,
                                       lw=1.6, color="#222222"))

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = os.path.join(OUT, f"patterngrad_conceptual_diagram.{ext}")
        fig.savefig(out, format=ext, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
