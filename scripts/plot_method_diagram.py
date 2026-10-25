#!/usr/bin/env python3
r"""PatternGrad conceptual diagram (Figure 1, publication style, v2).

Compact two-tier landscape (1.90:1) for full-width placement in a
two-column paper:

  Upper tier: Inputs -> 1 Deterministic IG -> 2 Neighborhood retrieval ->
              3 Adaptive Gaussian locality -> 4 Covariance conversion
  Lower tier: explanation comparison for ONE held-out instance
              (ground-truth support F+, raw IG, PatternGrad), the
              training-set source with an explicit retrieval path rising
              into stage 2, and a legend/provenance block.

Scientific conventions follow sn-article.tex exactly:
  w^PG = s_hat^IG(x*)            (M-step quadrature from x' = 0)
  N_K(x*): K nearest training samples, Euclidean metric
  d_j = ||x^(j) - x*||_2 ;  sigma = median_j d_j  (adaptive, no hyper-param)
  pi_j = 1[j in N_K] K_sigma(d_j) / sum_{l in N_K} K_sigma(d_l)
  x~^(j) = x^(j) - mu, mu = sum_j pi_j x^(j)   (centered neighborhood)
  a = C_K w^PG / ((w^PG)^T C_K w^PG + lambda)  (l2-ridge closed form)
  C_K w^PG = Cov_{Pi_K(x*)}[x, y~]             (h = id)
  two-pass product X~_K^T [pi o (X~_K w^PG)]   (O(KD): conversion step only)

Illustrated instance (reproducible): LIN-WHITE, alpha = 0.2, class y = 0,
test index 78 (first held-out evaluation sample of the sweep bundle).
F+ for LIN is the union of the a^T and a^L supports (manuscript Sect. 4):
the absence of one tetromino is as informative as the presence of the other.

Outputs (repo organization: results/figures/):
  patterngrad_conceptual_diagram.pdf / .svg / .png
LaTeX snippet: results/figures/PATTERNGRAD_FIGURE_SNIPPET.tex

Print-size note: at \textwidth in a two-column paper this canvas shrinks
about 2x; annotation fonts are sized so printed text stays >= 4.5 pt.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.0,
})

HERE = os.path.dirname(__file__)
BUNDLE = os.path.join(HERE, "..", "results",
                      "patternlocal-repro-figure-data-export", "figure_bundle")
OUT = os.path.join(HERE, "..", "results", "figures")

W, H = 11.4, 6.0      # inches; aspect 1.90
SAMPLE = 0            # first held-out eval sample of the bundle

FS_HEAD = 10.5
FS_LABEL = 8.5
FS_NOTE = 8.4
FS_LEGEND = 8.4


def R(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def signed_sqrt(w):
    """Sign-preserving display stretch (identical for every attribution map)."""
    return np.sign(w) * np.sqrt(np.abs(w))


def maxabs(a):
    return a / max(np.max(np.abs(a)), 1e-12)


def load_real():
    """One held-out instance feeds every comparison panel."""
    z = np.load(os.path.join(BUNDLE, "qual_LIN_WHITE_0.2.npz"))
    x_inst = maxabs(np.asarray(z["X"][SAMPLE], dtype=np.float64))
    f_plus = (np.asarray(z["mask"][SAMPLE], dtype=np.float64) > 0.5).astype(float)
    w_ig = maxabs(signed_sqrt(np.asarray(z["plain_ig"][SAMPLE],
                                         dtype=np.float64)))
    a_out = maxabs(signed_sqrt(np.asarray(z["analytical_ig"][SAMPLE],
                                          dtype=np.float64)))
    return x_inst, f_plus, w_ig, a_out


def label(fig, x, y, text, ha="center", va="top", fontsize=FS_LABEL,
          color="#333333", **kw):
    fig.text(x / W, y / H, text, ha=ha, va=va, fontsize=fontsize,
             color=color, **kw)


def heading(fig, cx, text, color="#111111"):
    label(fig, cx, 5.72, text, va="center", fontsize=FS_HEAD, color=color,
          fontweight="bold")


def note(fig, cx, text, y, fontsize=FS_NOTE, color="#555555", ha="center"):
    label(fig, cx, y, text, fontsize=fontsize, color=color, linespacing=1.35,
          ha=ha)


def tile(fig, x, y, s, arr, cmap, vmin=-1.0, vmax=1.0, lw=0.6, ec="#444444"):
    ax = fig.add_axes(R(x, y, s, s))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(lw), sp.set_edgecolor(ec)
    return ax


def main():
    import seaborn as sns
    cmap_data = "coolwarm"
    cmap_attr = sns.color_palette("icefire", as_cmap=True)
    cmap_bin = ListedColormap(["#f4f4f4", "#b2182b"])   # binary F+
    x_inst, f_plus, w_ig, a_out = load_real()

    fig = plt.figure(figsize=(W, H))

    # single overlay for all schematic shapes; absolute-inch coordinates
    ov = fig.add_axes([0, 0, 1, 1])
    ov.axis("off")
    ov.set_xlim(0, W), ov.set_ylim(0, H)
    ov.set_zorder(3)

    # -------- upper-tier stage columns (inch origin, width, center) ------
    c0, w0 = 0.30, 1.90
    c1, w1 = 2.55, 2.00
    c2, w2 = 4.90, 1.95
    c3, w3 = 7.20, 1.95
    c4, w4 = 9.50, 1.65
    ctr = lambda x, w: x + w / 2

    # ============================ INPUTS ================================
    heading(fig, ctr(c0, w0), "Inputs")
    tile(fig, ctr(c0, w0) - 0.44, 4.66, 0.88, x_inst, cmap_data)
    label(fig, ctr(c0, w0), 4.50,
          r"$\mathbf{x}_\star$  target instance", fontsize=FS_LABEL)
    ov.add_patch(Rectangle((ctr(c0, w0) - 0.44, 3.80), 0.88, 0.42,
                           fill=False, lw=1.0, ec="#2b6cb0"))
    label(fig, ctr(c0, w0), 4.01, r"$f$  differentiable classifier",
          va="center", fontsize=FS_LABEL)

    # ==================== STAGE 1: DETERMINISTIC IG =====================
    heading(fig, ctr(c1, w1), "1   Deterministic IG", color="#2d6a4f")
    tile(fig, c1 + 0.12, 4.68, 0.72, np.zeros((64, 64)), cmap_data)
    ov.add_patch(FancyArrowPatch((c1 + 0.90, 5.04), (c1 + 1.16, 5.04),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#2d6a4f"))
    label(fig, c1 + 1.03, 5.16, r"$\gamma$", fontsize=8.0, va="bottom",
          color="#2d6a4f")
    tile(fig, c1 + 1.18, 4.68, 0.72, x_inst, cmap_data)
    label(fig, c1 + 0.48, 4.54, r"$\mathbf{x}' = \mathbf{0}$", fontsize=7.4)
    label(fig, c1 + 1.54, 4.54, r"$\mathbf{x}_\star$", fontsize=7.4)
    ov.add_patch(FancyArrowPatch((ctr(c1, w1), 4.44), (ctr(c1, w1), 4.12),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#555555"))
    tile(fig, ctr(c1, w1) - 0.40, 3.26, 0.80, w_ig, cmap_attr)
    label(fig, ctr(c1, w1), 3.12,
          r"$\mathbf{w}^{\mathrm{PG}} = \hat{\mathbf{s}}^{\mathrm{IG}}(\mathbf{x}_\star)$",
          fontsize=8.2)
    note(fig, ctr(c1, w1),
         r"pixel-level; $M$-step quadrature;"
         "\nno random masks, no surrogate fit;"
         "\ndeterministic for fixed $(f, \\mathbf{x}', M)$", y=2.78)

    # ================ STAGE 2: NEIGHBORHOOD RETRIEVAL ====================
    heading(fig, ctr(c2, w2), "2   Neighborhood", color="#2b6cb0")
    label(fig, ctr(c2, w2), 5.48,
          r"$\mathcal{N}_K(\mathbf{x}_\star)$", fontsize=8.8)
    ov.add_patch(Circle((c2 + 0.28, 4.28), 0.13, facecolor="#2b6cb0",
                        edgecolor="#1a3a66", lw=0.8, zorder=5))
    label(fig, c2 + 0.28, 4.04, r"$\mathbf{x}_\star$", fontsize=7.4)
    for gx, gy in [(c2 + 0.75, 4.48), (c2 + 1.28, 4.48),
                   (c2 + 0.75, 3.88), (c2 + 1.28, 3.88)]:
        ov.plot([c2 + 0.41, gx], [4.28, gy + 0.21], color="#9fb3cc",
                lw=0.7, zorder=1)
        ov.add_patch(Rectangle((gx, gy), 0.42, 0.42, facecolor="#eef0f3",
                               edgecolor="#8a8f98", lw=0.6, zorder=3))
    note(fig, ctr(c2, w2),
         r"$K$ nearest in $\mathcal{D}_{\mathrm{train}}$"
         "\n(Euclidean); no perturbations", y=3.56)

    # ================= STAGE 3: ADAPTIVE LOCALITY =======================
    heading(fig, ctr(c3, w3), "3   Gaussian locality", color="#d98b5f")
    kd = fig.add_axes(R(c3 + 0.20, 4.62, 1.55, 0.80))
    dd = np.linspace(0, 2.6, 200)
    kd.plot(dd, np.exp(-0.5 * dd ** 2), color="#d98b5f", lw=1.3)
    rng = np.random.default_rng(3)
    dn = np.sort(rng.gamma(2.2, 0.42, 12))
    dn = dn / np.median(dn)
    wts = np.exp(-0.5 * dn ** 2)
    kd.scatter(dn, wts, s=4 + 18 * wts, c=wts, cmap="YlOrRd", vmin=0,
               vmax=1, edgecolors="#555555", linewidths=0.4, zorder=5)
    kd.axvline(1.0, color="#555555", ls="--", lw=0.8)
    kd.text(1.07, 0.88, r"$\sigma$", fontsize=8.0, va="top", color="#555555")
    kd.set_ylabel(r"$K_\sigma$", fontsize=8.0)
    kd.set_xlim(0, 2.6), kd.set_ylim(0, 1.02)
    kd.tick_params(labelsize=6.4, direction="in", length=2.2, width=0.7)
    kd.set_xticks([0, 1, 2]), kd.set_yticks([0, 0.5, 1.0])
    kd.spines["top"].set_visible(False)
    kd.spines["right"].set_visible(False)
    note(fig, ctr(c3, w3),
         r"$\pi_j = K_\sigma(d_j)\,/\sum_{\ell \in \mathcal{N}_K} K_\sigma(d_\ell)$"
         "\n"
         r"$d_j = \|\mathbf{x}^{(j)} - \mathbf{x}_\star\|_2$ (units of $\sigma$)",
         y=4.24)
    pis = np.sort(wts)[::-1][:4]
    for i, p in enumerate(pis):
        yy = 3.72 - i * 0.12
        ov.add_patch(Rectangle((ctr(c3, w3) - p * 0.70 / 2, yy), p * 0.70,
                               0.09, facecolor=plt.cm.YlOrRd(0.15 + 0.8 * p),
                               edgecolor="#777777", lw=0.4))
    note(fig, ctr(c3, w3),
         r"$\widetilde{\mathbf{X}}_K$:  $\tilde{\mathbf{x}}^{(j)} = \mathbf{x}^{(j)} - \boldsymbol{\mu}$,"
         "  $\\boldsymbol{\\mu} = \\sum_j \\pi_j \\mathbf{x}^{(j)}$",
         y=3.22, fontsize=7.6)
    note(fig, ctr(c3, w3),
         "adaptive, hyper-parameter-free;\nnot a retraining step", y=2.94,
         fontsize=7.4)

    # ================= STAGE 4: COVARIANCE CONVERSION ====================
    ov.add_patch(Rectangle((c4 + 0.02, 3.36), w4 - 0.04, 2.24,
                           facecolor="#fdf6f5", edgecolor="#d8d8d8",
                           lw=0.8, zorder=0))
    heading(fig, ctr(c4, w4), "4   Conversion", color="#c0392b")
    label(fig, ctr(c4, w4), 5.42,
          r"$\mathbf{a} = \dfrac{\mathbf{C}_K\, \mathbf{w}^{\mathrm{PG}}}"
          r"{(\mathbf{w}^{\mathrm{PG}})^{\!\top} \mathbf{C}_K\, \mathbf{w}^{\mathrm{PG}} + \lambda}$",
          fontsize=10.0, va="top", color="#c0392b")
    label(fig, ctr(c4, w4), 4.68,
          r"$\mathbf{C}_K\, \mathbf{w}^{\mathrm{PG}}"
          r" = \mathrm{Cov}_{\Pi_K(\mathbf{x}_\star)}[\mathbf{x},\, \tilde{y}]$",
          fontsize=8.0)
    ay = 4.06
    ov.add_patch(FancyArrowPatch((c4 + 0.22, ay), (c4 + 0.50, ay),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.1, color="#333333"))
    ov.add_patch(FancyArrowPatch((c4 + 1.14, ay), (c4 + 1.42, ay),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.1, color="#333333"))
    label(fig, c4 + 0.10, ay, r"$\mathbf{w}^{\mathrm{PG}}$", va="center",
          fontsize=7.8)
    ov.add_patch(Rectangle((c4 + 0.54, ay - 0.19), 0.56, 0.38, fill=False,
                           lw=1.0, ec="#c0392b"))
    label(fig, c4 + 0.82, ay + 0.02, "never\nformed", va="center",
          fontsize=6.4, color="#c0392b", linespacing=1.1)
    label(fig, c4 + 1.52, ay, r"$\mathbf{a}$", va="center", fontsize=8.5)
    label(fig, ctr(c4, w4), 3.78,
          r"$\widetilde{\mathbf{X}}_K^{\!\top}\!\left[\boldsymbol{\pi} \odot "
          r"\left(\widetilde{\mathbf{X}}_K \mathbf{w}^{\mathrm{PG}}\right)\right]$",
          fontsize=8.0)
    note(fig, ctr(c4, w4),
         r"$\mathcal{O}(KD)$: covariance conversion only", y=3.52,
         fontsize=7.2, color="#8a2f23")

    # ================= inter-stage arrows (upper tier) ==================
    for xa, xb in [(2.24, 2.51), (4.59, 4.86), (6.89, 7.16), (9.19, 9.46)]:
        fig.add_artist(FancyArrowPatch((xa / W, 4.55 / H), (xb / W, 4.55 / H),
                                       arrowstyle="-|>", mutation_scale=12,
                                       lw=1.5, color="#222222"))

    # ======================= LOWER TIER =================================
    # --- explanation comparison (one held-out instance) ------------------
    comp_y0, comp_s = 0.90, 1.25
    for i, (arr, cm, vmin, vmax, lbl) in enumerate([
        (f_plus, cmap_bin, 0.0, 1.0, r"$\mathcal{F}^+$ (ground truth)"),
        (w_ig, cmap_attr, -1.0, 1.0, "raw Integrated Gradients"),
        (a_out, cmap_attr, -1.0, 1.0, "PatternGrad"),
    ]):
        cx = 0.35 + i * 1.50
        tile(fig, cx, comp_y0, comp_s, arr, cm, vmin=vmin, vmax=vmax)
        label(fig, cx + comp_s / 2, comp_y0 - 0.13, lbl, fontsize=8.0)

    # --- training-set source rising into stage 2 (explicit path) ---------
    for dx, dy in [(0.0, 0.0), (0.10, 0.12), (0.20, 0.24)]:
        ov.add_patch(Rectangle((5.10 + dx, 1.00 + dy), 0.55, 0.55,
                               facecolor="#e8eaee", edgecolor="#8a8f98",
                               lw=0.7, zorder=3 - 3 * dy))
    label(fig, 5.40, 0.76,
          r"$\mathcal{D}_{\mathrm{train}}$: the $K$ nearest neighbors"
          "\nof $\\mathbf{x}_\\star$ are retrieved here",
          fontsize=7.8, linespacing=1.25)
    fig.add_artist(FancyArrowPatch((5.40 / W, 1.85 / H), (5.40 / W, 3.26 / H),
                                   arrowstyle="-|>", mutation_scale=12,
                                   lw=1.3, color="#2b6cb0",
                                   linestyle=(0, (4, 2))))

    # --- legend / provenance block ---------------------------------------
    lx = 6.90
    label(fig, lx, 2.62, "Legend", fontsize=9.0, ha="left", va="top",
          fontweight="bold")
    grad = np.linspace(-1, 1, 100).reshape(1, -1)
    tile(fig, lx, 1.92, 0.44, grad, cmap_data, lw=0.4)
    label(fig, lx + 0.54, 2.14,
          "input data; binary support $\\mathcal{F}^+$",
          fontsize=FS_LEGEND, ha="left", va="center")
    tile(fig, lx, 1.26, 0.44, grad, cmap_attr, lw=0.4)
    label(fig, lx + 0.54, 1.48,
          "attributions: signed, max-$|\\cdot|$ normalized,\n"
          "identical signed-sqrt display",
          fontsize=FS_LEGEND, ha="left", va="center", linespacing=1.2)
    label(fig, lx, 0.90,
          "Held-out example:  LIN-WHITE, $\\alpha = 0.2$, $y = 0$,"
          "\ntest index 78 (first held-out eval sample)",
          fontsize=FS_LEGEND, ha="left", va="top", linespacing=1.25)
    label(fig, lx, 0.46,
          "PatternGrad: deterministic pixel-level IG + covariance"
          "\nconversion on the training neighborhood."
          "\nPatternLocal: stochastic binary-mask perturbations,"
          "\nper-instance surrogate refit.",
          fontsize=7.8, ha="left", va="top", color="#555555",
          linespacing=1.3)

    # ============================ export =================================
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = os.path.join(OUT, f"patterngrad_conceptual_diagram.{ext}")
        fig.savefig(out, format=ext, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
