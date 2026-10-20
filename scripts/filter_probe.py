#!/usr/bin/env python3
"""Probe: which Sobel/Laplace variant reproduces the paper's flat, high EMD?

The paper's XOR CORR filters stay at ~0.12-0.14 (Sobel) / 0.08-0.135 (Laplace)
across ALL alpha and never improve, per the Remark: "increasing alpha sharpens
tetromino boundaries and causes the filters to respond outside the true region".

Ours collapse to ~0.02 at alpha=0.9 because 88% of the filter mass lands inside
the (large, 26%-of-image) F+ mask.  This script recomputes EMD for many variant
definitions of the filter explanation and reports how much mass lands inside F+.

Uses the 5 saved samples per (scenario, alpha) npz.  Indicative, not exact -- but
the effect size (0.006 vs 0.125) is ~20x, so 5 samples discriminates fine.

Run:  python3 scripts/filter_probe.py
"""

import glob
import os
import sys

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from paper_repro.metrics.emd import compute_emd  # noqa: E402

# ---------------------------------------------------------------- kernels ---
# 3x3 Sobel (the classic Gx/Gy).  scipy's ndimage.sobel uses a separable
# [-1,0,1] first-difference instead, so these are genuinely different.
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = _SOBEL_X.T
_LAPLACE4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
_LAPLACE8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


def _conv(img, k):
    return ndimage.convolve(img, k, mode="constant", cval=0.0)


# ------------------------------------------------------------- variants -----
# Each variant is (name, func) where func(x2d, x_signal2d) -> 1-D explanation.

def v_sobel_ndimage(x2, xs):
    """Current implementation: scipy.ndimage.sobel magnitude."""
    gx = ndimage.sobel(x2, axis=0)
    gy = ndimage.sobel(x2, axis=1)
    return np.sqrt(gx**2 + gy**2).ravel()


def v_sobel_ndimage_sep(x2, xs):
    """scipy.ndimage.sobel, but TRUE 3x3 Sobel via convolve."""
    gx = _conv(x2, _SOBEL_X)
    gy = _conv(x2, _SOBEL_Y)
    return np.sqrt(gx**2 + gy**2).ravel()


def v_sobel_absmax(x2, xs):
    """sqrt replaced by max(|gx|,|gy|)."""
    gx = _conv(x2, _SOBEL_X)
    gy = _conv(x2, _SOBEL_Y)
    return np.maximum(np.abs(gx), np.abs(gy)).ravel()


def v_sobel_sumabs(x2, xs):
    """|gx| + |gy|."""
    gx = _conv(x2, _SOBEL_X)
    gy = _conv(x2, _SOBEL_Y)
    return (np.abs(gx) + np.abs(gy)).ravel()


def v_laplace_ndimage(x2, xs):
    """Current implementation: |scipy.ndimage.laplace(x)|."""
    return np.abs(ndimage.laplace(x2)).ravel()


def v_laplace_signed(x2, xs):
    """Signed Laplace (no abs)."""
    return ndimage.laplace(x2).ravel()


def v_laplace_neg(x2, xs):
    """-Laplace (the 'blob detector' sign convention)."""
    return -ndimage.laplace(x2).ravel()


def v_laplace_8(x2, xs):
    """8-neighbour Laplacian kernel."""
    return np.abs(_conv(x2, _LAPLACE8)).ravel()


def v_laplace_ndimage_signal(x2, xs):
    """Laplace applied to the SIGNAL component only (alpha*(H o a))."""
    return np.abs(ndimage.laplace(xs)).ravel()


def v_sobel_ndimage_signal(x2, xs):
    """Sobel applied to the SIGNAL component only."""
    gx = ndimage.sobel(xs, axis=0)
    gy = ndimage.sobel(xs, axis=1)
    return np.sqrt(gx**2 + gy**2).ravel()


def _blur(x2, sigma):
    return ndimage.gaussian_filter(x2, sigma=sigma)


def v_sobel_blur(x2, xs):
    """Sobel after a 1.5 blur (matches the signal smoothing sigma)."""
    return v_sobel_ndimage(_blur(x2, 1.5), xs)


def v_laplace_blur(x2, xs):
    return v_laplace_ndimage(_blur(x2, 1.5), xs)


VARIANTS = [
    ("sobel_ndimage (OURS)", v_sobel_ndimage),
    ("sobel_3x3_kernel", v_sobel_ndimage_sep),
    ("sobel_absmax", v_sobel_absmax),
    ("sobel_sumabs", v_sobel_sumabs),
    ("sobel_on_signal", v_sobel_ndimage_signal),
    ("sobel_blur1.5", v_sobel_blur),
    ("laplace_ndimage |.| (OURS)", v_laplace_ndimage),
    ("laplace_signed", v_laplace_signed),
    ("laplace_neg", v_laplace_neg),
    ("laplace_8kernel", v_laplace_8),
    ("laplace_on_signal |.|", v_laplace_ndimage_signal),
    ("laplace_blur1.5 |.|", v_laplace_blur),
]


def _normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


def load_samples():
    """Return {(scenario, alpha): (X, mask)} from the saved npz files."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results",
                                    "patternlocal-repro-*-sweep-*",
                                    "*_sample_images.npz")):
        base = os.path.basename(f)  # e.g. alpha0.9_sample_images.npz
        alpha = float(base.split("alpha")[1].split("_")[0])
        # scenario comes from the parent directory slug, e.g.
        # patternlocal-repro-xor-corr-sweep-9 -> XOR_CORR
        parent = os.path.basename(os.path.dirname(f))
        slug = parent.replace("patternlocal-repro-", "")
        slug = slug.rsplit("-sweep-", 1)[0]
        scen = slug.replace("-", "_").upper()
        d = np.load(f)
        out[(scen, alpha)] = (np.asarray(d["X"], np.float64), d["mask"])
    return out


def signal_component(x, alpha):
    """Recover alpha*(H o a) = x - (1-alpha)*(G o eta) is not separable without
    the noise.  We instead approximate the signal as the alpha-scaled smoothed
    version of the largest connected structure -- but simpler: return x itself
    for WHITE/CORR we cannot recover the split, so use a blurred x as a proxy.
    """
    # NOTE: kept for API symmetry; variants that need it use a render of the
    # *image* itself in practice because the split is not stored in the npz.
    return x


def main():
    samples = load_samples()
    if not samples:
        print("no sample npz found under results/")
        return
    print("loaded:", sorted(samples))

    for (scen, alpha), (X, mask) in sorted(samples.items()):
        print("\n" + "=" * 78)
        print(f"{scen} alpha={alpha}   (n={len(X)} saved samples)")
        print("=" * 78)
        fp_frac = mask.reshape(len(mask), -1).sum(1) / mask[0].size
        print(f"  F+ covers {100*fp_frac.mean():.1f}% of the image on average")
        print(f"  {'variant':<28} {'EMD':>8} {'mass-in-F+':>11}")
        print("  " + "-" * 50)
        for name, fn in VARIANTS:
            emds, inside = [], []
            for i in range(len(X)):
                x2 = X[i]
                expl = fn(x2, x2)
                e = compute_emd(_normalize(expl), mask[i])
                emds.append(e)
                s = np.abs(_normalize(expl))
                s = s / s.sum()
                inside.append((s * mask[i].ravel()).sum())
            print(f"  {name:<28} {np.mean(emds):8.4f} {100*np.mean(inside):10.1f}%")


if __name__ == "__main__":
    main()
