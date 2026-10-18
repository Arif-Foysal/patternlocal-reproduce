"""Plot qualitative comparisons from our .npz sample-images outputs.

Reads alpha{alpha}_qualitative_ridge.npz files from results/ and plots, per alpha:
  input | ground truth mask | LIME | PatternLocal-Ridge | Analytical-IG-Ridge

Attribution columns share a single Feature Importance colorbar (-1 to 1).
Saves one figure per alpha to results/figures/.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # registers 'icefire' colormap
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
OUT_DIR = RESULTS_DIR / "figures"

METHODS = [
    ("input",               "Input"),
    ("mask",                "Ground Truth"),
    ("lime",                "LIME"),
    ("patternlocal_ridge",  "PatternLocal-Ridge"),
    ("analytical_ig_ridge", "Analytical-IG-Ridge"),
]

CMAPS = {
    "input":               "RdBu_r",
    "mask":                "magma",
    "lime":                "icefire",
    "patternlocal_ridge":  "icefire",
    "analytical_ig_ridge": "icefire",
}

# Columns that share the Feature Importance colorbar (-1 to 1).
ATTRIB_COLUMNS = (2, 3, 4)


def _threshold_attr(arr, percentile=80):
    """Visualization-only thresholding: zero out bottom percentile of |values|."""
    threshold = np.percentile(np.abs(arr), percentile)
    arr_thresh = arr.copy()
    arr_thresh[np.abs(arr_thresh) < threshold] = 0
    return arr_thresh


def plot_alpha(npz_path, out_path):
    data = np.load(npz_path, allow_pickle=True)
    tags = [t.decode() if isinstance(t, bytes) else str(t) for t in data["tags"]]
    X = data["X"]                    # (N, 64, 64)
    mask = data["mask"]              # (N, 64, 64)
    lime = data["lime"]
    pl = data["patternlocal_ridge"]
    aig = data["analytical_ig_ridge"]

    arrays = [X, mask, lime, pl, aig]
    n_samples = X.shape[0]
    n_methods = len(arrays)

    # Extra column on the right for the shared Feature Importance colorbar.
    fig = plt.figure(figsize=(n_methods * 2.5 + 0.8, n_samples * 2.5))
    gs = fig.add_gridspec(n_samples, n_methods + 1,
                          width_ratios=[1] * n_methods + [0.15],
                          wspace=0.05, hspace=0.05)

    for s in range(n_samples):
        tag = tags[s]
        for m_idx, (key, label) in enumerate(METHODS):
            ax = fig.add_subplot(gs[s, m_idx])
            arr = arrays[m_idx][s]
            if arr.ndim == 1:
                arr = arr.reshape(64, 64)

            # 1. Input Image Scaling (Dynamic Min-Max to force contrast)
            if key == "input":
                arr_min = arr.min()
                arr_max = arr.max()
                arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
                im = ax.imshow(arr_norm, cmap=CMAPS[key], vmin=0, vmax=1,
                               interpolation='nearest')

            # 2. Ground Truth Mask Scaling
            elif key == "mask":
                im = ax.imshow(arr, cmap=CMAPS[key], vmin=0, vmax=1,
                               interpolation='nearest')

            # 3. Attribution Scaling (icefire diverging, fixed -1 to 1)
            else:
                # Visualization-only: zero out bottom 80% for crisper look
                arr = _threshold_attr(arr, percentile=80)
                im = ax.imshow(arr, cmap="icefire", vmin=-1.0, vmax=1.0,
                               interpolation='nearest')

            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_title(label, fontsize=12)
            if m_idx == 0:
                ax.set_ylabel(tag.replace("alpha", "α ").replace("_sample", " s"),
                              fontsize=10)

        # Spacer column for colorbar label area
        ax_spacer = fig.add_subplot(gs[s, n_methods])
        ax_spacer.axis("off")

    # Shared Feature Importance colorbar spanning all attribution columns.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Feature Importance", fontsize=12)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Match both XOR CORR and XOR WHITE qualitative npz files.
    npz_files = sorted(glob.glob(str(RESULTS_DIR / "patternlocal-repro-xor-corr-qualitative-lasso" / "alpha*_qualitative_ridge.npz")))
    npz_files += sorted(glob.glob(str(RESULTS_DIR / "patternlocal-repro-xor-white-qualitative-ridge" / "alpha*_qualitative_ridge_xor_white.npz")))
    npz_files = sorted(set(npz_files))
    if not npz_files:
        print("No sample-images npz files found.")
        return
    for npz_path in npz_files:
        # Extract alpha from filenames like alpha0.1_qualitative_ridge.npz or alpha0.1_qualitative_ridge_xor_white.npz
        alpha = os.path.basename(npz_path).split("_")[0].replace("alpha", "")
        scenario = "xor_white" if "xor_white" in npz_path else "xor_corr"
        out_path = OUT_DIR / f"qualitative_{scenario}_alpha{alpha}.png"
        plot_alpha(npz_path, out_path)


if __name__ == "__main__":
    main()
