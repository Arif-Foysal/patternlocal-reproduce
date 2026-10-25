#!/usr/bin/env python3
"""Figure data export: dataset-overview samples + full qualitative bundles.

Outputs to /kaggle/working/figure_bundle/:
  dataset_overview.npz : for all 6 scenarios at alpha=0.2, the FIRST test
                         index with y=0 and the FIRST with y=1: instance (x)
                         and ground-truth mask, raw (un-normalized), 64x64.
  qual_<ROW>.npz       : for the 6 qualitative figure rows, the first 5
                         held-out eval instances (same seed-42 split as the
                         sweeps) with ALL 13 method attributions, raw.
  manifest.json        : provenance for every array (scenario, alpha, index).

The qualitative attributions reuse the sweeps' tuned hyperparameters
(k_grad, k_ig, LIME bandwidth, PatternLocal k_ratio) read from the fetched
sweep results, and the same split/seed so sample p matches the existing
alpha*_sample_images.npz files exactly.
"""
import json
import os
import sys

import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "eval_common.py")):
        sys.path.insert(0, root)
        break

import xai.eval_common as ec  # noqa: E402  (mounted under <root>/xai/)


SCENARIOS = ["LIN_CORR", "LIN_WHITE", "XOR_CORR", "XOR_WHITE",
             "RIGID_CORR", "RIGID_WHITE"]
IMAGE_SHAPE = ec.IMAGE_SHAPE  # (64, 64)
N_TEST = 600
SEED = 42
TUNE_SAMPLES = ec.TUNE_SAMPLES          # 30
N_SAMPLES = ec.N_SAVE_IMAGES            # 5

METHODS = ["lime", "patternlocal", "analytical_grad", "analytical_ig",
           "plain_grad", "plain_ig", "plain_smoothgrad", "saliency",
           "deeplift", "gradient_shap", "guided_backprop", "sobel", "laplace"]

# Tuned per-sweep hyperparameters, read from the fetched sweep results JSONs
# (results/patternlocal-repro-*-sweep-N/sweep_partN_results.json).
QUAL_ROWS = [
    # row_key,            scenario,     alpha, k_grad, k_ig,  lime_bw, pat_k_ratio
    ("LIN_WHITE_0.2",   "LIN_WHITE",   0.2, 36000, 36000, 30.0, 1.0),
    ("LIN_CORR_0.2",    "LIN_CORR",    0.2, 5000,  1800,  20.0, 0.05),
    ("XOR_WHITE_0.2",   "XOR_WHITE",   0.2, 36000, 36000, 20.0, 1.0),
    ("XOR_CORR_0.2",    "XOR_CORR",    0.2, 10000, 1800,  10.0, 0.05),
    ("RIGID_WHITE_0.7", "RIGID_WHITE", 0.7, 1800,  1800,  30.0, 0.05),
    ("RIGID_CORR_0.3",  "RIGID_CORR",  0.3, 1800,  36000, 30.0, 0.05),
]

# Dataset-overview rows: alpha=0.2 for every scenario (data npz only, no model).
OVERVIEW_ALPHA = 0.2

OUT_DIR = "/kaggle/working/figure_bundle"


def split_eval_idx(X_test):
    """Replicate evaluate_alpha's held-out split (seed 42, n_test 600)."""
    rng = np.random.default_rng(SEED)
    n = min(N_TEST, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    n_tune = min(TUNE_SAMPLES, n // 2)
    return idx[n_tune:]


def collect_all_attrs(ev, x, grad, ig, dists, idx_nn, k_grad, k_ig,
                      lime_v, lime_b, X_train, pat_k_ratio):
    """All 13 method attributions for one instance, using the exact helpers
    evaluate_instance() calls internally (its return dict only exposes 3)."""
    x = np.asarray(x, dtype=np.float32)
    attrs = {
        "lime": lime_v,
        "patternlocal": ec.pattern_from_lime(
            x, lime_v, lime_b, X_train,
            ec.PAT_BANDWIDTH, pat_k_ratio, ec.SHRINKAGE),
        "analytical_grad": ev.analytical_attr(
            x, grad, dists, idx_nn, k_grad, "gradient"),
        "analytical_ig": ev.analytical_attr(
            x, ig, dists, idx_nn, k_ig, "ig"),
        "plain_grad": ec._plain_gradient(ev, x),
        "plain_ig": ec._plain_integrated_gradients(ev, x),
        "plain_smoothgrad": ec._plain_smoothgrad(ev, x),
        "saliency": ec._plain_saliency(ev, x),
        "deeplift": ec._plain_deeplift(ev, x),
        "gradient_shap": ec._plain_gradient_shap(ev, x),
        "guided_backprop": ec._plain_guided_backprop(ev, x),
        "sobel": ec._sobel(ev, x),
        "laplace": ec._laplace(ev, x),
    }
    return attrs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"dataset_overview": {}, "qualitative": {}}

    # ------------------------------------------------------------------
    # 1) Dataset overview: first y=0 and y=1 test sample at alpha=0.2.
    # ------------------------------------------------------------------
    ov = {}
    for scen in SCENARIOS:
        pattern = f"{scen}_alpha{OVERVIEW_ALPHA:.1f}.npz"
        path = ec.find_file(pattern)
        if not path:
            print(f"ERROR: {pattern} not found; skipping {scen}")
            continue
        data = np.load(path)
        X_test, y_test, masks = data["X_test"], data["y_test"], data["masks_test"]
        i0 = int(np.argmax(y_test == 0))
        i1 = int(np.argmax(y_test == 1))
        ov[scen] = {
            "y0_x": X_test[i0].reshape(IMAGE_SHAPE),
            "y0_mask": masks[i0].reshape(IMAGE_SHAPE),
            "y1_x": X_test[i1].reshape(IMAGE_SHAPE),
            "y1_mask": masks[i1].reshape(IMAGE_SHAPE),
        }
        manifest["dataset_overview"][scen] = {
            "alpha": OVERVIEW_ALPHA, "idx_y0": i0, "idx_y1": i1,
            "data_file": os.path.basename(path),
        }
        print(f"  overview {scen}: y0 idx {i0}, y1 idx {i1}", flush=True)

    np.savez(os.path.join(OUT_DIR, "dataset_overview.npz"), **ov)

    # ------------------------------------------------------------------
    # 2) Qualitative rows: 5 eval samples x 13 methods each.
    # ------------------------------------------------------------------
    for (row_key, scen, alpha, k_grad, k_ig, lime_bw, pat) in QUAL_ROWS:
        print(f"\n=== {row_key} ===", flush=True)
        pattern = f"{scen}_alpha{alpha:.1f}.npz"
        model_pattern = f"{scen}_alpha{alpha:.1f}_model.pt"
        data_path = ec.find_file(pattern)
        model_path = ec.find_file(model_pattern)
        if not data_path or not model_path:
            print(f"ERROR: missing data or model for {row_key}; skipping")
            continue
        data = np.load(data_path)
        X_train, X_test = data["X_train"], data["X_test"]
        y_test, masks_test = data["y_test"], data["masks_test"]
        D = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
        model = ec.load_model(model_path, D)
        model.to(ec.DEVICE)

        eval_idx = split_eval_idx(X_test)
        ev = ec.XORCorrEvaluator(model, X_train, device=ec.DEVICE)

        arrays = {"X": [], "mask": [], "y": []}
        for m in METHODS:
            arrays[m] = []
        tags = []
        for p, i in enumerate(eval_idx[:N_SAMPLES]):
            x = X_test[i]
            mask = masks_test[i]
            v, b = ec.lime_explain(model, x, n_samples=ec.N_LIME,
                                   bandwidth=lime_bw, seed=int(i),
                                   device=ec.DEVICE)
            grad = ev.gradient(x)
            ig = ev.integrated_gradients(x, steps=ec.IG_STEPS)
            dists, idx_nn = ev.knn(x.reshape(1, -1).astype(np.float32))
            attrs = collect_all_attrs(ev, x, grad, ig, dists, idx_nn,
                                      k_grad, k_ig, v, b, X_train, pat)
            arrays["X"].append(x.reshape(IMAGE_SHAPE))
            arrays["mask"].append(mask.reshape(IMAGE_SHAPE))
            arrays["y"].append(int(y_test[i]))
            for m in METHODS:
                arrays[m].append(np.asarray(attrs[m], dtype=np.float64).reshape(IMAGE_SHAPE))
            tags.append(f"alpha{alpha:.1f}_sample{p}")
            print(f"  sample {p} (test idx {i}, y={int(y_test[i])}) done", flush=True)

        np.savez(os.path.join(OUT_DIR, f"qual_{row_key}.npz"),
                 tags=np.array(tags),
                 X=np.stack(arrays["X"]),
                 mask=np.stack(arrays["mask"]),
                 y=np.array(arrays["y"]),
                 **{m: np.stack(arrays[m]) for m in METHODS})
        manifest["qualitative"][row_key] = {
            "scenario": scen, "alpha": alpha,
            "k_grad": k_grad, "k_ig": k_ig,
            "lime_bandwidth": lime_bw, "pat_k_ratio": pat,
            "test_indices": [int(i) for i in eval_idx[:N_SAMPLES]],
        }
        print(f"  saved qual_{row_key}.npz", flush=True)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nFigure bundle complete:", os.listdir(OUT_DIR), flush=True)


if __name__ == "__main__":
    main()
