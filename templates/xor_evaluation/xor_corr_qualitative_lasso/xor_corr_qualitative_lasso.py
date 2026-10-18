#!/usr/bin/env python3
"""XOR CORR Qualitative XAI with Ridge (alpha = 0.1 - 0.9).

For each alpha, selects 5 correctly predicted test samples and extracts:
  Input Image, Ground Truth Mask, LIME attribution,
  PatternLocal-Ridge attribution, Analytical-IG-Ridge attribution.

Ridge (closed-form, l1_ratio=0) produces dense attribution maps matching
the original PatternLocal paper. Ground-truth support masking suppresses
attributions outside the mask region.
Saves alpha{alpha}_qualitative_ridge.npz per alpha.
"""
import os
import sys
import time

import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "eval_common.py")):
        sys.path.insert(0, root)
        break

from xai.eval_common import find_file, format_result, XORCorrEvaluator
from xai.lime import lime_explain, model_logits
from xai.patternlocal import patternlocal_explain
from metrics.emd import compute_emd
from metrics.ime import compute_ime

IMAGE_SHAPE = (64, 64)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
N_SAMPLES = 5
N_LIME = 2000
K_NEIGHBORS = 300
IG_STEPS = 50


def _normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


def _normalize_percentile(w, pct=99):
    """Normalize by percentile clip instead of max-abs (suppresses noise)."""
    vmax = np.percentile(np.abs(w), pct)
    if vmax <= 0:
        vmax = 1e-8
    return np.clip(w / vmax, -1, 1)


def _score(expl, mask):
    return compute_emd(expl, mask, image_shape=IMAGE_SHAPE), compute_ime(expl, mask)


def main():
    t_start = time.time()
    print("=" * 60)
    print("XOR CORR Qualitative XAI with Ridge (alpha = 0.1 - 0.9)")
    print("=" * 60)

    all_results = {}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("\n" + "=" * 60)
        print(f"XOR CORR alpha={alpha}")
        print("=" * 60)

        data_path = find_file(f"XOR_CORR_alpha{alpha:.1f}.npz")
        model_path = find_file(f"XOR_CORR_alpha{alpha:.1f}_model.pt")
        if not data_path or not model_path:
            print(f"ERROR: data/model not found for alpha={alpha}")
            continue

        data = np.load(data_path)
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        masks_test = data["masks_test"]
        D = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

        import torch
        from models.mlp import MLPModel
        model = MLPModel(input_dim=D)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(DEVICE)
        model.eval()

        # Test accuracy gate
        logits = model_logits(model, X_test, device=DEVICE)
        preds = logits.argmax(axis=1)
        test_acc = float((preds == y_test).mean())
        print(f"Test accuracy: {test_acc:.4f}")
        if test_acc < 0.90:
            print(f"  SKIP: accuracy {test_acc:.4f} < 0.90")
            continue

        format_result(f"alpha{alpha:.1f}_test_acc", f"{test_acc:.4f}")

        # Find correctly predicted samples
        correct_idx = np.where(preds == y_test)[0]
        print(f"Correctly predicted: {len(correct_idx)} / {len(X_test)}")
        if len(correct_idx) < N_SAMPLES:
            print(f"  SKIP: not enough correct samples ({len(correct_idx)} < {N_SAMPLES})")
            continue

        selected_idx = correct_idx[:N_SAMPLES]

        # Build evaluator (Ridge, original PatternLocal + Analytical-IG)
        ev = XORCorrEvaluator(model, X_train, k_neighbors=K_NEIGHBORS, device=DEVICE)

        # Precompute gradients for selected samples
        grads = {}
        for i in selected_idx:
            grads[int(i)] = ev.gradient(X_test[i])

        # Collect results
        samples = {
            "X": [], "mask": [], "lime": [],
            "patternlocal_ridge": [], "analytical_ig_ridge": [],
        }

        for i in selected_idx:
            x = X_test[i]
            mask = masks_test[i]

            # LIME attribution
            v, b = lime_explain(model, x, n_samples=N_LIME,
                                bandwidth=0.75 * np.sqrt(D), seed=int(i),
                                device=DEVICE)
            lime_attr = _normalize(v)

            # PatternLocal-Ridge attribution (original closed-form)
            pl_ridge = patternlocal_explain(
                v, b, x, X_train, k_ratio=0.05, lam=1e-5,
            )
            pl_ridge_attr = _normalize(pl_ridge)

            # Analytical-IG-Ridge attribution (via evaluator)
            ig = ev.integrated_gradients(x, steps=IG_STEPS)
            dists, idx_nn = ev.knn(np.asarray(x, dtype=np.float32).reshape(1, -1))
            aig_ridge = ev.analytical_attr(x, ig, dists, idx_nn, K_NEIGHBORS, "ig")
            aig_ridge_attr = _normalize(aig_ridge)

            # Metrics on raw attributions (no masking — alignment measured by EMD/IME)
            lime_e, lime_i = _score(lime_attr, mask)
            pl_e, pl_i = _score(pl_ridge_attr, mask)
            aig_e, aig_i = _score(aig_ridge_attr, mask)

            print(f"  Sample {i}: LIME EMD={lime_e:.4f} IME={lime_i:.4f} | "
                  f"PL-Ridge EMD={pl_e:.4f} IME={pl_i:.4f} | "
                  f"AIG-Ridge EMD={aig_e:.4f} IME={aig_i:.4f}")

            samples["X"].append(x.reshape(IMAGE_SHAPE))
            samples["mask"].append(mask.reshape(IMAGE_SHAPE))
            samples["lime"].append(lime_attr.reshape(IMAGE_SHAPE))
            samples["patternlocal_ridge"].append(pl_ridge_attr.reshape(IMAGE_SHAPE))
            samples["analytical_ig_ridge"].append(aig_ridge_attr.reshape(IMAGE_SHAPE))

        # Save .npz
        npz_path = f"/kaggle/working/alpha{alpha:.1f}_qualitative_ridge.npz"
        np.savez(npz_path,
                 tags=[f"alpha{alpha:.1f}_sample{j}" for j in range(N_SAMPLES)],
                 X=np.stack(samples["X"]),
                 mask=np.stack(samples["mask"]),
                 lime=np.stack(samples["lime"]),
                 patternlocal_ridge=np.stack(samples["patternlocal_ridge"]),
                 analytical_ig_ridge=np.stack(samples["analytical_ig_ridge"]))
        print(f"  Saved: {npz_path}")

        format_result(f"alpha{alpha:.1f}_samples", str(N_SAMPLES))
        all_results[f"alpha{alpha:.1f}"] = {
            "test_acc": test_acc,
            "n_samples": N_SAMPLES,
        }

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print("Qualitative XAI complete.")


if __name__ == "__main__":
    main()
