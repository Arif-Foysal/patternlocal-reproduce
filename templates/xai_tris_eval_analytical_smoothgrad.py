#!/usr/bin/env python3
"""Targeted evaluation: Analytical PatternLocal (gradient vs SmoothGrad).

Compares the analytical explainer's two surrogate choices on XOR CORR
alpha=0.2:

  * LIME (fixed bandwidth, reference)
  * PatternLocal (fixed auto bandwidth, reference)
  * Analytical-gradient (exact input gradient + KNN covariance)
  * Analytical-smoothgrad (SmoothGrad surrogate + KNN covariance)

Analytical config: FIXED kernel=gaussian + auto (median k-NN distance)
bandwidth; k_neighbors and (for smoothgrad) noise level are tuned on a
held-out split by minimizing EMD. Only the analytical method is the focus;
LIME/PatternLocal are references.

NOTE: the cached XAI-TRIS archives store masks only for the test split, so the
tuning split is drawn from the test set and the final comparison uses the
complement.
"""
import glob
import json
import os
import sys
import time

import numpy as np

# --- locate and import the core library (uploaded as the core dataset) -------
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "patternlocal.py")):
        sys.path.insert(0, root)
        break

from xai.patternlocal import patternlocal_explain
from xai.lime import lime_explain, model_logits
from xai.analytical_patternlocal import (
    AnalyticalPatternLocalExplainer, kernel_weights,
)
from metrics.emd import compute_emd
from metrics.ime import compute_ime

CONFIG = "XOR_CORR_alpha0.2"
IMAGE_SHAPE = (64, 64)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Evaluation sizing.
N_TUNE = 30
N_TEST = 150
N_LIME = 2000
K_RATIO = 0.05

# FIXED analytical knobs (kernel tuning hurt EMD).
ANALYTICAL_KERNEL = "gaussian"
ANALYTICAL_BANDWIDTH = None  # auto = median k-NN distance

# Tuning grids for the analytical method.
ANALYTICAL_KS = [100, 300, 500, 800]
SMOOTHGRAD_NOISE = [0.05, 0.1, 0.2]
SMOOTHGRAD_N = 50


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def save_results(results_dict, path="/kaggle/working/eval_results.json"):
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {path}")


def find_file(pattern):
    matches = glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True)
    return matches[0] if matches else None


def _score(expl, mask):
    return compute_emd(expl, mask, image_shape=IMAGE_SHAPE), compute_ime(expl, mask)


def _normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


def pattern_explain(x, v, b, X_train, bandwidth, k_ratio, shrinkage):
    return patternlocal_explain(
        lime_weights=v,
        lime_intercept=b,
        instance=x,
        X_train=X_train,
        bandwidth=bandwidth,
        kernel="gaussian",
        k_ratio=k_ratio,
        lam=1e-5,
        shrinkage=shrinkage,
    )


def analytic_attr(explainer, x, grad, dists, idx_nn, k, surrogate, noise):
    """Analytical attribution for instance x using cached KNN."""
    X_K = explainer.X_train[idx_nn[:k]]
    dists_k = dists[:k]
    bw = float(np.median(dists_k))
    if bw == 0:
        bw = np.finfo(float).eps
    weights = kernel_weights(dists_k, bw, ANALYTICAL_KERNEL)
    w_grad = grad if surrogate == "gradient" else explainer.smoothgrad(
        x, n_samples=SMOOTHGRAD_N, noise_level=noise, seed=0,
    )
    return explainer._attribution(w_grad, X_K, weights)


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Targeted Evaluation: {CONFIG}")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    data_path = find_file(f"{CONFIG}.npz")
    model_path = find_file(f"{CONFIG}_model.pt")
    if not data_path or not model_path:
        print(f"ERROR: data/model not found for {CONFIG}")
        sys.exit(1)
    print(f"Data:   {data_path}")
    print(f"Model:  {model_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    masks_test = data["masks_test"]
    print(f"X_train={X_train.shape}  X_test={X_test.shape}")

    import torch
    from models.mlp import MLPModel

    model = MLPModel(input_dim=IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_acc = float((model_logits(model, X_test).argmax(axis=1) == y_test).mean())
    print(f"Test accuracy: {test_acc:.4f}")
    format_result("test_accuracy", f"{test_acc:.4f}")

    # Held-out tuning split.
    rng = np.random.default_rng(42)
    all_idx = rng.choice(len(X_test), size=min(N_TEST, len(X_test)), replace=False)
    tune_idx = all_idx[:N_TUNE]
    eval_idx = all_idx[N_TUNE:]
    print(f"Tune samples: {len(tune_idx)}  Eval samples: {len(eval_idx)}")

    X_tune = X_test[tune_idx]
    X_eval = X_test[eval_idx]
    mask_tune = masks_test[tune_idx]
    mask_eval = masks_test[eval_idx]

    # Fixed baselines.
    lime_bw = 0.75 * np.sqrt(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    pat_bw = None
    print(f"LIME bw={lime_bw:.2f}  PatternLocal bw=auto  "
          f"Analytical kernel={ANALYTICAL_KERNEL} bw=auto  ks={ANALYTICAL_KS}")
    format_result("lime_bandwidth", f"{lime_bw:.2f}")
    format_result("patternlocal_bandwidth", "auto")

    # ---- build analytical explainer + cache gradients/knn --------------
    print("\n--- Building analytical explainer ---")
    max_k = max(ANALYTICAL_KS)
    explainer = AnalyticalPatternLocalExplainer(
        model, X_train, k_neighbors=max_k, device=DEVICE,
        normalize=False, kernel=ANALYTICAL_KERNEL, bandwidth=ANALYTICAL_BANDWIDTH,
    )
    t0 = time.time()
    tune_grads = np.array([explainer.gradient(X_test[i]) for i in tune_idx])
    eval_grads = np.array([explainer.gradient(X_test[i]) for i in eval_idx])
    print(f"  {len(tune_grads)+len(eval_grads)} gradients in {time.time()-t0:.1f}s")
    t0 = time.time()
    tune_dist, tune_idx_nn = explainer.knn.kneighbors(X_tune, n_neighbors=max_k)
    eval_dist, eval_idx_nn = explainer.knn.kneighbors(X_eval, n_neighbors=max_k)
    print(f"  KNN cached ({max_k}) in {time.time()-t0:.1f}s")

    # ---- tune analytical-gradient k on tuning split ----------------------
    print("\n--- Tuning Analytical-gradient k ---")
    best_grad_k = (None, np.inf)
    for k in ANALYTICAL_KS:
        emds = []
        for j, i in enumerate(tune_idx):
            a = analytic_attr(explainer, X_test[i], tune_grads[j],
                              tune_dist[j], tune_idx_nn[j], k, "gradient", 0.0)
            e, _ = _score(_normalize(a), mask_tune[j])
            emds.append(e)
        m = float(np.mean(emds))
        print(f"  grad k={k:5d}  EMD={m:.4f}")
        if m < best_grad_k[1]:
            best_grad_k = (k, m)
    print(f"  -> grad k = {best_grad_k[0]}")
    format_result("analytical_grad_k", f"{best_grad_k[0]}")

    # ---- tune analytical-smoothgrad (k + noise) on tuning split ----------
    print("\n--- Tuning Analytical-smoothgrad k + noise ---")
    best_sg = (None, None, np.inf)
    for k in ANALYTICAL_KS:
        for noise in SMOOTHGRAD_NOISE:
            emds = []
            for j, i in enumerate(tune_idx):
                a = analytic_attr(explainer, X_test[i], tune_grads[j],
                                  tune_dist[j], tune_idx_nn[j], k,
                                  "smoothgrad", noise)
                e, _ = _score(_normalize(a), mask_tune[j])
                emds.append(e)
            m = float(np.mean(emds))
            print(f"  sg k={k:5d} noise={noise}  EMD={m:.4f}")
            if m < best_sg[2]:
                best_sg = (k, noise, m)
    print(f"  -> sg k={best_sg[0]} noise={best_sg[1]}")
    format_result("analytical_sg_k", f"{best_sg[0]}")
    format_result("smoothgrad_noise", f"{best_sg[1]}")

    # ---- final comparison on the held-out eval split -----------------------
    print("\n--- Final evaluation ---")
    lime_emds, lime_imes = [], []
    pat_emds, pat_imes = [], []
    grad_emds, grad_imes = [], []
    sg_emds, sg_imes = [], []
    t0 = time.time()
    for j, i in enumerate(eval_idx):
        x = X_eval[j]
        mask = mask_eval[j]

        # LIME
        v, b = lime_explain(model, x, bandwidth=lime_bw, seed=int(i))
        e, im = _score(_normalize(v), mask)
        lime_emds.append(e)
        lime_imes.append(im)

        # PatternLocal
        p = pattern_explain(x, v, b, X_train, pat_bw, K_RATIO, 0.0)
        e, im = _score(_normalize(p), mask)
        pat_emds.append(e)
        pat_imes.append(im)

        # Analytical-gradient (best k)
        gk = best_grad_k[0]
        ag = analytic_attr(explainer, x, eval_grads[j],
                           eval_dist[j], eval_idx_nn[j], gk, "gradient", 0.0)
        e, im = _score(_normalize(ag), mask)
        grad_emds.append(e)
        grad_imes.append(im)

        # Analytical-smoothgrad (best k, best noise)
        sk, sn = best_sg[0], best_sg[1]
        asg = analytic_attr(explainer, x, eval_grads[j],
                            eval_dist[j], eval_idx_nn[j], sk, "smoothgrad", sn)
        e, im = _score(_normalize(asg), mask)
        sg_emds.append(e)
        sg_imes.append(im)
    print(f"  evaluated {len(eval_idx)} samples in {time.time()-t0:.1f}s")

    lime_emd, lime_ime = float(np.mean(lime_emds)), float(np.mean(lime_imes))
    pat_emd, pat_ime = float(np.mean(pat_emds)), float(np.mean(pat_imes))
    grad_emd, grad_ime = float(np.mean(grad_emds)), float(np.mean(grad_imes))
    sg_emd, sg_ime = float(np.mean(sg_emds)), float(np.mean(sg_imes))

    print(f"\nResults over {len(eval_idx)} test samples:")
    print(f"  LIME           EMD: {lime_emd:.4f} +/- {np.std(lime_emds):.4f}")
    print(f"  LIME           IME: {lime_ime:.4f} +/- {np.std(lime_imes):.4f}")
    print(f"  PatternLocal   EMD: {pat_emd:.4f} +/- {np.std(pat_emds):.4f}")
    print(f"  PatternLocal   IME: {pat_ime:.4f} +/- {np.std(pat_imes):.4f}")
    print(f"  Analytical-grad EMD: {grad_emd:.4f} +/- {np.std(grad_emds):.4f}")
    print(f"  Analytical-grad IME: {grad_ime:.4f} +/- {np.std(grad_imes):.4f}")
    print(f"  Analytical-sg   EMD: {sg_emd:.4f} +/- {np.std(sg_emds):.4f}")
    print(f"  Analytical-sg   IME: {sg_ime:.4f} +/- {np.std(sg_imes):.4f}")
    print(f"\n  Analytical-sg EMD improvement vs LIME: {lime_emd - sg_emd:+.4f}")
    print(f"  Analytical-sg IME improvement vs LIME: {lime_ime - sg_ime:+.4f}")
    print(f"  Analytical-sg EMD improvement vs grad: {grad_emd - sg_emd:+.4f}")
    print(f"  Analytical-sg IME improvement vs grad: {grad_ime - sg_ime:+.4f}")

    for k, v in {
        "lime_emd_mean": f"{lime_emd:.4f}", "lime_ime_mean": f"{lime_ime:.4f}",
        "patternlocal_emd_mean": f"{pat_emd:.4f}", "patternlocal_ime_mean": f"{pat_ime:.4f}",
        "analytical_grad_emd_mean": f"{grad_emd:.4f}",
        "analytical_grad_ime_mean": f"{grad_ime:.4f}",
        "analytical_sg_emd_mean": f"{sg_emd:.4f}",
        "analytical_sg_ime_mean": f"{sg_ime:.4f}",
        "analytical_sg_emd_std": f"{np.std(sg_emds):.4f}",
        "analytical_sg_ime_std": f"{np.std(sg_imes):.4f}",
        "analytical_grad_k": best_grad_k[0],
        "analytical_sg_k": best_sg[0],
        "smoothgrad_noise": best_sg[1],
        "smoothgrad_better_emd_vs_lime": bool(sg_emd < lime_emd),
        "smoothgrad_better_ime_vs_lime": bool(sg_ime < lime_ime),
        "smoothgrad_better_emd_vs_grad": bool(sg_emd < grad_emd),
        "smoothgrad_better_ime_vs_grad": bool(sg_ime < grad_ime),
    }.items():
        format_result(k, v)

    # Gate: smoothgrad beats LIME on EMD.
    passed = bool(sg_emd < lime_emd)
    format_result("PASS" if passed else "FAIL", "analytical-smoothgrad vs LIME (EMD)")
    format_result("total_seconds", f"{time.time()-t_start:.1f}")

    results = {
        "config": CONFIG,
        "test_accuracy": test_acc,
        "n_eval": int(len(eval_idx)),
        "device": DEVICE,
        "lime_bandwidth": float(lime_bw),
        "analytical_kernel": ANALYTICAL_KERNEL,
        "analytical_grad_k": best_grad_k[0],
        "analytical_sg_k": best_sg[0],
        "smoothgrad_noise": best_sg[1],
        "lime": {"emd_mean": lime_emd, "ime_mean": lime_ime},
        "patternlocal": {"emd_mean": pat_emd, "ime_mean": pat_ime},
        "analytical_grad": {"emd_mean": grad_emd, "ime_mean": grad_ime},
        "analytical_smoothgrad": {
            "emd_mean": sg_emd, "ime_mean": sg_ime,
            "emd_std": float(np.std(sg_emds)),
            "ime_std": float(np.std(sg_imes)),
        },
        "passed": passed,
    }
    save_results(results)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
