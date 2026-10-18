#!/usr/bin/env python3
"""Targeted evaluation: Analytical PatternLocal with Integrated Gradients surrogate.

Compares the analytical explainer's surrogate choices on XOR CORR alpha=0.2:

  * LIME (fixed bandwidth, reference)
  * PatternLocal (fixed auto bandwidth, reference)
  * Analytical-gradient (exact input gradient + KNN covariance)
  * Analytical-integrated_gradients (IG surrogate + KNN covariance)

Analytical config: FIXED kernel=gaussian + auto bandwidth; k_neighbors tuned
on a held-out split by minimizing EMD for each surrogate.

Integrated Gradients: path from baseline (zeros) to x, average gradients along
the path. Satisfies completeness, smoother spatial focus than raw gradient,
helps EMD while keeping IME low.
"""
import glob
import json
import os
import sys
import time

import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "patternlocal.py")):
        sys.path.insert(0, root)
        break

from xai.patternlocal import patternlocal_explain
from xai.lime import lime_explain, model_logits
from xai.analytical_patternlocal import AnalyticalPatternLocalExplainer, kernel_weights
from metrics.emd import compute_emd
from metrics.ime import compute_ime

CONFIG = "XOR_CORR_alpha0.2"
IMAGE_SHAPE = (64, 64)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

N_TUNE = 30
N_TEST = 150
K_RATIO = 0.05

ANALYTICAL_KERNEL = "gaussian"
ANALYTICAL_KS = [100, 300, 500, 800]
IG_STEPS = 50


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


def analytic_attr(explainer, x, grad, dists, idx_nn, k):
    X_K = explainer.X_train[idx_nn[:k]]
    dists_k = dists[:k]
    bw = float(np.median(dists_k))
    if bw == 0:
        bw = np.finfo(float).eps
    weights = kernel_weights(dists_k, bw, ANALYTICAL_KERNEL)
    return explainer._attribution(grad, X_K, weights)


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

    rng = np.random.default_rng(42)
    all_idx = rng.choice(len(X_test), size=min(N_TEST, len(X_test)), replace=False)
    tune_idx = all_idx[:N_TUNE]
    eval_idx = all_idx[N_TUNE:]
    print(f"Tune samples: {len(tune_idx)}  Eval samples: {len(eval_idx)}")

    X_tune = X_test[tune_idx]
    X_eval = X_test[eval_idx]
    mask_tune = masks_test[tune_idx]
    mask_eval = masks_test[eval_idx]

    lime_bw = 0.75 * np.sqrt(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    pat_bw = None
    print(f"LIME bw={lime_bw:.2f}  PatternLocal bw=auto  Analytical kernel={ANALYTICAL_KERNEL} "
          f"bw=auto  ks={ANALYTICAL_KS}  ig_steps={IG_STEPS}")
    format_result("lime_bandwidth", f"{lime_bw:.2f}")
    format_result("patternlocal_bandwidth", "auto")
    format_result("analytical_kernel", ANALYTICAL_KERNEL)
    format_result("ig_steps", f"{IG_STEPS}")

    # ---- build explainer + cache gradients/IG/knn -----------------------
    print("\n--- Building analytical explainer ---")
    max_k = max(ANALYTICAL_KS)
    explainer = AnalyticalPatternLocalExplainer(
        model, X_train, k_neighbors=max_k, device=DEVICE,
        normalize=False, kernel=ANALYTICAL_KERNEL, bandwidth=None,
    )

    def ig_grad(x_star, steps=IG_STEPS, seed=0):
        """Integrated Gradients: mean grad from baseline=0 to x."""
        x_np = np.asarray(x_star, dtype=np.float32).ravel()
        baseline = np.zeros_like(x_np)
        grads = np.zeros_like(x_np, dtype=np.float64)
        for s in range(1, steps + 1):
            alpha = s / steps
            interpolated = baseline + alpha * (x_np - baseline)
            grads += explainer.gradient(interpolated)
        grads /= steps
        return (x_np - baseline) * grads

    t0 = time.time()
    tune_grads = np.array([explainer.gradient(X_test[i]) for i in tune_idx])
    eval_grads = np.array([explainer.gradient(X_test[i]) for i in eval_idx])
    tune_igs = np.array([ig_grad(X_test[i]) for i in tune_idx])
    eval_igs = np.array([ig_grad(X_test[i]) for i in eval_idx])
    print(f"  grads+IG for {len(tune_grads)+len(eval_grads)} samples in {time.time()-t0:.1f}s")

    t0 = time.time()
    tune_dist, tune_idx_nn = explainer.knn.kneighbors(X_tune, n_neighbors=max_k)
    eval_dist, eval_idx_nn = explainer.knn.kneighbors(X_eval, n_neighbors=max_k)
    print(f"  KNN cached ({max_k}) in {time.time()-t0:.1f}s")

    # ---- tune analytical-gradient k -------------------------------
    print("\n--- Tuning Analytical-gradient k ---")
    best_grad_k = (None, np.inf)
    for k in ANALYTICAL_KS:
        emds = []
        for j, i in enumerate(tune_idx):
            a = analytic_attr(explainer, X_test[i], tune_grads[j],
                              tune_dist[j], tune_idx_nn[j], k)
            e, _ = _score(_normalize(a), mask_tune[j])
            emds.append(e)
        m = float(np.mean(emds))
        print(f"  grad k={k:5d}  EMD={m:.4f}")
        if m < best_grad_k[1]:
            best_grad_k = (k, m)
    print(f"  -> grad k = {best_grad_k[0]}")
    format_result("analytical_grad_k", f"{best_grad_k[0]}")

    # ---- tune analytical-IG k ---------------------------------------
    print("\n--- Tuning Analytical-IG k ---")
    best_ig_k = (None, np.inf)
    for k in ANALYTICAL_KS:
        emds = []
        for j, i in enumerate(tune_idx):
            a = analytic_attr(explainer, X_test[i], tune_igs[j],
                              tune_dist[j], tune_idx_nn[j], k)
            e, _ = _score(_normalize(a), mask_tune[j])
            emds.append(e)
        m = float(np.mean(emds))
        print(f"  ig k={k:5d}  EMD={m:.4f}")
        if m < best_ig_k[1]:
            best_ig_k = (k, m)
    print(f"  -> ig k = {best_ig_k[0]}")
    format_result("analytical_ig_k", f"{best_ig_k[0]}")

    # ---- final comparison on eval split -------------------------------
    print("\n--- Final evaluation ---")
    lime_emds, lime_imes = [], []
    pat_emds, pat_imes = [], []
    grad_emds, grad_imes = [], []
    ig_emds, ig_imes = [], []
    t0 = time.time()
    for j, i in enumerate(eval_idx):
        x = X_eval[j]
        mask = mask_eval[j]

        v, b = lime_explain(model, x, bandwidth=lime_bw, seed=int(i))
        e, im = _score(_normalize(v), mask)
        lime_emds.append(e)
        lime_imes.append(im)

        p = pattern_explain(x, v, b, X_train, pat_bw, K_RATIO, 0.0)
        e, im = _score(_normalize(p), mask)
        pat_emds.append(e)
        pat_imes.append(im)

        gk = best_grad_k[0]
        ag = analytic_attr(explainer, x, eval_grads[j],
                           eval_dist[j], eval_idx_nn[j], gk)
        e, im = _score(_normalize(ag), mask)
        grad_emds.append(e)
        grad_imes.append(im)

        ik = best_ig_k[0]
        aig = analytic_attr(explainer, x, eval_igs[j],
                            eval_dist[j], eval_idx_nn[j], ik)
        e, im = _score(_normalize(aig), mask)
        ig_emds.append(e)
        ig_imes.append(im)
    print(f"  evaluated {len(eval_idx)} samples in {time.time()-t0:.1f}s")

    lime_emd, lime_ime = float(np.mean(lime_emds)), float(np.mean(lime_imes))
    pat_emd, pat_ime = float(np.mean(pat_emds)), float(np.mean(pat_imes))
    grad_emd, grad_ime = float(np.mean(grad_emds)), float(np.mean(grad_imes))
    ig_emd, ig_ime = float(np.mean(ig_emds)), float(np.mean(ig_imes))

    print(f"\nResults over {len(eval_idx)} test samples:")
    print(f"  LIME           EMD: {lime_emd:.4f} +/- {np.std(lime_emds):.4f}")
    print(f"  LIME           IME: {lime_ime:.4f} +/- {np.std(lime_imes):.4f}")
    print(f"  PatternLocal   EMD: {pat_emd:.4f} +/- {np.std(pat_emds):.4f}")
    print(f"  PatternLocal   IME: {pat_ime:.4f} +/- {np.std(pat_imes):.4f}")
    print(f"  Analytical-grad EMD: {grad_emd:.4f} +/- {np.std(grad_emds):.4f}")
    print(f"  Analytical-grad IME: {grad_ime:.4f} +/- {np.std(grad_imes):.4f}")
    print(f"  Analytical-IG   EMD: {ig_emd:.4f} +/- {np.std(ig_emds):.4f}")
    print(f"  Analytical-IG   IME: {ig_ime:.4f} +/- {np.std(ig_imes):.4f}")
    print(f"\n  Analytical-IG EMD improvement vs LIME: {lime_emd - ig_emd:+.4f}")
    print(f"  Analytical-IG IME improvement vs LIME: {lime_ime - ig_ime:+.4f}")
    print(f"  Analytical-IG EMD improvement vs grad: {grad_emd - ig_emd:+.4f}")

    for k, v in {
        "lime_emd_mean": f"{lime_emd:.4f}", "lime_ime_mean": f"{lime_ime:.4f}",
        "patternlocal_emd_mean": f"{pat_emd:.4f}", "patternlocal_ime_mean": f"{pat_ime:.4f}",
        "analytical_grad_emd_mean": f"{grad_emd:.4f}", "analytical_grad_ime_mean": f"{grad_ime:.4f}",
        "analytical_ig_emd_mean": f"{ig_emd:.4f}", "analytical_ig_ime_mean": f"{ig_ime:.4f}",
        "analytical_ig_emd_std": f"{np.std(ig_emds):.4f}",
        "analytical_ig_ime_std": f"{np.std(ig_imes):.4f}",
        "analytical_grad_k": best_grad_k[0],
        "analytical_ig_k": best_ig_k[0],
        "ig_better_emd_vs_lime": bool(ig_emd < lime_emd),
        "ig_better_ime_vs_lime": bool(ig_ime < lime_ime),
        "ig_better_emd_vs_grad": bool(ig_emd < grad_emd),
    }.items():
        format_result(k, v)

    passed = bool(ig_emd < lime_emd)
    format_result("PASS" if passed else "FAIL", "analytical-IG vs LIME (EMD)")
    format_result("total_seconds", f"{time.time()-t_start:.1f}")

    results = {
        "config": CONFIG,
        "test_accuracy": test_acc,
        "n_eval": int(len(eval_idx)),
        "device": DEVICE,
        "lime_bandwidth": float(lime_bw),
        "analytical_kernel": ANALYTICAL_KERNEL,
        "analytical_grad_k": best_grad_k[0],
        "analytical_ig_k": best_ig_k[0],
        "ig_steps": IG_STEPS,
        "lime": {"emd_mean": lime_emd, "ime_mean": lime_ime},
        "patternlocal": {"emd_mean": pat_emd, "ime_mean": pat_ime},
        "analytical_grad": {"emd_mean": grad_emd, "ime_mean": grad_ime},
        "analytical_integrated_gradients": {
            "emd_mean": ig_emd, "ime_mean": ig_ime,
            "emd_std": float(np.std(ig_emds)), "ime_std": float(np.std(ig_imes)),
        },
        "passed": passed,
    }
    save_results(results)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
