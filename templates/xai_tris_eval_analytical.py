#!/usr/bin/env python3
"""Single-config evaluation: LIME vs PatternLocal vs Analytical PatternLocal.

Reproduces the phase-3 gate on XOR CORR alpha=0.2 and additionally evaluates
the analytical (exact-gradient + KNN-covariance) PatternLocal explainer:

  * LIME: weighted local-linear surrogate (paper Eq. 5).
  * PatternLocal: LIME weights transformed by the local covariance (Eq. 7).
  * Analytical PatternLocal: exact input gradient + KNN local covariance,
    O(K * D), no LIME fit, no bandwidth.

Hyperparameters are selected on a held-out split by minimizing EMD, then the
methods are compared on the remaining test samples.

NOTE: the cached XAI-TRIS archives store masks only for the test split, so the
tuning split is drawn from the test set and the final comparison uses the
complement. The proper fix is to persist masks_val in the data phase.
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
from xai.analytical_patternlocal import AnalyticalPatternLocalExplainer
from metrics.emd import compute_emd
from metrics.ime import compute_ime

CONFIG = "XOR_CORR_alpha0.2"
IMAGE_SHAPE = (64, 64)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Evaluation sizing. Kept modest so the whole run fits comfortably in a Kaggle
# notebook; increase N_TEST for the full gate.
N_TUNE = 30
N_TEST = 150
N_LIME = 2000          # perturbed samples per LIME fit
K_RATIO = 0.05         # local neighbourhood for PatternLocal covariance
SHRINKAGE = 0.0

# Small validation grids.
D = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
LIME_BWS = [0.375 * np.sqrt(D), 0.75 * np.sqrt(D), 1.5 * np.sqrt(D)]
PAT_BWS = [None, 1.0, 5.0]
ANALYTICAL_KS = [50, 100, 200, 500]   # KNN neighbourhood sizes for analytical


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


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Single-Config Evaluation: {CONFIG}")
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

    model = MLPModel(input_dim=D)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_acc = float((model_logits(model, X_test).argmax(axis=1) == y_test).mean())
    print(f"Test accuracy: {test_acc:.4f}")
    format_result("test_accuracy", f"{test_acc:.4f}")

    # Held-out tuning split (val masks are not persisted in the archive).
    rng = np.random.default_rng(42)
    all_idx = rng.choice(len(X_test), size=min(N_TEST, len(X_test)), replace=False)
    tune_idx = all_idx[:N_TUNE]
    eval_idx = all_idx[N_TUNE:]
    print(f"Tune samples: {len(tune_idx)}  Eval samples: {len(eval_idx)}")

    X_tune = X_test[tune_idx]
    X_eval = X_test[eval_idx]
    mask_tune = masks_test[tune_idx]
    mask_eval = masks_test[eval_idx]

    # ---- tune LIME bandwidth on the tuning split ---------------------------
    print("\n--- Tuning LIME bandwidth ---")
    best_lime = (None, np.inf)
    for bw in LIME_BWS:
        emds = []
        t0 = time.time()
        for i in tune_idx:
            v, b = lime_explain(model, X_test[i], bandwidth=bw, seed=int(i))
            e, _ = _score(_normalize(v), masks_test[i])
            emds.append(e)
        m = float(np.mean(emds))
        print(f"  lime_bw={bw:7.2f}  EMD={m:.4f}  ({time.time()-t0:.1f}s)")
        if m < best_lime[1]:
            best_lime = (bw, m)
    print(f"  -> LIME bandwidth = {best_lime[0]:.2f}")
    format_result("lime_bandwidth", f"{best_lime[0]:.2f}")

    # ---- tune PatternLocal bandwidth on the tuning split -------------------
    print("\n--- Tuning PatternLocal bandwidth ---")
    best_pat = (None, np.inf)
    for bw in PAT_BWS:
        emds = []
        t0 = time.time()
        for i in tune_idx:
            v, b = lime_explain(model, X_test[i], bandwidth=best_lime[0], seed=int(i))
            p = pattern_explain(X_test[i], v, b, X_train, bw, K_RATIO, SHRINKAGE)
            e, _ = _score(_normalize(p), masks_test[i])
            emds.append(e)
        m = float(np.mean(emds))
        label = "auto" if bw is None else f"{bw:.2f}"
        print(f"  pat_bw={label:>7}  EMD={m:.4f}  ({time.time()-t0:.1f}s)")
        if m < best_pat[1]:
            best_pat = (bw, m)
    print(f"  -> PatternLocal bandwidth = {'auto' if best_pat[0] is None else best_pat[0]:}")
    format_result("patternlocal_bandwidth", "auto" if best_pat[0] is None else f"{best_pat[0]:.2f}")

    # ---- compute exact gradients once (shared by analytical) --------------
    print("\n--- Computing exact input gradients ---")
    max_k = max(ANALYTICAL_KS)
    explainer = AnalyticalPatternLocalExplainer(
        model, X_train, k_neighbors=max_k, device=DEVICE, normalize=False
    )
    t0 = time.time()
    tune_grads = np.array([explainer.gradient(X_test[i]) for i in tune_idx])
    eval_grads = np.array([explainer.gradient(X_test[i]) for i in eval_idx])
    print(f"  {len(tune_grads)+len(eval_grads)} gradients in {time.time()-t0:.1f}s")

    # ---- tune Analytical PatternLocal k_neighbors on the tuning split ------
    print("\n--- Tuning Analytical PatternLocal k_neighbors ---")
    best_analytical = (None, np.inf)
    for k in ANALYTICAL_KS:
        _, indices = explainer.knn.kneighbors(X_tune, n_neighbors=k)
        emds = []
        for j, i in enumerate(tune_idx):
            a = explainer.attribution(tune_grads[j], explainer.X_train[indices[j]])
            e, _ = _score(_normalize(a), mask_tune[j])
            emds.append(e)
        m = float(np.mean(emds))
        print(f"  analytical_k={k:5d}  EMD={m:.4f}")
        if m < best_analytical[1]:
            best_analytical = (k, m)
    print(f"  -> Analytical k_neighbors = {best_analytical[0]}")
    format_result("analytical_k_neighbors", f"{best_analytical[0]}")

    # ---- final comparison on the held-out eval split -----------------------
    print("\n--- Final evaluation ---")
    lime_emds, lime_imes = [], []
    pat_emds, pat_imes = [], []
    ana_emds, ana_imes = [], []
    t0 = time.time()
    for j, i in enumerate(eval_idx):
        x = X_eval[j]
        mask = mask_eval[j]

        # LIME
        v, b = lime_explain(model, x, bandwidth=best_lime[0], seed=int(i))
        e, im = _score(_normalize(v), mask)
        lime_emds.append(e)
        lime_imes.append(im)

        # PatternLocal
        p = pattern_explain(x, v, b, X_train, best_pat[0], K_RATIO, SHRINKAGE)
        e, im = _score(_normalize(p), mask)
        pat_emds.append(e)
        pat_imes.append(im)

        # Analytical PatternLocal
        _, a_indices = explainer.knn.kneighbors(x.reshape(1, -1), n_neighbors=best_analytical[0])
        ana = explainer.attribution(eval_grads[j], explainer.X_train[a_indices[0]])
        e, im = _score(_normalize(ana), mask)
        ana_emds.append(e)
        ana_imes.append(im)
    print(f"  evaluated {len(eval_idx)} samples in {time.time()-t0:.1f}s")

    lime_emd, lime_ime = float(np.mean(lime_emds)), float(np.mean(lime_imes))
    pat_emd, pat_ime = float(np.mean(pat_emds)), float(np.mean(pat_imes))
    ana_emd, ana_ime = float(np.mean(ana_emds)), float(np.mean(ana_imes))

    print(f"\nResults over {len(eval_idx)} test samples:")
    print(f"  LIME          EMD: {lime_emd:.4f} +/- {np.std(lime_emds):.4f}")
    print(f"  LIME          IME: {lime_ime:.4f} +/- {np.std(lime_imes):.4f}")
    print(f"  PatternLocal  EMD: {pat_emd:.4f} +/- {np.std(pat_emds):.4f}")
    print(f"  PatternLocal  IME: {pat_ime:.4f} +/- {np.std(pat_imes):.4f}")
    print(f"  Analytical    EMD: {ana_emd:.4f} +/- {np.std(ana_emds):.4f}")
    print(f"  Analytical    IME: {ana_ime:.4f} +/- {np.std(ana_imes):.4f}")
    print(f"\n  Analytical EMD improvement vs LIME: {lime_emd - ana_emd:+.4f}")
    print(f"  Analytical IME improvement vs LIME: {lime_ime - ana_ime:+.4f}")
    print(f"  Analytical EMD improvement vs PL:   {pat_emd - ana_emd:+.4f}")
    print(f"  Analytical IME improvement vs PL:   {pat_ime - ana_ime:+.4f}")

    for k, v in {
        "lime_emd_mean": f"{lime_emd:.4f}", "lime_emd_std": f"{np.std(lime_emds):.4f}",
        "lime_ime_mean": f"{lime_ime:.4f}", "lime_ime_std": f"{np.std(lime_imes):.4f}",
        "patternlocal_emd_mean": f"{pat_emd:.4f}", "patternlocal_emd_std": f"{np.std(pat_emds):.4f}",
        "patternlocal_ime_mean": f"{pat_ime:.4f}", "patternlocal_ime_std": f"{np.std(pat_imes):.4f}",
        "analytical_emd_mean": f"{ana_emd:.4f}", "analytical_emd_std": f"{np.std(ana_emds):.4f}",
        "analytical_ime_mean": f"{ana_ime:.4f}", "analytical_ime_std": f"{np.std(ana_imes):.4f}",
        "analytical_better_emd_vs_lime": bool(ana_emd < lime_emd),
        "analytical_better_ime_vs_lime": bool(ana_ime < lime_ime),
        "analytical_better_emd_vs_pl": bool(ana_emd < pat_emd),
        "analytical_better_ime_vs_pl": bool(ana_ime < pat_ime),
    }.items():
        format_result(k, v)

    passed = (ana_emd < lime_emd) and (ana_ime < lime_ime)
    format_result("PASS" if passed else "FAIL", "analytical phase-3 gate")
    format_result("total_seconds", f"{time.time()-t_start:.1f}")

    results = {
        "config": CONFIG,
        "test_accuracy": test_acc,
        "n_eval": int(len(eval_idx)),
        "device": DEVICE,
        "lime_bandwidth": None if best_lime[0] is None else float(best_lime[0]),
        "patternlocal_bandwidth": None if best_pat[0] is None else float(best_pat[0]),
        "analytical_k_neighbors": best_analytical[0],
        "lime": {"emd_mean": lime_emd, "emd_std": float(np.std(lime_emds)),
                 "ime_mean": lime_ime, "ime_std": float(np.std(lime_imes))},
        "patternlocal": {"emd_mean": pat_emd, "emd_std": float(np.std(pat_emds)),
                         "ime_mean": pat_ime, "ime_std": float(np.std(pat_imes))},
        "analytical": {"emd_mean": ana_emd, "emd_std": float(np.std(ana_emds)),
                       "ime_mean": ana_ime, "ime_std": float(np.std(ana_imes))},
        "passed": bool(passed),
    }
    save_results(results)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
