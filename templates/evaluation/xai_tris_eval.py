#!/usr/bin/env python3
"""Single-config evaluation: PatternLocal vs LIME on XAI-TRIS.

Reproduces the phase-3 gate: on one configuration (XOR CORR alpha=0.2),
does PatternLocal beat LIME on EMD and IME?

Methodology (paper, Sec. 4 / App. D.2):
  * LIME is a weighted local-linear surrogate over binary perturbations of the
    identity representation (Eq. 5).
  * PatternLocal transforms the LIME weights with the local covariance (Eq. 7).
  * Hyperparameters are selected on a held-out split by minimizing EMD, then
    the methods are compared on the remaining test samples.

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
from metrics.emd import compute_emd
from metrics.ime import compute_ime

CONFIG = "XOR_CORR_alpha0.2"
IMAGE_SHAPE = (64, 64)

# Evaluation sizing. Kept modest so the whole run fits comfortably in a Kaggle
# notebook; increase N_TEST for the full gate.
N_TUNE = 30
N_TEST = 150
N_LIME = 2000          # perturbed samples per LIME fit
K_RATIO = 0.05         # local neighbourhood for the PatternLocal covariance
SHRINKAGE = 0.0        # covariance shrinkage intensity

# Small validation grids. LIME's kernel acts on the binary perturbation space
# (scale ~ sqrt(D)); PatternLocal's acts on the continuous input (scale ~ 1).
D = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
LIME_BWS = [0.375 * np.sqrt(D), 0.75 * np.sqrt(D), 1.5 * np.sqrt(D)]
PAT_BWS = [None, 1.0, 5.0]


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def save_results(results_dict, path="/kaggle/working/eval_results.json"):
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {path}")


def find_file(pattern):
    matches = glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True)
    return matches[0] if matches else None


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


def _normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


def _score(expl, mask):
    return compute_emd(expl, mask, image_shape=IMAGE_SHAPE), compute_ime(expl, mask)


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Single-Config Evaluation: {CONFIG}")
    print("=" * 60)

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

    # ---- final comparison on the held-out eval split -----------------------
    print("\n--- Final evaluation ---")
    lime_emds, lime_imes, pat_emds, pat_imes = [], [], [], []
    t0 = time.time()
    for i in eval_idx:
        x = X_test[i]
        mask = masks_test[i]

        v, b = lime_explain(model, x, bandwidth=best_lime[0], seed=int(i))
        e, im = _score(_normalize(v), mask)
        lime_emds.append(e)
        lime_imes.append(im)

        p = pattern_explain(x, v, b, X_train, best_pat[0], K_RATIO, SHRINKAGE)
        e, im = _score(_normalize(p), mask)
        pat_emds.append(e)
        pat_imes.append(im)
    print(f"  evaluated {len(eval_idx)} samples in {time.time()-t0:.1f}s")

    lime_emd, lime_ime = float(np.mean(lime_emds)), float(np.mean(lime_imes))
    pat_emd, pat_ime = float(np.mean(pat_emds)), float(np.mean(pat_imes))

    print(f"\nResults over {len(eval_idx)} test samples:")
    print(f"  LIME          EMD: {lime_emd:.4f} +/- {np.std(lime_emds):.4f}")
    print(f"  LIME          IME: {lime_ime:.4f} +/- {np.std(lime_imes):.4f}")
    print(f"  PatternLocal  EMD: {pat_emd:.4f} +/- {np.std(pat_emds):.4f}")
    print(f"  PatternLocal  IME: {pat_ime:.4f} +/- {np.std(pat_imes):.4f}")
    print(f"\n  PatternLocal EMD improvement: {lime_emd - pat_emd:+.4f}")
    print(f"  PatternLocal IME improvement: {lime_ime - pat_ime:+.4f}")

    for k, v in {
        "lime_emd_mean": f"{lime_emd:.4f}", "lime_emd_std": f"{np.std(lime_emds):.4f}",
        "lime_ime_mean": f"{lime_ime:.4f}", "lime_ime_std": f"{np.std(lime_imes):.4f}",
        "patternlocal_emd_mean": f"{pat_emd:.4f}", "patternlocal_emd_std": f"{np.std(pat_emds):.4f}",
        "patternlocal_ime_mean": f"{pat_ime:.4f}", "patternlocal_ime_std": f"{np.std(pat_imes):.4f}",
        "patternlocal_better_emd": bool(pat_emd < lime_emd),
        "patternlocal_better_ime": bool(pat_ime < lime_ime),
    }.items():
        format_result(k, v)

    passed = (pat_emd < lime_emd) and (pat_ime < lime_ime)
    format_result("PASS" if passed else "FAIL", "phase-3 gate")
    format_result("total_seconds", f"{time.time()-t_start:.1f}")

    results = {
        "config": CONFIG,
        "test_accuracy": test_acc,
        "n_eval": int(len(eval_idx)),
        "lime_bandwidth": None if best_lime[0] is None else float(best_lime[0]),
        "patternlocal_bandwidth": None if best_pat[0] is None else float(best_pat[0]),
        "lime": {"emd_mean": lime_emd, "emd_std": float(np.std(lime_emds)),
                 "ime_mean": lime_ime, "ime_std": float(np.std(lime_imes))},
        "patternlocal": {"emd_mean": pat_emd, "emd_std": float(np.std(pat_emds)),
                         "ime_mean": pat_ime, "ime_std": float(np.std(pat_imes))},
        "passed": bool(passed),
    }
    save_results(results)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
