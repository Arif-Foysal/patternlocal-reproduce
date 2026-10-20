"""Local hyperparameter probe on the FIXED XOR generator.

Trains one small XOR WHITE model, then evaluates LIME / PatternLocal across
bandwidths to see which settings reproduce the paper's trend
(PatternLocal < LIME on XOR).

Run with:  /tmp/xai_venv/bin/python -u scripts/local_hparam_probe.py
"""
import sys, time, json
sys.path.insert(0, 'src/paper_repro')

import numpy as np
import torch

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model
from xai.lime import lime_explain
from xai.patternlocal import patternlocal_explain
from metrics.emd import compute_emd
from metrics.ime import compute_ime

N = 10000
N_EVAL = 20
ALPHA = 0.9
NOISE = "WHITE"
DEVICE = "cpu"


def norm(w):
    a = np.abs(w)
    return w / a.max() if a.max() else w


def score(attr, mask):
    return compute_emd(attr, mask), compute_ime(attr, mask)


def main():
    print(f"generating {NOISE} alpha={ALPHA} n={N}", flush=True)
    X, y, masks = generate_xai_tris(n_samples=N, scenario="XOR", noise=NOISE,
                                    alpha=ALPHA, image_size=64, seed=7)
    (Xtr, ytr, _), (Xval, yval, _), (Xte, yte, mte) = split_dataset(X, y, masks, seed=7)
    model = MLPModel(input_dim=4096)
    model, hist, bv = train_mlp(model, Xtr, ytr, Xval, yval, lr=1e-4,
                                batch_size=128, max_epochs=60, patience=20, seed=7)
    print("test acc", evaluate_model(model, Xte, yte), flush=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), size=N_EVAL, replace=False)

    # ---- LIME bandwidth sweep (weights reused for PatternLocal) ----
    lime_bws = [2.0, 5.0, 10.0, 20.0, 30.0, 48.0]
    lime_scores = {bw: [] for bw in lime_bws}
    lime_weights = {bw: [] for bw in lime_bws}
    t0 = time.time()
    for j, i in enumerate(idx):
        for bw in lime_bws:
            v, b = lime_explain(model, Xte[i], n_samples=2000, bandwidth=bw,
                                seed=int(i), device=DEVICE)
            lime_scores[bw].append(score(norm(v), mte[i]))
            lime_weights[bw].append((v, b))
        if (j + 1) % 5 == 0:
            print(f"  lime {j+1}/{N_EVAL} ({time.time()-t0:.0f}s)", flush=True)

    print("\n== LIME bandwidth sweep ==", flush=True)
    for bw in lime_bws:
        a = np.array(lime_scores[bw])
        print(f"  bw={bw:5.1f}  EMD {a[:,0].mean():.4f}  IME {a[:,1].mean():.4f}", flush=True)

    # ---- PatternLocal sweep on LIME weights at best-ish bandwidth 30 ----
    ref_bw = 30.0
    print("\n== PatternLocal sweep (LIME bw=30) ==", flush=True)
    results = {}
    t0 = time.time()
    for kr in [0.005, 0.02, 0.05, 0.2]:
        for bwmode in ["auto", 0.25, 0.5, 1.0, 2.0]:
            for lam in [1e-5, 1e-2]:
                key = f"k{kr}_bw{bwmode}_lam{lam}"
                vals = []
                for j, i in enumerate(idx):
                    v, b = lime_weights[ref_bw][j]
                    pl = patternlocal_explain(
                        v, b, Xte[i], Xtr, bandwidth=None if bwmode == "auto" else bwmode,
                        kernel="gaussian", k_ratio=kr, lam=lam)
                    vals.append(score(norm(pl), mte[i]))
                a = np.array(vals)
                results[key] = (a[:, 0].mean(), a[:, 1].mean())
                print(f"  {key:28s} EMD {a[:,0].mean():.4f}  IME {a[:,1].mean():.4f}", flush=True)
    print(f"patternlocal sweep took {time.time()-t0:.0f}s", flush=True)

    best = sorted(results.items(), key=lambda kv: kv[1][0])[:5]
    print("\nBEST PatternLocal:", json.dumps(best, indent=1), flush=True)


if __name__ == "__main__":
    main()
