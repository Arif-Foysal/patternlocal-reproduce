"""Local smoke test: does the fixed XOR generator restore the paper's trend?

Generates a small XOR WHITE dataset, trains the paper MLP, and compares
LIME / PatternLocal / Analytical-IG with EMD and IME.
"""
import sys, time
sys.path.insert(0, 'src/paper_repro')

import numpy as np
import torch

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model
from xai.lime import lime_explain, model_logits
from xai.patternlocal import patternlocal_explain
from xai.analytical_patternlocal import AnalyticalPatternLocalExplainer, kernel_weights
from metrics.emd import compute_emd
from metrics.ime import compute_ime

N_SAMPLES = 8000
N_EVAL = 15
ALPHAS = [0.9]
LIME_N = 2000
IG_STEPS = 30
DEVICE = "cpu"


def normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


def score(attr, mask):
    return compute_emd(attr, mask), compute_ime(attr, mask)


def run_alpha(alpha, seed=0):
    print(f"\n{'='*60}\nXOR WHITE alpha={alpha}\n{'='*60}")
    X, y, masks = generate_xai_tris(
        n_samples=N_SAMPLES, scenario="XOR", noise="WHITE",
        alpha=alpha, image_size=64, seed=seed,
    )
    (Xtr, ytr, _), (Xval, yval, _), (Xte, yte, mte) = split_dataset(X, y, masks, seed=seed)
    print(f"train={Xtr.shape} test={Xte.shape} class_balance={yte.mean():.3f}")

    model = MLPModel(input_dim=4096)
    t0 = time.time()
    model, hist, best_val = train_mlp(
        model, Xtr, ytr, Xval, yval,
        lr=1e-4, batch_size=128, max_epochs=50, patience=15, seed=seed,
    )
    acc = evaluate_model(model, Xte, yte)
    print(f"test_acc={acc:.4f} (best_val={best_val:.4f}) train_time={time.time()-t0:.1f}s")

    expl = AnalyticalPatternLocalExplainer(model, Xtr, k_neighbors=300, device=DEVICE)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xte), size=N_EVAL, replace=False)

    res = {m: [] for m in ["lime", "patternlocal", "analytical_ig"]}
    t0 = time.time()
    for j, i in enumerate(idx):
        x = Xte[i]
        mask = mte[i]
        v, b = lime_explain(model, x, n_samples=LIME_N, seed=int(i), device=DEVICE)
        res["lime"].append(score(normalize(v), mask))

        pl = patternlocal_explain(v, b, x, Xtr, k_ratio=0.05, lam=1e-5)
        res["patternlocal"].append(score(normalize(pl), mask))

        ig = expl.integrated_gradients(x, steps=IG_STEPS)
        dists, idx_nn = expl.knn.kneighbors(np.asarray(x, dtype=np.float32).reshape(1, -1))
        dists = np.asarray(dists).ravel(); idx_nn = np.asarray(idx_nn).ravel()
        k = 300
        X_K = Xtr[idx_nn[:k]]; dk = dists[:k]
        bw = float(np.median(dk)) or np.finfo(float).eps
        wts = kernel_weights(dk, bw, "gaussian")
        aig = expl._attribution(ig, X_K, wts)
        res["analytical_ig"].append(score(normalize(aig), mask))

        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{N_EVAL} eval ({time.time()-t0:.1f}s)")

    print(f"\n{'method':16s} {'EMD':>8s} {'IME':>8s}")
    for m, vals in res.items():
        arr = np.array(vals)
        print(f"{m:16s} {arr[:,0].mean():8.4f} {arr[:,1].mean():8.4f}")


if __name__ == "__main__":
    for a in ALPHAS:
        run_alpha(a, seed=int(a * 100))
