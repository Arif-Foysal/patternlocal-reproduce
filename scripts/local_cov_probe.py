"""Decisive probe: does PatternLocal's LOCAL covariance kill the signal?

Hypothesis: with a k-NN-only neighborhood (k_ratio <= 0.2) all neighbours share
the identical class-conditional signal, so the local covariance has (nearly)
zero variance along the signal direction -> a = C w projects the LIME weights
onto the noise subspace and destroys attribution. The paper's bandwidth range
is huge relative to image distances, i.e. effectively a (near-)GLOBAL
covariance, which retains signal variance and amplifies the support.

Fixtures: XOR x {WHITE, CORR} x alpha {0.5, 0.9}, 4 eval samples each.

Run: /tmp/xai_venv/bin/python -u scripts/local_cov_probe.py
"""
import sys
import time

sys.path.insert(0, "src/paper_repro")

import numpy as np

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model
from xai.lime import lime_explain
from xai.patternlocal import patternlocal_explain
from metrics.emd import compute_emd
from metrics.ime import compute_ime

N = 4000
N_EVAL = 4
LIME_BW = 20.0

CONFIGS = [
    ("local k0.02 auto", dict(k_ratio=0.02, bandwidth=None, use_global_cov=False)),
    ("local k0.10 auto", dict(k_ratio=0.10, bandwidth=None, use_global_cov=False)),
    ("local k1.00 auto", dict(k_ratio=1.00, bandwidth=None, use_global_cov=False)),
    ("local k1.00 bw1.8", dict(k_ratio=1.00, bandwidth=1.8, use_global_cov=False)),
    ("local k1.00 bw10", dict(k_ratio=1.00, bandwidth=10.0, use_global_cov=False)),
    ("global cov", dict(use_global_cov=True)),
]


def norm(w):
    a = np.abs(w)
    return w / a.max() if a.max() else w


def score(attr, mask):
    return compute_emd(attr, mask), compute_ime(attr, mask)


def run(scenario, noise, alpha):
    print(f"\n=== {scenario} {noise} alpha={alpha} ===", flush=True)
    X, y, masks = generate_xai_tris(n_samples=N, scenario=scenario,
                                    noise=noise, alpha=alpha,
                                    image_size=64, seed=21)
    (Xtr, ytr, _), (Xval, yval, _), (Xte, yte, mte) = split_dataset(
        X, y, masks, seed=21)
    model = MLPModel(input_dim=4096)
    model, _, _ = train_mlp(model, Xtr, ytr, Xval, yval, lr=1e-4,
                            batch_size=128, max_epochs=60, patience=15, seed=21)
    acc = evaluate_model(model, Xte, yte)
    print(f"  n_train={len(Xtr)} test_acc={acc:.3f}", flush=True)

    rng = np.random.default_rng(3)
    idx = rng.choice(len(Xte), size=min(N_EVAL, len(Xte)), replace=False)

    lime_scores, lime_ws = [], []
    for i in idx:
        v, b = lime_explain(model, Xte[i], n_samples=2000,
                            bandwidth=LIME_BW, seed=int(i))
        lime_scores.append(score(norm(v), mte[i]))
        lime_ws.append((v, b))
    a = np.array(lime_scores)
    print(f"  LIME bw={LIME_BW:4.1f}          EMD {a[:,0].mean():.4f}  "
          f"IME {a[:,1].mean():.4f}", flush=True)

    for name, kw in CONFIGS:
        vals = []
        t0 = time.time()
        for j, i in enumerate(idx):
            v, b = lime_ws[j]
            pl = patternlocal_explain(v, b, Xte[i], Xtr, **kw)
            vals.append(score(norm(pl), mte[i]))
        a = np.array(vals)
        print(f"  {name:18s} EMD {a[:,0].mean():.4f}  IME {a[:,1].mean():.4f}"
              f"   ({(time.time()-t0)/len(idx):.2f}s/sample)", flush=True)


def main():
    for noise in ("WHITE", "CORR"):
        for alpha in (0.5, 0.9):
            run("XOR", noise, alpha)


if __name__ == "__main__":
    main()
