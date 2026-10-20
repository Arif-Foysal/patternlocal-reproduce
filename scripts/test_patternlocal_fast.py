"""Equivalence test: fast PatternLocal (no explicit C) vs naive float64 C.

Run: /tmp/xai_venv/bin/python -u scripts/test_patternlocal_fast.py
"""
import sys
import time

sys.path.insert(0, "src/paper_repro")

import numpy as np

from xai.patternlocal import patternlocal_explain, kernel_weights
from scipy.spatial.distance import cdist


def naive(lime_weights, instance, X_train, bandwidth, kernel, k_ratio, lam,
          use_global_cov, shrinkage):
    """Reference implementation: build C explicitly in float64."""
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    if use_global_cov:
        X_local, v = X, np.full(n, 1.0 / n)
    else:
        k = max(1, int(k_ratio * n))
        dist = cdist(X, instance.reshape(1, -1)).ravel()
        k_idx = np.argsort(dist)[:k]
        dk = dist[k_idx]
        if bandwidth is None:
            bandwidth = np.median(dk)
        v = kernel_weights(dk, bandwidth, kernel)
        v = v / v.sum()
        X_local = X[k_idx]
    mu = np.average(X_local, weights=v, axis=0)
    diffs = X_local - mu
    C = (v[:, None] * diffs).T @ diffs
    if shrinkage > 0:
        avg = float(np.mean(np.diag(C)))
        C = (1 - shrinkage) * C + shrinkage * avg * np.eye(d)
    Cw = C @ lime_weights
    return Cw / (lime_weights @ Cw + lam)


def main():
    rng = np.random.default_rng(0)
    N, D = 3000, 256
    X = rng.standard_normal((N, D)).astype(np.float32)
    # make data look like XAI-TRIS: unit-norm rows, a shared signal direction
    sig = np.zeros(D)
    sig[20:40] = 1.0
    sig /= np.linalg.norm(sig)
    signs = rng.choice([-1.0, 1.0], N).astype(np.float32)
    X = 0.9 * signs[:, None] * sig[None, :] + 0.1 * X
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    inst = X[7]

    w = rng.standard_normal(D)
    cases = [
        dict(k_ratio=0.05, bandwidth=None, use_global_cov=False, shrinkage=0.0),
        dict(k_ratio=0.5, bandwidth=None, use_global_cov=False, shrinkage=0.0),
        dict(k_ratio=1.0, bandwidth=None, use_global_cov=False, shrinkage=0.0),
        dict(k_ratio=1.0, bandwidth=1.8, use_global_cov=False, shrinkage=0.0),
        dict(k_ratio=1.0, bandwidth=0.5, kernel="epanechnikov", shrinkage=0.0),
        dict(k_ratio=1.0, bandwidth=None, use_global_cov=False, shrinkage=0.3),
        dict(use_global_cov=True, shrinkage=0.0),
        dict(use_global_cov=True, shrinkage=0.5),
    ]
    ok = True
    for c in cases:
        c = dict(c)
        fast = patternlocal_explain(w, 0.0, inst, X, lam=1e-5, **c)
        ref = naive(w, inst, X, c.get("bandwidth"), c.get("kernel", "gaussian"),
                    c.get("k_ratio", 1.0), 1e-5, c.get("use_global_cov", False),
                    c.get("shrinkage", 0.0))
        if np.abs(ref).max() < 1e-12:
            # Degenerate: the kernel collapses to a single (self) neighbour, so
            # the reference covariance is exactly zero.  The fast path keeps a
            # ~1e-7 float32 round-off that lambda then amplifies; both outputs
            # are meaningless, so only check that they are finite and small.
            good = np.all(np.isfinite(fast)) and np.abs(fast).max() < 1e-2
            print(f"{'DEG' if good else 'FAIL'} (degenerate kernel) "
                  f"{str(c)[:60]:60s} |fast|={np.abs(fast).max():.3e}", flush=True)
            ok = ok and good
            continue
        denom = max(np.abs(ref).max(), 1e-12)
        rel = np.abs(fast - ref).max() / denom
        status = "OK " if rel < 1e-4 else "FAIL"
        if rel >= 1e-4:
            ok = False
        print(f"{status} {str(c)[:78]:78s} rel_err={rel:.2e}", flush=True)

    # timing at full XAI-TRIS scale (N=40000, D=4096)
    print("\ntiming at N=40000, D=4096 ...", flush=True)
    Xb = rng.standard_normal((40000, 4096)).astype(np.float32)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    wb = rng.standard_normal(4096)
    ib = Xb[0]
    for kr in (0.05, 1.0):
        t0 = time.time()
        for _ in range(3):
            patternlocal_explain(wb, 0.0, ib, Xb, k_ratio=kr, bandwidth=None)
        print(f"  k_ratio={kr}: {(time.time()-t0)/3:.3f}s/sample", flush=True)
    print("OK" if ok else "FAILURES", flush=True)


if __name__ == "__main__":
    main()
