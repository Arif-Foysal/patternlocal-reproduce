"""Reusable XAI-TRIS XOR CORR evaluation core.

Shared by the SNR-sweep notebooks so the same evaluation code isn't duplicated
across them. Each notebook calls :func:`evaluate_alpha` for its assigned
alpha(s).

Computes, per test instance:
  * LIME — weighted local-linear surrogate (paper Eq. 5)
  * PatternLocal — LIME weights transformed by local covariance (Eq. 7)
  * Analytical-gradient — exact input gradient + KNN covariance
  * Analytical-IG — Integrated Gradients + KNN covariance (proposed method)
  * Plain Gradient / Plain IG / Plain SmoothGrad — no-KNN ablations
  * Saliency, DeepLift (Rescale), GradientShap, GuidedBackprop, Sobel, Laplace
    — the baseline set of paper Sec. 4.3

All explanations are compared against the ground-truth mask with EMD and IME.

For the first 5 eval instances per alpha, also saves:
  * input image, ground-truth mask, LIME attribution, PatternLocal attribution,
    Analytical-IG attribution (as PNGs and packed into an .npz).
"""

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# --- locate and import the core library (uploaded as the core dataset) -------
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "patternlocal.py")):
        sys.path.insert(0, root)
        break

from xai.lime import lime_explain, model_logits
from xai.analytical_patternlocal import (
    AnalyticalPatternLocalExplainer, kernel_weights,
)
from xai.patternlocal import patternlocal_explain
from metrics.emd import compute_emd
from metrics.ime import compute_ime

IMAGE_SHAPE = (64, 64)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Fixed analytical knobs (kernel tuning hurt EMD; gradient/IG surrogates are
# the variables).
ANALYTICAL_KERNEL = "gaussian"
ANALYTICAL_BANDWIDTH = None  # auto = median k-NN distance

# LIME / PatternLocal knobs.  The paper tunes LIME's bandwidth and
# PatternLocal's bandwidth / lambda / kernel on the validation split with TPE
# (200 trials).  We use a small deterministic proxy search on a held-out tuning
# subset: LIME bandwidth on the paper's [0.5, 30] range (upper end; smaller
# values underflow for D'=4096 binary masks) and PatternLocal's effective
# locality knob k_ratio (bandwidth is auto = median distance, lambda = 1e-5,
# kernel = gaussian, which are invariant after explanation normalization).
LIME_BANDWIDTH = None  # set by tune_lime_bandwidth()
LIME_BW_GRID = (10.0, 20.0, 30.0)
PAT_BANDWIDTH = None
# k_ratio = locality of the kernel-weighted covariance.  CRITICAL: at small k
# the k nearest neighbours all share the identical class-conditional signal, so
# the local covariance has ~zero variance along the signal direction and the
# transform annihilates the very attribution we want (EMD ends up no better
# than LIME).  k_ratio = 1.0 puts the kernel over the whole training set — the
# paper's Eq. (6)/(7) expectation over P_X, and the upstream default — which
# retains signal variance and is also ~10x cheaper with the gemv formulation.
PAT_K_GRID = (0.05, 0.5, 1.0)
K_RATIO = 0.05
SHRINKAGE = 0.0
IG_STEPS = 50
N_LIME = 2000

# EMD (POT network simplex) dominates runtime.  It releases the GIL, so scoring
# the methods of one instance on a small thread pool gives a ~2-2.5x speedup.
SCORE_WORKERS = int(os.environ.get("SCORE_WORKERS", "4"))
_SCORE_POOL = None


def _score_pool():
    global _SCORE_POOL
    if _SCORE_POOL is None:
        _SCORE_POOL = ThreadPoolExecutor(max_workers=SCORE_WORKERS)
    return _SCORE_POOL


# How many of the first eval instances to save as images + .npz per alpha.
N_SAVE_IMAGES = 5

# Output dir for sample images / npz (Kaggle working dir; overridable locally).
OUT_DIR = os.environ.get("EVAL_OUT_DIR", "/kaggle/working")

# Held-out tuning subset size (paper: TPE on the validation split).
TUNE_SAMPLES = 30

# Canonical evaluated methods: proposed (PatternLocal, Analytical-IG), the
# no-KNN ablations, and the paper's baseline set (Sec. 4.3): LIME, Integrated
# Gradients, Saliency, DeepLift, GradientShap, GuidedBackProp, Sobel, Laplace.
METHODS = [
    "lime", "patternlocal", "analytical_grad", "analytical_ig",
    "plain_grad", "plain_ig", "plain_smoothgrad",
    "saliency", "deeplift", "gradient_shap", "guided_backprop",
    "sobel", "laplace",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_python(o):
    """Recursively convert numpy scalars / arrays to JSON-serializable Python types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_python(v) for v in o]
    return o


def _score(expl, mask):
    return compute_emd(expl, mask, image_shape=IMAGE_SHAPE), compute_ime(expl, mask)


def _score_many(items):
    """Score a list of (attribution, mask) pairs, threaded when available."""
    if SCORE_WORKERS > 1 and len(items) > 1:
        return list(_score_pool().map(lambda it: _score(it[0], it[1]), items))
    return [_score(a, m) for a, m in items]


def _normalize(w):
    a = np.abs(w)
    return w / a.max() if a.max() > 0 else w


# Plain (no-KNN) baselines — isolated attribution methods, used to ablate
# whether gains come from the surrogate (gradient/IG/SmoothGrad) vs. the KNN
# covariance transform.
def _plain_gradient(ev, x):
    return ev.explainer.gradient(x)


def _plain_integrated_gradients(ev, x, steps=IG_STEPS):
    return ev.explainer.integrated_gradients(x, steps=steps)


def _plain_smoothgrad(ev, x, n_samples=50, noise_level=0.1, seed=0):
    return ev.explainer.smoothgrad(x, n_samples=n_samples,
                                   noise_level=noise_level, seed=seed)


# Paper baselines implemented in xai.baselines (no captum on the Kaggle image).
def _baseline_grad(ev):
    if getattr(ev, "_baseline_grad", None) is None:
        from xai.baselines import GradientBaselines
        ev._baseline_grad = GradientBaselines(
            ev.explainer.model, device=str(ev.explainer.device))
    return ev._baseline_grad


def _plain_saliency(ev, x):
    return _baseline_grad(ev).saliency(x)


def _plain_deeplift(ev, x):
    return _baseline_grad(ev).deeplift(x)


def _plain_gradient_shap(ev, x):
    return _baseline_grad(ev).gradient_shap(x)


def _plain_guided_backprop(ev, x):
    return _baseline_grad(ev).guided_backprop(x)


def _sobel(ev, x):
    from xai.baselines import sobel_explanation
    return sobel_explanation(np.asarray(x, dtype=np.float32).reshape(IMAGE_SHAPE))


def _laplace(ev, x):
    from xai.baselines import laplace_explanation
    return laplace_explanation(np.asarray(x, dtype=np.float32).reshape(IMAGE_SHAPE))


def pattern_from_lime(x, v, b, X_train, bandwidth, k_ratio, shrinkage):
    """PatternLocal from LIME weights (Eq. 7)."""
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


def _save_sample_images(alpha, idx_in_eval, x, mask, lime_v, pl, aig,
                        out_dir=None):
    """Save input/mask/attribution PNGs for one instance."""
    out_dir = OUT_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_tag = f"alpha{alpha:.1f}" if alpha is not None else "alpha_unknown"
    tag = f"{alpha_tag}_sample{idx_in_eval}"
    imgs = {
        "input": x.reshape(IMAGE_SHAPE),
        "mask": mask.reshape(IMAGE_SHAPE),
        "lime": _normalize(lime_v).reshape(IMAGE_SHAPE),
        "patternlocal": _normalize(pl).reshape(IMAGE_SHAPE),
        "analytical_ig": _normalize(aig).reshape(IMAGE_SHAPE),
    }
    for name, arr in imgs.items():
        fn = os.path.join(out_dir, f"{tag}_{name}.png")
        plt.imsave(fn, arr, cmap="hot", vmin=0, vmax=1)
    return tag


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class XORCorrEvaluator:
    """Holds a trained model + analytical explainer with cached KNN/gradients.

    Build once per alpha config; reuse across its test instances.
    """

    def __init__(self, model, X_train, k_neighbors=None, device=DEVICE):
        self.model = model
        self.X_train = X_train
        self.device = device
        if k_neighbors is None:
            k_neighbors = len(X_train)  # kernel over ALL training data
        self.explainer = AnalyticalPatternLocalExplainer(
            model, X_train, k_neighbors=k_neighbors, device=device,
            normalize=False, kernel=ANALYTICAL_KERNEL,
            bandwidth=ANALYTICAL_BANDWIDTH,
        )
        self.max_k = self.explainer.k_neighbors
        self._grad_cache = {}
        self._knn_cache = {}

    def gradient(self, x_star, seed=0):
        return self.explainer.gradient(x_star)

    def integrated_gradients(self, x_star, steps=IG_STEPS):
        return self.explainer.integrated_gradients(x_star, steps=steps)

    def knn(self, x_batch):
        """Return (distances, indices) for a batch.

        Cached by *content*: caching by ``id()`` was unsound — per-instance
        ``(1, D)`` temporaries are allocated and freed every iteration, so
        CPython reuses their addresses and the cache silently handed back a
        previous instance's neighbourhood.
        """
        key = np.ascontiguousarray(x_batch, dtype=np.float32).tobytes()
        hit = self._knn_cache.get(key)
        if hit is None:
            hit = self.explainer.knn.kneighbors(
                x_batch, n_neighbors=self.max_k
            )
            if len(self._knn_cache) >= 64:
                self._knn_cache.clear()
            self._knn_cache[key] = hit
        return hit

    def analytical_attr(self, x_star, surrogate_vector, dists, idx_nn, k,
                        surrogate="ig"):
        """Kernel-weighted covariance transform of a gradient / IG surrogate.

        ``surrogate_vector`` is the precomputed gradient (``surrogate=
        "gradient"``) or integrated-gradients (``surrogate="ig"``) vector for
        this instance.  Callers precompute it once per instance so the k-grid
        search does not recompute the (expensive) IG path.
        """
        # kneighbors on a single instance returns shape (1, max_k); flatten to
        # (max_k,) so slicing [:k] picks neighbours, not rows.
        dists = np.asarray(dists).ravel()
        idx_nn = np.asarray(idx_nn).ravel()
        k = int(max(1, min(int(k), len(dists))))
        dists_k = dists[:k]
        bw = float(np.median(dists_k))
        if bw == 0:
            bw = np.finfo(float).eps
        weights = kernel_weights(dists_k, bw, ANALYTICAL_KERNEL)
        w_grad = surrogate_vector

        n_total = len(self.explainer.X_train)
        if k >= n_total:
            # Every training row is used: scatter the kernel weights back into
            # train order and skip the N x D gather copy (655 MB at 40k x 4096).
            v = np.zeros(n_total, dtype=np.float64)
            v[idx_nn[:k]] = weights / weights.sum()
            a = self.explainer._cov_transform(w_grad, self.explainer.X_train, v)
            if self.explainer.normalize:
                m = float(np.max(np.abs(a)))
                if m > 0:
                    a = a / m
            return a

        X_K = self.explainer.X_train[idx_nn[:k]]
        return self.explainer._attribution(w_grad, X_K, weights)

    def evaluate_instance(self, x, mask, lime_v, lime_b, grad_vector,
                          ig_vector, k_grad, k_ig, pat_k_ratio=None):
        """Run all methods on a single instance; return dict of (emd, ime).

        Attribution computation is serial (shares cached KNN / gradients);
        EMD scoring of every map runs on a small thread pool because the POT
        network simplex releases the GIL and dominates runtime.
        """
        pat_k_ratio = K_RATIO if pat_k_ratio is None else pat_k_ratio
        times = {}
        attrs = {}

        # LIME baseline (precomputed by the caller)
        attrs["lime"] = lime_v

        # PatternLocal
        ts = time.time()
        attrs["patternlocal"] = pattern_from_lime(
            x, lime_v, lime_b, self.X_train,
            PAT_BANDWIDTH, pat_k_ratio, SHRINKAGE)
        times["patternlocal"] = time.time() - ts

        # knn for this instance
        dists, idx_nn = self.knn(np.asarray(x, dtype=np.float32).reshape(1, -1))

        # Analytical-gradient
        ts = time.time()
        attrs["analytical_grad"] = self.analytical_attr(
            x, grad_vector, dists, idx_nn, k_grad, "gradient")
        times["analytical_grad"] = time.time() - ts

        # Analytical-IG
        ts = time.time()
        attrs["analytical_ig"] = self.analytical_attr(
            x, ig_vector, dists, idx_nn, k_ig, "ig")
        times["analytical_ig"] = time.time() - ts

        # ---- plain baselines (no KNN) ------------------------------------
        ts = time.time()
        attrs["plain_grad"] = _plain_gradient(self, x)
        times["plain_grad"] = time.time() - ts

        ts = time.time()
        attrs["plain_ig"] = _plain_integrated_gradients(self, x)
        times["plain_ig"] = time.time() - ts

        ts = time.time()
        attrs["plain_smoothgrad"] = _plain_smoothgrad(self, x)
        times["plain_smoothgrad"] = time.time() - ts

        # ---- paper baselines (Sundararajan/Shrikumar/Springenberg/Captum) --
        ts = time.time()
        attrs["saliency"] = _plain_saliency(self, x)
        times["saliency"] = time.time() - ts

        ts = time.time()
        attrs["deeplift"] = _plain_deeplift(self, x)
        times["deeplift"] = time.time() - ts

        ts = time.time()
        attrs["gradient_shap"] = _plain_gradient_shap(self, x)
        times["gradient_shap"] = time.time() - ts

        ts = time.time()
        attrs["guided_backprop"] = _plain_guided_backprop(self, x)
        times["guided_backprop"] = time.time() - ts

        ts = time.time()
        attrs["sobel"] = _sobel(self, x)
        times["sobel"] = time.time() - ts

        ts = time.time()
        attrs["laplace"] = _laplace(self, x)
        times["laplace"] = time.time() - ts

        # ---- scoring (threaded) -------------------------------------------
        order = list(METHODS)
        ts = time.time()
        normalized = [(m, _normalize(attrs[m])) for m in order]
        if SCORE_WORKERS > 1 and len(normalized) > 1:
            scored = list(_score_pool().map(
                lambda mi: _score(mi[1], mask), normalized))
            scores = dict(zip(order, scored))
        else:
            scores = {m: _score(a, mask) for m, a in normalized}
        times["scoring"] = time.time() - ts

        return scores, {
            "lime": lime_v,
            "patternlocal": attrs["patternlocal"],
            "analytical_ig": attrs["analytical_ig"],
        }, times


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xor_corr_data(data_dir, alpha):
    """Load the cached XOR CORR npz for a given alpha."""
    pattern = f"XOR_CORR_alpha{alpha:.1f}.npz"
    matches = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{pattern} not found under {data_dir}")
    return np.load(matches[0])


def load_model(model_path, input_dim):
    import torch
    from models.mlp import MLPModel

    model = MLPModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def find_file(pattern):
    matches = glob.glob(os.path.join("/kaggle/input", "**", pattern), recursive=True)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_alpha(data_path, model_path, n_test=150, k_grad=300, k_ig=300,
                   lime_bandwidth=None, seed=42, alpha=None,
                   save_images=True, min_accuracy=0.90):
    """Run the full multi-method evaluation for one alpha config.

    Returns a results dict with per-method EMD/IME means/stds (one entry per
    name in :data:`METHODS`) and the raw per-sample lists.  Also saves images +
    .npz for the first ``N_SAVE_IMAGES`` eval instances when
    ``save_images=True``.
    """
    import torch

    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    masks_test = data["masks_test"]
    D = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

    model = load_model(model_path, D)
    model.to(DEVICE)
    test_acc = float(
        (model_logits(model, X_test, device=DEVICE).argmax(axis=1) == y_test).mean()
    )

    if test_acc < min_accuracy:
        print(f"\n  SKIP alpha={alpha}: test accuracy {test_acc:.4f} "
              f"below minimum {min_accuracy:.2f}")
        return {
            "config": os.path.basename(data_path),
            "alpha": float(alpha) if alpha is not None else None,
            "test_accuracy": test_acc,
            "n_eval": 0,
            "k_grad": None,
            "k_ig": None,
            "skipped": True,
            "skip_reason": f"test_acc {test_acc:.4f} < min_accuracy {min_accuracy:.2f}",
        }

    # Held-out split: tune on the first n_tune, evaluate on the rest.
    rng = np.random.default_rng(seed)
    n_test = min(n_test, len(X_test))
    idx = rng.choice(len(X_test), size=n_test, replace=False)
    n_tune = min(TUNE_SAMPLES, n_test // 2)
    tune_idx = idx[:n_tune]
    eval_idx = idx[n_tune:]

    # LIME weights helper (explicit bandwidth — no hidden closure state).
    def lime_for(i, bw):
        return lime_explain(model, X_test[i], n_samples=N_LIME,
                            bandwidth=bw, seed=int(i), device=DEVICE)

    # Build evaluator (fits KNN once, caches gradients).  max_k = all training
    # rows so the k-grids below can reach the full-data (global) covariance.
    ev = XORCorrEvaluator(model, X_train, device=DEVICE)

    # Precompute gradients + integrated-gradients for tune + eval instances.
    # IG is the expensive part (IG_STEPS forward/backward passes), so it is
    # computed once per instance and shared by the k-grid search and the final
    # evaluation instead of being recomputed inside analytical_attr().
    grads = {int(i): ev.gradient(X_test[i]) for i in list(tune_idx) + list(eval_idx)}
    igs = {int(i): ev.integrated_gradients(X_test[i], steps=IG_STEPS)
           for i in list(tune_idx) + list(eval_idx)}
    # ---- tune LIME bandwidth (paper: TPE over [0.5, 30]) ------------------
    bw_grid = list(LIME_BW_GRID) if lime_bandwidth is None else [lime_bandwidth]
    if len(bw_grid) > 1:
        best_bw = (None, np.inf)
        for bw in bw_grid:
            items = []
            for i in tune_idx:
                v, b = lime_for(i, bw)
                items.append((_normalize(v), masks_test[i]))
            m = float(np.mean([s[0] for s in _score_many(items)]))
            print(f"  tune LIME bw={bw:5.1f} -> EMD {m:.4f}", flush=True)
            if m < best_bw[1]:
                best_bw = (bw, m)
        lime_bw = best_bw[0]
    else:
        lime_bw = bw_grid[0]
    print(f"  chosen LIME bandwidth: {lime_bw}", flush=True)

    # Tune-set LIME weights (shared by every PatternLocal k candidate).
    tune_lime = {int(i): lime_for(i, lime_bw) for i in tune_idx}
    tune_knn = {int(i): ev.knn(np.asarray(X_test[i], dtype=np.float32).reshape(1, -1))
                for i in tune_idx}

    # ---- tune PatternLocal k_ratio (proxy for bandwidth/lambda/kernel) ----
    best_pat_k = (None, np.inf)
    for kr in PAT_K_GRID:
        items = []
        for i in tune_idx:
            v, b = tune_lime[int(i)]
            pl = pattern_from_lime(X_test[i], v, b, X_train,
                                   PAT_BANDWIDTH, kr, SHRINKAGE)
            items.append((_normalize(pl), masks_test[i]))
        m = float(np.mean([s[0] for s in _score_many(items)]))
        print(f"  tune PatternLocal k_ratio={kr} -> EMD {m:.4f}", flush=True)
        if m < best_pat_k[1]:
            best_pat_k = (kr, m)
    pat_k_ratio = best_pat_k[0]
    print(f"  chosen PatternLocal k_ratio: {pat_k_ratio}", flush=True)

    # ---- tune k_grad / k_ig for the analytical methods --------------------
    # The small-k regime is *structurally* wrong for these data: the k nearest
    # neighbours share the identical class-conditional signal, so the local
    # covariance has ~zero variance along the signal direction and the transform
    # annihilates the attribution (see NOTES_patternlocal_rootcause.md).  A
    # 30-sample tune set can nonetheless prefer such a k by chance, which is how
    # analytical_ig picked k=5000 and scored IME 0.56 (~LIME) on WHITE a=0.9
    # while patternlocal scored 0.001.  So the k-grid is restricted to the
    # regime that retains signal variance, and the floor is the k that
    # PatternLocal's own (already-tuned) k_ratio search selected.
    n_train = len(X_train)
    k_floor = max(1, int(round(pat_k_ratio * n_train)))
    k_grid = sorted({int(k) for k in (k_floor, 5000, 10000, 20000, n_train)
                     if k_floor <= k <= n_train} | {int(n_train)})
    print(f"  analytical k-grid (floor {k_floor} from pat_k_ratio="
          f"{pat_k_ratio}): {k_grid}", flush=True)

    best_grad_k = (None, np.inf)
    for k in k_grid:
        items = []
        for i in tune_idx:
            dists, idx_nn = tune_knn[int(i)]
            a = ev.analytical_attr(X_test[i], grads[int(i)], dists, idx_nn,
                                   k, "gradient")
            items.append((_normalize(a), masks_test[i]))
        m = float(np.mean([s[0] for s in _score_many(items)]))
        print(f"  tune analytical-grad k={k} -> EMD {m:.4f}", flush=True)
        if m < best_grad_k[1]:
            best_grad_k = (k, m)

    tune_ig = igs
    best_ig_k = (None, np.inf)
    for k in k_grid:
        items = []
        for i in tune_idx:
            dists, idx_nn = tune_knn[int(i)]
            a = ev.analytical_attr(X_test[i], tune_ig[int(i)], dists, idx_nn,
                                   k, "ig")
            items.append((_normalize(a), masks_test[i]))
        m = float(np.mean([s[0] for s in _score_many(items)]))
        print(f"  tune analytical-IG  k={k} -> EMD {m:.4f}", flush=True)
        if m < best_ig_k[1]:
            best_ig_k = (k, m)

    print(f"  chosen k_grad={best_grad_k[0]}  k_ig={best_ig_k[0]}", flush=True)
    k_grad, k_ig = best_grad_k[0], best_ig_k[0]

    # ---- final evaluation on held-out split -------------------------------
    print("\n--- Final evaluation ---")
    methods = list(METHODS)
    results = {m: [] for m in methods}
    result_times = {m: [] for m in methods}
    score_times = []
    t0 = time.time()

    # Containers for image / npz saving (first N_SAVE_IMAGES eval instances).
    saved_tags = []
    saved_arrays = {
        "X": [], "mask": [],
        "lime": [], "patternlocal": [], "analytical_ig": [],
    }

    for eval_pos, i in enumerate(eval_idx):
        x = X_test[i]
        mask = masks_test[i]
        v, b = lime_for(i, lime_bw)

        scores, attrs, inst_times = ev.evaluate_instance(x, mask, v, b,
                                             grads[int(i)], igs[int(i)],
                                             k_grad, k_ig, pat_k_ratio)

        # ---- per-method scoring + timing ----------------------------------
        # Generic over `methods` so every registered method (including the
        # paper baselines: saliency/deeplift/gradient_shap/guided_backprop/
        # sobel/laplace) is collected, not just the first seven.
        for m in methods:
            emd_m, ime_m = scores[m]
            results[m].append((emd_m, ime_m))
            result_times[m].append(inst_times.get(m, 0.0))
        score_times.append(inst_times.get("scoring", 0.0))

        # ---- image saving for first N_SAVE_IMAGES --------------------------
        if save_images and eval_pos < N_SAVE_IMAGES:
            tag = _save_sample_images(alpha, eval_pos, x, mask, v,
                                      attrs["patternlocal"],
                                      attrs["analytical_ig"])
            saved_tags.append(tag)
            saved_arrays["X"].append(x.reshape(IMAGE_SHAPE))
            saved_arrays["mask"].append(mask.reshape(IMAGE_SHAPE))
            saved_arrays["lime"].append(_normalize(v).reshape(IMAGE_SHAPE))
            saved_arrays["patternlocal"].append(_normalize(attrs["patternlocal"]).reshape(IMAGE_SHAPE))
            saved_arrays["analytical_ig"].append(_normalize(attrs["analytical_ig"]).reshape(IMAGE_SHAPE))

    def agg(pairs):
        emds = [p[0] for p in pairs]
        imes = [p[1] for p in pairs]
        return {
            "emd_mean": float(np.mean(emds)), "emd_std": float(np.std(emds)),
            "ime_mean": float(np.mean(imes)), "ime_std": float(np.std(imes)),
        }

    def timings(seconds_list):
        return {
            "sec_per_sample_mean": float(np.mean(seconds_list)),
            "sec_per_sample_std": float(np.std(seconds_list)),
            "sec_per_sample_total": float(np.sum(seconds_list)),
        }

    # ---- save images .npz --------------------------------------------------
    npz_path = None
    if save_images and saved_arrays["X"]:
        npz_path = os.path.join(OUT_DIR,
                                f"alpha{alpha:.1f}_sample_images.npz")
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(npz_path,
                 tags=saved_tags,
                 X=np.stack(saved_arrays["X"]),
                 mask=np.stack(saved_arrays["mask"]),
                 lime=np.stack(saved_arrays["lime"]),
                 patternlocal=np.stack(saved_arrays["patternlocal"]),
                 analytical_ig=np.stack(saved_arrays["analytical_ig"]))
        print(f"  Saved sample images + npz: {npz_path}")

    ret = {
        "config": os.path.basename(data_path),
        "alpha": float(alpha) if alpha is not None else None,
        "test_accuracy": test_acc,
        "n_eval": len(eval_idx),
        "k_grad": best_grad_k[0],
        "k_ig": best_ig_k[0],
        "lime_bandwidth": lime_bw,
        "pat_k_ratio": pat_k_ratio,
        "time": {m: timings(result_times[m]) for m in methods},
        "raw": {m: [[float(e), float(i)] for e, i in v]
                for m, v in results.items()},
        "saved_image_tags": saved_tags,
        "saved_image_npz": npz_path,
    }
    ret["time"]["scoring"] = timings(score_times)
    for m in methods:
        ret[m] = agg(results[m])
    return ret


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def save_results(results_dict, path="/kaggle/working/eval_results.json"):
    serializable = _to_python(results_dict)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")
