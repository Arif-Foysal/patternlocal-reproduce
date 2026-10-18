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

# LIME / PatternLocal knobs (identity rep -> LIME bandwidth ineffective;
# PatternLocal auto bandwidth == best tested).
LIME_BANDWIDTH = None  # set per-run in evaluate()
PAT_BANDWIDTH = None
K_RATIO = 0.05
SHRINKAGE = 0.0
IG_STEPS = 50
N_LIME = 2000

# How many of the first eval instances to save as images + .npz per alpha.
N_SAVE_IMAGES = 5


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
                        out_dir="/kaggle/working"):
    """Save input/mask/attribution PNGs for one instance."""
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

    def __init__(self, model, X_train, k_neighbors=300, device=DEVICE):
        self.model = model
        self.X_train = X_train
        self.device = device
        self.explainer = AnalyticalPatternLocalExplainer(
            model, X_train, k_neighbors=k_neighbors, device=device,
            normalize=False, kernel=ANALYTICAL_KERNEL,
            bandwidth=ANALYTICAL_BANDWIDTH,
        )
        self.max_k = k_neighbors
        self._grad_cache = {}
        self._knn_cache = {}

    def gradient(self, x_star, seed=0):
        return self.explainer.gradient(x_star)

    def integrated_gradients(self, x_star, steps=IG_STEPS):
        return self.explainer.integrated_gradients(x_star, steps=steps)

    def knn(self, x_batch):
        """Return (distances, indices) for a batch, cached per batch object."""
        key = id(x_batch)
        if key not in self._knn_cache:
            self._knn_cache[key] = self.explainer.knn.kneighbors(
                x_batch, n_neighbors=self.max_k
            )
        return self._knn_cache[key]

    def analytical_attr(self, x_star, grad, dists, idx_nn, k, surrogate="ig"):
        # kneighbors on a single instance returns shape (1, max_k); flatten to
        # (max_k,) so slicing [:k] picks neighbours, not rows.
        dists = np.asarray(dists).ravel()
        idx_nn = np.asarray(idx_nn).ravel()
        X_K = self.explainer.X_train[idx_nn[:k]]
        dists_k = dists[:k]
        bw = float(np.median(dists_k))
        if bw == 0:
            bw = np.finfo(float).eps
        weights = kernel_weights(dists_k, bw, ANALYTICAL_KERNEL)
        w_grad = grad if surrogate == "gradient" else self.integrated_gradients(
            x_star, steps=IG_STEPS
        )
        return self.explainer._attribution(w_grad, X_K, weights)

    def evaluate_instance(self, x, mask, lime_v, lime_b, grad_vector, k_grad,
                          k_ig):
        """Run all seven methods on a single instance; return dict of (emd, ime)."""
        times = {}
        # LIME baseline
        ts = time.time()
        lime_e, lime_i = _score(_normalize(lime_v), mask)
        times["lime"] = time.time() - ts

        # PatternLocal
        ts = time.time()
        pl = pattern_from_lime(x, lime_v, lime_b, self.X_train,
                               PAT_BANDWIDTH, K_RATIO, SHRINKAGE)
        pl_e, pl_i = _score(_normalize(pl), mask)
        times["patternlocal"] = time.time() - ts

        # knn for this instance
        dists, idx_nn = self.knn(np.asarray(x, dtype=np.float32).reshape(1, -1))

        # Analytical-gradient
        ts = time.time()
        ag = self.analytical_attr(x, grad_vector, dists, idx_nn, k_grad,
                                  "gradient")
        ag_e, ag_i = _score(_normalize(ag), mask)
        times["analytical_grad"] = time.time() - ts

        # Analytical-IG
        ts = time.time()
        ig = self.analytical_attr(x, grad_vector, dists, idx_nn, k_ig, "ig")
        ig_e, ig_i = _score(_normalize(ig), mask)
        times["analytical_ig"] = time.time() - ts

        # ---- plain baselines (no KNN) ------------------------------------
        ts = time.time()
        pg = _plain_gradient(self, x)
        pg_e, pg_i = _score(_normalize(pg), mask)
        times["plain_grad"] = time.time() - ts

        ts = time.time()
        pig = _plain_integrated_gradients(self, x)
        pig_e, pig_i = _score(_normalize(pig), mask)
        times["plain_ig"] = time.time() - ts

        ts = time.time()
        psg = _plain_smoothgrad(self, x)
        psg_e, psg_i = _score(_normalize(psg), mask)
        times["plain_smoothgrad"] = time.time() - ts

        return {
            "lime": (lime_e, lime_i),
            "patternlocal": (pl_e, pl_i),
            "analytical_grad": (ag_e, ag_i),
            "analytical_ig": (ig_e, ig_i),
            "plain_grad": (pg_e, pg_i),
            "plain_ig": (pig_e, pig_i),
            "plain_smoothgrad": (psg_e, psg_i),
        }, {
            "lime": lime_v,
            "patternlocal": pl,
            "analytical_ig": ig,
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
    """Run the full 7-method evaluation for one alpha config.

    Returns a results dict with per-method EMD/IME means/stds and the raw
    per-sample lists. Also saves images + .npz for the first
    ``N_SAVE_IMAGES`` eval instances when ``save_images=True``.
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

    # Held-out split: tune k on the first n_tune, evaluate on the rest.
    rng = np.random.default_rng(seed)
    n_test = min(n_test, len(X_test))
    idx = rng.choice(len(X_test), size=n_test, replace=False)
    n_tune = min(30, n_test // 2)
    tune_idx = idx[:n_tune]
    eval_idx = idx[n_tune:]

    lime_bw = lime_bandwidth if lime_bandwidth is not None else 0.75 * np.sqrt(D)

    # LIME weights for tune + eval (needed to tune k_grad for patternlocal).
    def lime_for(i):
        return lime_explain(model, X_test[i], n_samples=N_LIME,
                            bandwidth=lime_bw, seed=int(i),
                            device=DEVICE)

    # Build evaluator (fits KNN once, caches gradients).
    ev = XORCorrEvaluator(model, X_train, k_neighbors=max(k_grad, k_ig),
                          device=DEVICE)

    # Precompute gradients for tune + eval.
    grads = {int(i): ev.gradient(X_test[i]) for i in list(tune_idx) + list(eval_idx)}

    # ---- tune k_grad for analytical-gradient (patternlocal-style k) -------
    best_grad_k = (None, np.inf)
    for k in (100, 300, 500, 800):
        emds = []
        for i in tune_idx:
            v, b = lime_for(i)
            _, idx_nn = ev.knn(np.asarray(X_test[i], dtype=np.float32).reshape(1, -1))
            a = ev.analytical_attr(X_test[i], grads[int(i)],
                                   ev.knn(np.asarray(X_test[i], dtype=np.float32).reshape(1, -1))[0],
                                   idx_nn, k, "gradient")
            e, _ = _score(_normalize(a), masks_test[i])
            emds.append(e)
        m = float(np.mean(emds))
        if m < best_grad_k[1]:
            best_grad_k = (k, m)

    # ---- tune k_ig for analytical-IG -------------------------------------
    best_ig_k = (None, np.inf)
    for k in (100, 300, 500, 800):
        emds = []
        for i in tune_idx:
            dists, idx_nn = ev.knn(np.asarray(X_test[i], dtype=np.float32).reshape(1, -1))
            ig = ev.integrated_gradients(X_test[i], steps=IG_STEPS)
            a = ev.analytical_attr(X_test[i], ig, dists, idx_nn, k, "ig")
            e, _ = _score(_normalize(a), masks_test[i])
            emds.append(e)
        m = float(np.mean(emds))
        if m < best_ig_k[1]:
            best_ig_k = (k, m)

    # ---- final evaluation on held-out split -------------------------------
    print("\n--- Final evaluation ---")
    methods = ["lime", "patternlocal", "analytical_grad",
               "analytical_ig", "plain_grad", "plain_ig",
               "plain_smoothgrad"]
    results = {m: [] for m in methods}
    result_times = {m: [] for m in methods}
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
        v, b = lime_for(i)

        scores, attrs, inst_times = ev.evaluate_instance(x, mask, v, b,
                                             grads[int(i)], k_grad, k_ig)

        # ---- per-method scoring + timing ----------------------------------
        lime_e, lime_i = scores["lime"]
        pl_e, pl_i = scores["patternlocal"]
        ag_e, ag_i = scores["analytical_grad"]
        ig_e, ig_i = scores["analytical_ig"]
        pg_e, pg_i = scores["plain_grad"]
        pig_e, pig_i = scores["plain_ig"]
        psg_e, psg_i = scores["plain_smoothgrad"]

        results["lime"].append((lime_e, lime_i))
        results["patternlocal"].append((pl_e, pl_i))
        results["analytical_grad"].append((ag_e, ag_i))
        results["analytical_ig"].append((ig_e, ig_i))
        results["plain_grad"].append((pg_e, pg_i))
        results["plain_ig"].append((pig_e, pig_i))
        results["plain_smoothgrad"].append((psg_e, psg_i))

        for m in methods:
            result_times[m].append(inst_times[m])

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
        npz_path = os.path.join("/kaggle/working",
                                f"alpha{alpha:.1f}_sample_images.npz")
        np.savez(npz_path,
                 tags=saved_tags,
                 X=np.stack(saved_arrays["X"]),
                 mask=np.stack(saved_arrays["mask"]),
                 lime=np.stack(saved_arrays["lime"]),
                 patternlocal=np.stack(saved_arrays["patternlocal"]),
                 analytical_ig=np.stack(saved_arrays["analytical_ig"]))
        print(f"  Saved sample images + npz: {npz_path}")

    return {
        "config": os.path.basename(data_path),
        "alpha": float(alpha) if alpha is not None else None,
        "test_accuracy": test_acc,
        "n_eval": len(eval_idx),
        "k_grad": best_grad_k[0],
        "k_ig": best_ig_k[0],
        "lime": agg(results["lime"]),
        "patternlocal": agg(results["patternlocal"]),
        "analytical_grad": agg(results["analytical_grad"]),
        "analytical_ig": agg(results["analytical_ig"]),
        "plain_grad": agg(results["plain_grad"]),
        "plain_ig": agg(results["plain_ig"]),
        "plain_smoothgrad": agg(results["plain_smoothgrad"]),
        "time": {
            "lime": timings(result_times["lime"]),
            "patternlocal": timings(result_times["patternlocal"]),
            "analytical_grad": timings(result_times["analytical_grad"]),
            "analytical_ig": timings(result_times["analytical_ig"]),
            "plain_grad": timings(result_times["plain_grad"]),
            "plain_ig": timings(result_times["plain_ig"]),
            "plain_smoothgrad": timings(result_times["plain_smoothgrad"]),
        },
        "raw": {k: [[float(e), float(i)] for e, i in v]
                for k, v in results.items()},
        "saved_image_tags": saved_tags,
        "saved_image_npz": npz_path,
    }


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
