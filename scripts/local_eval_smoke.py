"""Micro end-to-end test of xai.eval_common.evaluate_alpha on a tiny fixture.

Exercises: LIME-bandwidth tuning, PatternLocal k tuning, k_grad/k_ig tuning,
threaded EMD scoring, results aggregation, sample-image saving.

Run: /tmp/xai_venv/bin/python -u scripts/local_eval_smoke.py
"""
import json
import os
import sys
import time

sys.path.insert(0, "src/paper_repro")

import numpy as np
import torch

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp

OUT = "/tmp/eval_smoke"
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("SCORE_WORKERS", "4")
os.environ.setdefault("EVAL_OUT_DIR", OUT)

# evaluate_alpha scans /kaggle/input for the library; we are local, so the
# module-level walk finds nothing and the imports above provide `xai.*`.
from xai.eval_common import evaluate_alpha  # noqa: E402

N = 1500
ALPHA = 0.9


def main():
    data_path = os.path.join(OUT, "XOR_WHITE_alpha0.9.npz")
    model_path = os.path.join(OUT, "XOR_WHITE_alpha0.9_model.pt")
    if os.path.exists(data_path) and os.path.exists(model_path):
        print("reusing existing fixture...", flush=True)
    else:
        print("building fixture...", flush=True)
        X, y, masks = generate_xai_tris(n_samples=N, scenario="XOR",
                                        noise="WHITE", alpha=ALPHA,
                                        image_size=64, seed=11)
        (Xtr, ytr, _), (Xval, yval, mval), (Xte, yte, mte) = split_dataset(
            X, y, masks, seed=11)
        model = MLPModel(input_dim=4096)
        model, hist, bv = train_mlp(model, Xtr, ytr, Xval, yval, lr=1e-4,
                                    batch_size=128, max_epochs=40, patience=10,
                                    seed=11)
        np.savez(data_path, X_train=Xtr, y_train=ytr, X_val=Xval, y_val=yval,
                 masks_val=mval, X_test=Xte, y_test=yte, masks_test=mte)
        torch.save(model.state_dict(), model_path)

    t0 = time.time()
    res = evaluate_alpha(data_path, model_path, n_test=8, alpha=ALPHA,
                         save_images=True, seed=11)
    print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)

    for m in ("lime", "patternlocal", "analytical_ig"):
        print(f"  {m:15s} {res[m]['emd_mean']:.4f} ± {res[m]['emd_std']:.4f} / "
              f"IME {res[m]['ime_mean']:.4f}", flush=True)
    print("  lime_bandwidth", res["lime_bandwidth"],
          "pat_k_ratio", res["pat_k_ratio"],
          "k_grad", res["k_grad"], "k_ig", res["k_ig"], flush=True)
    print("  n_eval", res["n_eval"], "scoring_total",
          round(res["time"]["scoring"]["sec_per_sample_total"], 1), flush=True)
    assert res["n_eval"] > 0 and res["lime_bandwidth"] in (10.0, 20.0, 30.0)
    assert res["pat_k_ratio"] in (0.05, 0.5, 1.0)
    assert len(res["raw"]["lime"]) == res["n_eval"]
    json.dumps(res)  # serializable?
    print("OK", flush=True)


if __name__ == "__main__":
    main()
