#!/usr/bin/env python3
"""Fast validation: does the fixed RIGID generator reach >90% with the REAL
torch MLPModel + train_mlp (not an sklearn proxy)?

Trains RIGID CORR at alpha in {0.2, 0.3, 0.5} plus RIGID WHITE at 0.3/0.5 on a
reduced sample count so it fits a CPU smoke-test budget. Prints [RESULT] lines.
"""
import sys
import os

for root, dirs, files in os.walk("/kaggle/input"):
    if "__init__.py" in files:
        sys.path.insert(0, root)

import numpy as np
from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model


def main():
    N = 12000
    for noise in ("CORR", "WHITE"):
        for alpha in (0.2, 0.3, 0.5):
            seed = 4242
            X, y, masks = generate_xai_tris(
                n_samples=N, scenario="RIGID", noise=noise,
                alpha=alpha, image_size=64, seed=seed)
            (Xtr, ytr, _), (Xv, yv, _), (Xt, yt, mt) = split_dataset(
                X, y, masks, seed=seed)
            model = MLPModel(input_dim=4096)
            model, hist, bva = train_mlp(
                model, Xtr, ytr, Xv, yv, lr=1e-4, batch_size=128,
                max_epochs=500, patience=100, seed=42)
            acc = evaluate_model(model, Xt, yt)
            mpx = float(mt.sum(axis=1).mean())
            print(f"[RESULT] RIGID_{noise}_alpha{alpha:.1f}: "
                  f"test_acc={acc:.4f} best_val={bva:.4f} "
                  f"epochs={len(hist['val_acc'])} mask_px={mpx:.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
