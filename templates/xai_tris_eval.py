#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if "__init__.py" in files:
        sys.path.insert(0, root)

from xai.patternlocal import patternlocal_explain
from metrics.emd import compute_emd
from metrics.ime import compute_ime

DATA_DIR = "/kaggle/input/xai-tris-xor-corr"


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def find_file(directory, pattern):
    import glob
    matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    return matches[0] if matches else None


def main():
    print("=" * 60)
    print("Single-Config Evaluation: XOR CORR alpha=0.2")
    print("=" * 60)

    config = "XOR_CORR_alpha0.2"

    print("Searching all /kaggle/input/ files:")
    all_files = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            all_files.append(os.path.join(root, f))
            if len(all_files) <= 30:
                print(f"  {os.path.join(root, f)}")
    print(f"Total files found: {len(all_files)}")

    target = f"{config}.npz"
    matches = [f for f in all_files if target in f]
    if not matches:
        print(f"ERROR: {target} not found")
        sys.exit(1)
    data_path = matches[0]
    print(f"Found data at: {data_path}")

    target_model = f"{config}_model.pt"
    matches_model = [f for f in all_files if target_model in f]
    if not matches_model:
        print(f"ERROR: {target_model} not found")
        sys.exit(1)
    model_path = matches_model[0]
    print(f"Found model at: {model_path}")

    data = np.load(data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    masks_test = data["masks_test"]

    import torch
    from models.mlp import MLPModel
    model = MLPModel(input_dim=4096)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    n_test = min(200, len(X_test))
    indices = np.random.default_rng(42).choice(len(X_test), n_test, replace=False)

    lime_emds, lime_imes = [], []
    pattern_emds, pattern_imes = [], []

    for idx in indices:
        x = X_test[idx]
        mask = masks_test[idx]

        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        x_tensor.requires_grad_(True)
        model.zero_grad()
        output = model(x_tensor)
        pred_class = output.argmax(dim=1).item()
        output[0, pred_class].backward()
        lime_weights = x_tensor.grad.data.squeeze(0).numpy()

        lime_abs = np.abs(lime_weights)
        if lime_abs.max() > 0:
            lime_weights = lime_weights / lime_abs.max()

        lime_emd = compute_emd(lime_weights, mask, image_shape=(64, 64))
        lime_ime = compute_ime(lime_weights, mask)
        lime_emds.append(lime_emd)
        lime_imes.append(lime_ime)

        pattern_weights = patternlocal_explain(
            lime_weights=lime_weights,
            lime_intercept=0.0,
            instance=x,
            X_train=data["X_train"][:5000],
            bandwidth=10.0,
            kernel="gaussian",
            k_ratio=0.1,
            lam=0.01,
        )

        pattern_abs = np.abs(pattern_weights)
        if pattern_abs.max() > 0:
            pattern_weights = pattern_weights / pattern_abs.max()

        pattern_emd = compute_emd(pattern_weights, mask, image_shape=(64, 64))
        pattern_ime = compute_ime(pattern_weights, mask)
        pattern_emds.append(pattern_emd)
        pattern_imes.append(pattern_ime)

    lime_emd_mean = np.mean(lime_emds)
    lime_emd_std = np.std(lime_emds)
    lime_ime_mean = np.mean(lime_imes)
    lime_ime_std = np.std(lime_imes)
    pattern_emd_mean = np.mean(pattern_emds)
    pattern_emd_std = np.std(pattern_emds)
    pattern_ime_mean = np.mean(pattern_imes)
    pattern_ime_std = np.std(pattern_imes)

    print(f"\nResults over {n_test} test samples:")
    print(f"  LIME      EMD: {lime_emd_mean:.4f} +/- {lime_emd_std:.4f}")
    print(f"  LIME      IME: {lime_ime_mean:.4f} +/- {lime_ime_std:.4f}")
    print(f"  PatternLocal EMD: {pattern_emd_mean:.4f} +/- {pattern_emd_std:.4f}")
    print(f"  PatternLocal IME: {pattern_ime_mean:.4f} +/- {pattern_ime_std:.4f}")
    print(f"\n  PatternLocal EMD improvement: {(lime_emd_mean - pattern_emd_mean):.4f}")
    print(f"  PatternLocal IME improvement: {(lime_ime_mean - pattern_ime_mean):.4f}")

    format_result("lime_emd_mean", f"{lime_emd_mean:.4f}")
    format_result("lime_emd_std", f"{lime_emd_std:.4f}")
    format_result("lime_ime_mean", f"{lime_ime_mean:.4f}")
    format_result("lime_ime_std", f"{lime_ime_std:.4f}")
    format_result("patternlocal_emd_mean", f"{pattern_emd_mean:.4f}")
    format_result("patternlocal_emd_std", f"{pattern_emd_std:.4f}")
    format_result("patternlocal_ime_mean", f"{pattern_ime_mean:.4f}")
    format_result("patternlocal_ime_std", f"{pattern_ime_std:.4f}")
    format_result("patternlocal_better_emd", pattern_emd_mean < lime_emd_mean)
    format_result("patternlocal_better_ime", pattern_ime_mean < lime_ime_mean)

    results = {
        "config": config,
        "n_test": n_test,
        "lime": {"emd_mean": lime_emd_mean, "emd_std": lime_emd_std, "ime_mean": lime_ime_mean, "ime_std": lime_ime_std},
        "patternlocal": {"emd_mean": pattern_emd_mean, "emd_std": pattern_emd_std, "ime_mean": pattern_ime_mean, "ime_std": pattern_ime_std},
    }
    with open("/kaggle/working/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()