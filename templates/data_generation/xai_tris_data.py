#!/usr/bin/env python3
import sys
import os
import json
import glob
import zlib
import numpy as np
import argparse

# Add all subdirectories containing __init__.py to sys.path
for root, dirs, files in os.walk("/kaggle/input"):
    if "__init__.py" in files:
        sys.path.insert(0, root)

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model

OUTPUT_DIR = "/kaggle/working/xai_tris_data"


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick mode with reduced samples")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples per config")
    parser.add_argument("--configs", type=str, default=None, help="Comma-separated list of configs to run")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("XAI-TRIS Data Generation and MLP Training")
    if args.quick:
        print("QUICK MODE: reduced samples")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = ["LIN", "XOR", "RIGID"]
    noise_types = ["WHITE", "CORR"]
    alphas = [round(0.1 * i, 1) for i in range(1, 10)]

    if args.configs:
        config_list = args.configs.split(",")
    else:
        config_list = [f"{s}_{n}_alpha{a:.1f}" for s in scenarios for n in noise_types for a in alphas]

    n_samples = args.n_samples if args.n_samples else (5000 if args.quick else 40000)
    print(f"Samples per config: {n_samples}")

    summary = {}

    for config_name in config_list:
        print(f"\n--- {config_name} ---")

        parts = config_name.split("_")
        if len(parts) != 3:
            print(f"  Skipping invalid config: {config_name}")
            continue
        scenario = parts[0]
        noise = parts[1]
        alpha = float(parts[2].replace("alpha", ""))

        cache_path = os.path.join(OUTPUT_DIR, f"{config_name}.npz")
        if os.path.exists(cache_path):
            print(f"  Cached, loading...")
            data = np.load(cache_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]
            masks_test = data["masks_test"]
        else:
            seed = zlib.crc32(config_name.encode()) % (2**31)
            X, y, masks = generate_xai_tris(
                n_samples=n_samples,
                scenario=scenario,
                noise=noise,
                alpha=alpha,
                image_size=64,
                seed=seed,
            )
            (X_train, y_train, _), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = split_dataset(X, y, masks, seed=seed)
            np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, masks_test=masks_test)
            print(f"  Generated and cached.")

        class_balance = float(np.mean(y_test))
        print(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        print(f"  Class balance (test): {class_balance:.3f}")

        model_cache = os.path.join(OUTPUT_DIR, f"{config_name}_model.pt")
        if os.path.exists(model_cache):
            print(f"  Loading cached model...")
            model = MLPModel(input_dim=4096)
            import torch
            model.load_state_dict(torch.load(model_cache, weights_only=True))
            test_acc = evaluate_model(model, X_test, y_test)
        else:
            model = MLPModel(input_dim=4096)
            model, history, best_val_acc = train_mlp(
                model, X_train, y_train, X_val, y_val,
                lr=1e-4, batch_size=128, max_epochs=500,
                patience=100, seed=42,
            )
            test_acc = evaluate_model(model, X_test, y_test)
            import torch
            torch.save(model.state_dict(), model_cache)

        print(f"  Test accuracy: {test_acc:.4f}")
        format_result(f"{config_name}_test_acc", f"{test_acc:.4f}")

        summary[config_name] = {
            "test_acc": test_acc,
            "class_balance": class_balance,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        }

        # Free memory
        del X_train, y_train, X_val, y_val, X_test, y_test, masks_test, masks_val
        del model

    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Completed {len(summary)} configs.")
    print("=" * 60)
    if summary:
        min_acc = min(v["test_acc"] for v in summary.values())
        print(f"Minimum test accuracy: {min_acc:.4f}")
        format_result("min_test_acc", f"{min_acc:.4f}")
        format_result("PASS" if min_acc > 0.90 else "FAIL", "90% accuracy criterion")


if __name__ == "__main__":
    main()