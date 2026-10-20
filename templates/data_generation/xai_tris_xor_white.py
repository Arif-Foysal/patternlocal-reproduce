#!/usr/bin/env python3
import sys
import os
import json
import glob
import zlib
import shutil
import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if "__init__.py" in files:
        sys.path.insert(0, root)

from data.xai_tris import generate_xai_tris, split_dataset
from models.mlp import MLPModel
from models.train import train_mlp, evaluate_model

OUTPUT_DIR = "/kaggle/working/xai_tris_data"


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def check_disk():
    import subprocess
    result = subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True)
    print("Disk usage:")
    print(result.stdout)


def clear_temp():
    shutil.rmtree('/tmp', ignore_errors=True)


def main():
    print("=" * 60)
    print("XAI-TRIS Data Generation: XOR WHITE")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Source fingerprint (verify the correct core-dataset version is mounted).
    import hashlib
    src_candidates = glob.glob("/kaggle/input/**/data/xai_tris.py", recursive=True)
    if src_candidates:
        src_path = src_candidates[0]
        digest = hashlib.md5(open(src_path, "rb").read()).hexdigest()
        print(f"Source: {src_path}")
        print(f"Source fingerprint: {digest}")

    scenario = "XOR"
    noise = "WHITE"
    alphas = [round(0.1 * i, 1) for i in range(1, 10)]
    n_samples = 40000
    print(f"Samples per alpha: {n_samples}")

    summary = {}

    for alpha in alphas:
        config_name = f"{scenario}_{noise}_alpha{alpha:.1f}"
        print(f"\n--- {config_name} ---")
        check_disk()

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
            np.savez_compressed(cache_path,
                X_train=X_train.astype(np.float32),
                y_train=y_train.astype(np.int64),
                X_val=X_val.astype(np.float32),
                y_val=y_val.astype(np.int64),
                X_test=X_test.astype(np.float32),
                y_test=y_test.astype(np.int64),
                masks_test=masks_test.astype(np.float32),
                masks_val=masks_val.astype(np.float32),
            )
            print(f"  Generated and cached.")
            clear_temp()

        class_balance = float(np.mean(y_test))
        print(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        print(f"  Class balance (test): {class_balance:.3f}")
        print(f"  Mask pixels (test mean): {float(masks_test.sum(axis=1).mean()):.1f}")

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

        del X_train, y_train, X_val, y_val, X_test, y_test, masks_test, masks_val
        del model

    summary_path = os.path.join(OUTPUT_DIR, f"training_summary_{scenario}_{noise}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Completed {len(summary)} alphas for {scenario} {noise}")
    print("=" * 60)
    if summary:
        min_acc = min(v["test_acc"] for v in summary.values())
        print(f"Minimum test accuracy: {min_acc:.4f}")
        format_result("min_test_acc", f"{min_acc:.4f}")
        format_result("PASS" if min_acc > 0.90 else "FAIL", "90% accuracy criterion")


if __name__ == "__main__":
    main()