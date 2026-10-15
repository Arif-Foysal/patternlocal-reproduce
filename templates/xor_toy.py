#!/usr/bin/env python3
import sys
import os
import json
import glob
import numpy as np

for p in glob.glob("/kaggle/input/*/"):
    if os.path.isfile(os.path.join(p, "experiments", "xor_toy.py")):
        sys.path.insert(0, p.rstrip("/"))
        break
sys.path.insert(0, "/kaggle/input")

from experiments.xor_toy import run_xor_toy
from xai.baselines import sobel_explanation, laplace_explanation


def format_result(key, value):
    print(f"[RESULT] {key}: {value}")


def save_results(results_dict, path="/kaggle/working/xor_toy_results.json"):
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {path}")


def main():
    print("=" * 60)
    print("XOR Suppressor Toy Problem — Smoke Test")
    print("=" * 60)

    results = run_xor_toy(n_samples=2500, sigma_d=1.0, tau=0.1, seed=42)

    print(f"\nTest samples: {results['n_test']}")
    print(f"LIME suppressor (|w3| mean): {results['lime_suppressor_mean']:.4f}")
    print(f"PatternLocal suppressor (|a3| mean): {results['pattern_suppressor_mean']:.4f}")
    print(f"LIME suppressor (normalized mean): {results['lime_suppressor_normalized_mean']:.4f}")
    print(f"PatternLocal suppressor (normalized mean): {results['pattern_suppressor_normalized_mean']:.4f}")
    print(f"Suppression ratio (pattern/lime): {results['suppression_ratio']:.4f}")

    print(f"\nSample LIME weights (first 3 test points):")
    for i, w in enumerate(results['lime_weights_sample'][:3]):
        print(f"  Point {i}: w1={w[0]:.4f}, w2={w[1]:.4f}, w3={w[2]:.4f}")

    print(f"\nSample PatternLocal weights (first 3 test points):")
    for i, w in enumerate(results['pattern_weights_sample'][:3]):
        print(f"  Point {i}: a1={w[0]:.4f}, a2={w[1]:.4f}, a3={w[2]:.4f}")

    format_result("n_test", results['n_test'])
    format_result("lime_suppressor_mean", f"{results['lime_suppressor_mean']:.4f}")
    format_result("pattern_suppressor_mean", f"{results['pattern_suppressor_mean']:.4f}")
    format_result("lime_suppressor_normalized_mean", f"{results['lime_suppressor_normalized_mean']:.4f}")
    format_result("pattern_suppressor_normalized_mean", f"{results['pattern_suppressor_normalized_mean']:.4f}")
    format_result("suppression_ratio", f"{results['suppression_ratio']:.4f}")

    success = results['pattern_suppressor_normalized_mean'] < 0.3 * results['lime_suppressor_normalized_mean']
    format_result("suppressor_reduced", success)
    format_result("PASS" if success else "FAIL", "suppressor check")

    save_results(results)

    print("\n" + "=" * 60)
    print("XOR toy problem complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
