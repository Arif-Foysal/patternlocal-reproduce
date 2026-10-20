#!/usr/bin/env python3
"""XOR CORR SNR sweep part 7: alpha = 0.7.

Compares LIME, PatternLocal, Analytical-gradient, Analytical-IG (proposed),
plus plain baselines (no KNN): Plain Gradient, Plain Integrated Gradients,
Plain SmoothGrad, and the paper baseline set (Saliency, DeepLift, GradientShap,
GuidedBackprop, Sobel, Laplace).

Skips any alpha whose model test accuracy < 90%.
Uses shared core in src/paper_repro/xai/eval_common.py.
"""
import os
import sys
import time

import numpy as np

for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "xai", "eval_common.py")):
        sys.path.insert(0, root)
        break

from xai.eval_common import evaluate_alpha, find_file, format_result, save_results

ALPHAS = [0.7]
N_TEST = 600
MIN_ACCURACY = 0.90

METHODS = ["lime", "patternlocal", "analytical_grad",
           "analytical_ig", "plain_grad", "plain_ig",
           "plain_smoothgrad", "saliency", "deeplift",
           "gradient_shap", "guided_backprop", "sobel", "laplace"]


def main():
    t_start = time.time()
    print("=" * 60)
    print("XOR CORR SNR sweep part 7: alpha = 0.7")
    print("=" * 60)

    all_results = {}
    for alpha in ALPHAS:
        print("\n" + "=" * 60)
        print(f"XOR CORR alpha={alpha}")
        print("=" * 60)
        data_path = find_file(f"XOR_CORR_alpha{alpha:.1f}.npz")
        model_path = find_file(f"XOR_CORR_alpha{alpha:.1f}_model.pt")
        if not data_path or not model_path:
            print(f"ERROR: data/model not found for alpha={alpha}")
            continue
        print(f"Data:  {data_path}")
        print(f"Model: {model_path}")
        res = evaluate_alpha(data_path, model_path, n_test=N_TEST,
                             alpha=alpha, min_accuracy=MIN_ACCURACY)
        if res.get("skipped"):
            print(f"  Skipped: {res['skip_reason']}")
            continue
        all_results[f"alpha{alpha:.1f}"] = res
        for method in METHODS:
            m = res[method]
            print(f"  {method:16s} EMD {m['emd_mean']:.4f} "
                  f"IME {m['ime_mean']:.4f}")
        format_result(f"alpha{alpha:.1f}_test_acc", f"{res['test_accuracy']:.4f}")
        for method in METHODS:
            m = res[method]
            format_result(f"alpha{alpha:.1f}_{method}_emd",
                          f"{m['emd_mean']:.4f}")
            format_result(f"alpha{alpha:.1f}_{method}_ime",
                          f"{m['ime_mean']:.4f}")
            tm = res.get("time", {}).get(method, {})
            if tm:
                format_result(f"alpha{alpha:.1f}_{method}_sec_per_sample",
                              f"{tm['sec_per_sample_mean']:.4f}")

    save_results(all_results, "/kaggle/working/sweep_part7_results.json")
    print(f"\nTotal sweep time: {time.time()-t_start:.1f}s")
    print("Sweep part 7 complete.")


if __name__ == "__main__":
    main()
