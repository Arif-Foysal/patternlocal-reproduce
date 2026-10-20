#!/usr/bin/env python3
"""Regenerate the six XAI-TRIS SNR-sweep template sets as 9 single-alpha parts.

Each scenario previously had 5 parts (alpha 0.1-0.2, 0.3-0.4, 0.5-0.6, 0.7-0.8,
0.9).  This rewrites them into 9 parts, one per alpha, keeping the same kernel
slug convention ``<scenario>-sweep-<n>`` and JSON name
``sweep_part<n>_results.json``.

Usage: python3 scripts/split_sweeps_into_9.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"

ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]

# (directory, existing template to clone, scenario label shown in the header,
#  npz prefix)
SCENARIOS = [
    ("xor_evaluation/xor_corr_sweep", "xor_corr_sweep_5.py", "XOR CORR", "XOR_CORR"),
    ("xor_evaluation/xor_white_sweep", "xor_white_sweep_5.py", "XOR WHITE", "XOR_WHITE"),
    ("lin_evaluation/lin_corr_sweep", "lin_corr_sweep_5.py", "LIN CORR", "LIN_CORR"),
    ("lin_evaluation/lin_white_sweep", "lin_white_sweep_5.py", "LIN WHITE", "LIN_WHITE"),
    ("rigid_evaluation/rigid_corr_sweep", "rigid_corr_sweep_5.py", "RIGID CORR", "RIGID_CORR"),
    ("rigid_evaluation/rigid_white_sweep", "rigid_white_sweep_5.py", "RIGID WHITE", "RIGID_WHITE"),
]

# The block that must be regenerated per part.
TEMPLATE = '''#!/usr/bin/env python3
"""{label} SNR sweep part {n}: alpha = {alpha:.1f}.

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

ALPHAS = [{alpha:.1f}]
N_TEST = 600
MIN_ACCURACY = 0.90

METHODS = ["lime", "patternlocal", "analytical_grad",
           "analytical_ig", "plain_grad", "plain_ig",
           "plain_smoothgrad", "saliency", "deeplift",
           "gradient_shap", "guided_backprop", "sobel", "laplace"]


def main():
    t_start = time.time()
    print("=" * 60)
    print("{label} SNR sweep part {n}: alpha = {alpha:.1f}")
    print("=" * 60)

    all_results = {{}}
    for alpha in ALPHAS:
        print("\\n" + "=" * 60)
        print(f"{label} alpha={{alpha}}")
        print("=" * 60)
        data_path = find_file(f"{prefix}_alpha{{alpha:.1f}}.npz")
        model_path = find_file(f"{prefix}_alpha{{alpha:.1f}}_model.pt")
        if not data_path or not model_path:
            print(f"ERROR: data/model not found for alpha={{alpha}}")
            continue
        print(f"Data:  {{data_path}}")
        print(f"Model: {{model_path}}")
        res = evaluate_alpha(data_path, model_path, n_test=N_TEST,
                             alpha=alpha, min_accuracy=MIN_ACCURACY)
        if res.get("skipped"):
            print(f"  Skipped: {{res['skip_reason']}}")
            continue
        all_results[f"alpha{{alpha:.1f}}"] = res
        for method in METHODS:
            m = res[method]
            print(f"  {{method:16s}} EMD {{m['emd_mean']:.4f}} "
                  f"IME {{m['ime_mean']:.4f}}")
        format_result(f"alpha{{alpha:.1f}}_test_acc", f"{{res['test_accuracy']:.4f}}")
        for method in METHODS:
            m = res[method]
            format_result(f"alpha{{alpha:.1f}}_{{method}}_emd",
                          f"{{m['emd_mean']:.4f}}")
            format_result(f"alpha{{alpha:.1f}}_{{method}}_ime",
                          f"{{m['ime_mean']:.4f}}")
            tm = res.get("time", {{}}).get(method, {{}})
            if tm:
                format_result(f"alpha{{alpha:.1f}}_{{method}}_sec_per_sample",
                              f"{{tm['sec_per_sample_mean']:.4f}}")

    save_results(all_results, "/kaggle/working/sweep_part{n}_results.json")
    print(f"\\nTotal sweep time: {{time.time()-t_start:.1f}}s")
    print("Sweep part {n} complete.")


if __name__ == "__main__":
    main()
'''


def main():
    written = []
    for rel, base, label, prefix in SCENARIOS:
        d = TEMPLATES / rel
        if not d.is_dir():
            print(f"skip (missing dir): {d}")
            continue
        for n, alpha in enumerate(ALPHAS, start=1):
            path = d / f"{Path(base).stem.rsplit('_', 1)[0]}_{n}.py"
            path.write_text(TEMPLATE.format(label=label, prefix=prefix,
                                            n=n, alpha=alpha))
            written.append(path)
        # remove the old multi-alpha parts that do not match the new set
        stem = Path(base).stem.rsplit("_", 1)[0]
        new_names = {p.name for p in written}
        for old in d.glob(f"{stem}_*.py"):
            if old.name not in new_names:
                old.unlink()
                print(f"removed old: {old.relative_to(ROOT)}")
    print(f"\nwrote {len(written)} templates")


if __name__ == "__main__":
    main()
