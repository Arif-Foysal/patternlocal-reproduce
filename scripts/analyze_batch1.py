"""Analyze the first post-refactor batch (6 sweeps) against the paper's claims.

Reads the fetched JSONs under /tmp/b1/<kernel>/sweep_part*_results.json and
prints EMD/IME tables plus the key comparisons the paper makes (PatternLocal vs
LIME, and vs the filter baselines Sobel/Laplace for RIGID CORR).
"""

import glob
import json
import os
import re

ROOT = "/tmp/b1"

PAPER_METHODS = ["lime", "plain_ig", "plain_grad", "deeplift",
                 "gradient_shap", "guided_backprop", "sobel", "laplace"]
OURS = ["patternlocal", "analytical_ig", "analytical_grad"]
ABLATIONS = ["plain_smoothgrad", "saliency"]

ALL = ["lime"] + OURS + ["plain_grad", "plain_ig", "plain_smoothgrad",
                          "saliency", "deeplift", "gradient_shap",
                          "guided_backprop", "sobel", "laplace"]


def load():
    rows = {}
    for f in glob.glob(f"{ROOT}/*/sweep_part*_results.json"):
        for _, r in json.load(open(f)).items():
            rows[r["config"]] = r
    return rows


def sort_key(c):
    m = re.search(r"([A-Z]+)_([A-Z]+)_alpha([0-9.]+?)\.npz", c)
    return (m.group(1), m.group(2), float(m.group(3)))


def main():
    rows = load()
    print(f"found {len(rows)} config(s)\n")

    for c in sorted(rows, key=sort_key):
        r = rows[c]
        print("=" * 78)
        print(f"{c}   test_acc={r['test_accuracy']:.4f}  n_eval={r['n_eval']}"
              f"  lime_bw={r['lime_bandwidth']}  pat_k={r['pat_k_ratio']}"
              f"  k_grad={r['k_grad']}  k_ig={r['k_ig']}")
        print("=" * 78)
        if r.get("skipped"):
            print("  SKIPPED:", r.get("skip_reason"))
            continue

        hdr = f"{'method':18s} {'EMD':>8s} {'±':>7s} {'IME':>8s} {'±':>7s}"
        print(hdr)
        print("-" * len(hdr))
        for m in ALL:
            if m not in r:
                continue
            d = r[m]
            mark = ""
            if m == "patternlocal":
                mark = "  <== proposed"
            print(f"{m:18s} {d['emd_mean']:8.4f} {d['emd_std']:7.4f} "
                  f"{d['ime_mean']:8.4f} {d['ime_std']:7.4f}{mark}")

        # ---- decisive comparisons -----------------------------------------
        print("\n  Key comparisons (EMD lower=better, IME lower=better):")
        pl = r["patternlocal"]
        lime = r["lime"]
        print(f"    PatternLocal vs LIME      EMD "
              f"{'WIN ' if pl['emd_mean'] < lime['emd_mean'] else 'LOSE'} "
              f"({pl['emd_mean']:.4f} vs {lime['emd_mean']:.4f})   IME "
              f"{'WIN ' if pl['ime_mean'] < lime['ime_mean'] else 'LOSE'} "
              f"({pl['ime_mean']:.4f} vs {lime['ime_mean']:.4f})")
        for base in ("sobel", "laplace"):
            if base in r:
                b = r[base]
                print(f"    PatternLocal vs {base:8s}   EMD "
                      f"{'WIN ' if pl['emd_mean'] < b['emd_mean'] else 'LOSE'} "
                      f"({pl['emd_mean']:.4f} vs {b['emd_mean']:.4f})")
        # best method overall
        best = min(ALL, key=lambda m: r[m]["emd_mean"] if m in r else 1e9)
        besti = min(ALL, key=lambda m: r[m]["ime_mean"] if m in r else 1e9)
        print(f"    best EMD: {best} ({r[best]['emd_mean']:.4f})   "
              f"best IME: {besti} ({r[besti]['ime_mean']:.4f})")
        print()


if __name__ == "__main__":
    main()
