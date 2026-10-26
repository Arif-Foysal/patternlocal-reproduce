#!/usr/bin/env python3
r"""Paired statistical analysis for the Results section.

Input:  results/figures/raw_per_sample.json   (from aggregate_results.py)
        keyed "<scenario>|<alpha>|<method>" -> [[emd, ime], ...] per sample.

Analysis: for each evaluated scenario, at representative SNR levels
(0.2, 0.5, 0.9, or the nearest available), run two-sided Wilcoxon
signed-rank tests on the paired per-sample EMD and IME scores of
PatternGrad (analytical_ig) against PatternLocal, LIME, plain IG, and the
analytical_grad ablation. p-values are Holm-corrected within each
(config, metric) family of 4 comparisons. Effect size is the rank-biserial
correlation r = 1 - 2*W+/2T  (signed; negative favors PatternGrad, i.e.
lower error).

Output:
  - results/figures/stats_fragments.tex  (LaTeX table fragments)
  - stdout summary

Usage:
  .venv/bin/python scripts/aggregate_results.py && \
  .venv/bin/python scripts/results_stats.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "figures")

SCENARIOS = ["lin-white", "lin-corr", "xor-white", "xor-corr",
             "rigid-white", "rigid-corr"]
SCENARIO_LABELS = {
    "lin-white": "LIN-WHITE", "lin-corr": "LIN-CORR",
    "xor-white": "XOR-WHITE", "xor-corr": "XOR-CORR",
    "rigid-white": "RIGID-WHITE", "rigid-corr": "RIGID-CORR",
}
AVAILABLE = {
    "lin-white": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lin-corr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "xor-white": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "xor-corr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "rigid-white": [0.7, 0.8, 0.9],
    "rigid-corr": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
BASELINES = [("patternlocal", "PatternLocal"), ("lime", "LIME"),
             ("plain_ig", "IG"), ("analytical_grad", "PG-Grad (abl.)")]


def holm(pvals):
    """Holm-Bonferroni step-down correction; returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(x, y):
    """Rank-biserial correlation for paired samples x (ours), y (baseline)."""
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    return (r_plus - r_minus) / (r_plus + r_minus)


def fmt_p(p):
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}"


def main():
    with open(os.path.join(OUT_DIR, "raw_per_sample.json")) as f:
        raw = dict(json.load(f))

    tex = []
    print("== Wilcoxon signed-rank, PatternGrad vs baselines ==")
    tex.append("% ---- Wilcoxon table fragment (auto-generated) ----")
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Paired two-sided Wilcoxon signed-rank tests "
               r"comparing PatternGrad against the leading baselines on "
               r"per-sample EMD and IME at high signal-to-noise ratio "
               r"($\alpha$ = 0.9; RIGID-WHITE: $\alpha$ = 0.9; "
               r"$N_{\text{test}}$ = 600 paired instances per cell). "
               r"$p$-values are Holm-corrected across the four comparisons "
               r"per cell; $r$ is the rank-biserial effect size (positive "
               r"$r$: PatternGrad attains the lower error, negative $r$: "
               r"the baseline does). $W$ is the smaller of the two rank "
               r"sums, so $W = 0$ with $|r| = 1$ indicates complete "
               r"pairwise dominance by one method.}")
    tex.append(r"\label{tab:wilcoxon}")
    tex.append(r"\begin{tabular}{ll rrr rrr}")
    tex.append(r"\toprule")
    tex.append(r"& & \multicolumn{3}{c}{EMD} & \multicolumn{3}{c}{IME}\\")
    tex.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    tex.append(r"\textbf{Scenario} & \textbf{Baseline} & "
               r"$W$ & $p$ & $r$ & $W$ & $p$ & $r$\\")
    tex.append(r"\midrule")

    for scen in SCENARIOS:
        alphas = AVAILABLE[scen]
        alpha = 0.9 if 0.9 in alphas else alphas[-1]
        n = len(raw[f"{scen}|{alpha}|analytical_ig"])
        # gather all p-values first for Holm within (config, metric)
        pending = []  # (metric, bi, stat, p)
        for metric_idx, metric in enumerate(("emd", "ime")):
            ours = np.array([s[metric_idx] for s in raw[f"{scen}|{alpha}|analytical_ig"]])
            for mkey, _ in BASELINES:
                theirs = np.array([s[metric_idx] for s in raw[f"{scen}|{alpha}|{mkey}"]])
                assert len(theirs) == n
                stat, p = wilcoxon(ours, theirs, zero_method="wilcox",
                                   alternative="two-sided", method="approx")
                pending.append((metric_idx, mkey, stat, p))
        adj = holm([p for _, _, _, p in pending])
        # emit rows grouped by baseline
        for bi, (mkey, mlabel) in enumerate(BASELINES):
            cells = []
            for metric_idx in (0, 1):
                entry = next(e for e in pending
                             if e[0] == metric_idx and e[1] == mkey)
                adj_p = adj[pending.index(entry)]
                r = rank_biserial(
                    [s[metric_idx] for s in raw[f"{scen}|{alpha}|analytical_ig"]],
                    [s[metric_idx] for s in raw[f"{scen}|{alpha}|{mkey}"]])
                cells.append(f"{entry[2]:.0f} & {fmt_p(adj_p)} & {r:+.2f}")
            tex.append(f"{SCENARIO_LABELS[scen]} & {mlabel} & " + " & ".join(cells) + r"\\")
        if scen != SCENARIOS[-1]:
            tex.append(r"\addlinespace")
        # stdout mirror
        for metric_idx, metric in enumerate(("EMD", "IME")):
            line = []
            for mkey, mlabel in BASELINES:
                entry = next(e for e in pending
                             if e[0] == metric_idx and e[1] == mkey)
                adj_p = adj[pending.index(entry)]
                line.append(f"{mlabel}: p={adj_p:.2e}")
            print(f"  {SCENARIO_LABELS[scen]} {metric} a={alpha}: " + "; ".join(line))

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT_DIR, "stats_fragments.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\nWrote results/figures/stats_fragments.tex")


if __name__ == "__main__":
    main()
