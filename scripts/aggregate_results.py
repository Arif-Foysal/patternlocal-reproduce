#!/usr/bin/env python3
r"""Aggregate all XAI-TRIS sweep results into one master table.

Reads every results/patternlocal-repro-<scen>-sweep-<n>/sweep_part*_results.json
(one (scenario, alpha) config per file) and emits:

  1. results/figures/master_results.csv
       one row per (config, method): emd_mean/std, ime_mean/std, n_eval,
       accuracy, rank_emd, rank_ime, sec_per_sample, k_ig, k_ratio
  2. results/figures/config_summary.csv
       one row per config: accuracy, n_eval, k_ig, k_ratio, runtime (AIG/PL),
       speedup, gated flag
  3. stdout: the LaTeX fragments used in the Results section
       - K-selection table per scenario
       - runtime table per scenario
       - headline numbers cross-check (rank claims, wins, speedups)

Methods present in the JSONs (13): lime, patternlocal, analytical_grad,
analytical_ig, plain_grad, plain_ig, plain_smoothgrad, saliency, deeplift,
gradient_shap, guided_backprop, sobel, laplace.

Usage:
  .venv/bin/python scripts/aggregate_results.py
"""
import csv
import glob
import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "figures")

SCENARIO_LABELS = {
    "lin-white": "LIN-WHITE",
    "lin-corr": "LIN-CORR",
    "xor-white": "XOR-WHITE",
    "xor-corr": "XOR-CORR",
    "rigid-white": "RIGID-WHITE",
    "rigid-corr": "RIGID-CORR",
}

METHODS = [
    "lime", "patternlocal", "analytical_grad", "analytical_ig",
    "plain_grad", "plain_ig", "plain_smoothgrad", "saliency", "deeplift",
    "gradient_shap", "guided_backprop", "sobel", "laplace",
]


def load_all():
    rows = []
    configs = []
    pattern = os.path.join(RESULTS_DIR,
                           "patternlocal-repro-*-sweep-*",
                           "sweep_part*_results.json")
    for fpath in sorted(glob.glob(pattern)):
        m = re.search(r"patternlocal-repro-(.+)-sweep-(\d+)", fpath)
        scenario, sweep = m.group(1), int(m.group(2))
        with open(fpath) as f:
            payload = json.load(f)
        for key, inner in payload.items():
            if not isinstance(inner, dict):
                continue
            alpha = inner.get("alpha")
            if alpha is None:
                am = re.match(r"alpha([\d.]+)", key)
                alpha = float(am.group(1)) if am else None
            cfg = {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "alpha": alpha,
                "test_accuracy": inner.get("test_accuracy"),
                "n_eval": inner.get("n_eval"),
                "k_ig": inner.get("k_ig") or inner.get("k_grad"),
                "k_ratio": inner.get("k_ratio") or inner.get("pat_k_ratio"),
                "skipped": bool(inner.get("skipped") or inner.get("gated_out")),
            }
            configs.append(cfg)
            if cfg["skipped"]:
                continue
            raw = inner.get("raw", {})
            for method in METHODS:
                stats = (inner.get(method) or {})
                time_m = (inner.get("time") or {}).get(method) or {}
                rec = {
                    "scenario": cfg["scenario"],
                    "scenario_label": cfg["scenario_label"],
                    "alpha": alpha,
                    "method": method,
                    "emd_mean": stats.get("emd_mean"),
                    "emd_std": stats.get("emd_std"),
                    "ime_mean": stats.get("ime_mean"),
                    "ime_std": stats.get("ime_std"),
                    "n_eval": stats.get("n_eval", cfg["n_eval"]),
                    "sec_per_sample": time_m.get("sec_per_sample_mean"),
                    "test_accuracy": cfg["test_accuracy"],
                    "k_ig": cfg["k_ig"],
                    "k_ratio": cfg["k_ratio"],
                    "raw": raw.get(method),
                }
                rows.append(rec)
    return rows, configs


def add_ranks(rows):
    """Rank methods within each config (1 = best, lower is better)."""
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["scenario"], r["alpha"]), []).append(r)
    for cfg_rows in by_cfg.values():
        for metric, rank_key in (("emd_mean", "rank_emd"), ("ime_mean", "rank_ime")):
            present = [r for r in cfg_rows if r.get(metric) is not None]
            present.sort(key=lambda r: r[metric])
            for rank, r in enumerate(present, start=1):
                r[rank_key] = rank
    return rows


def write_csv(path, fieldnames, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k) for k in fieldnames})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, configs = load_all()
    rows = add_ranks(rows)

    master_fields = ["scenario", "scenario_label", "alpha", "method",
                     "emd_mean", "emd_std", "ime_mean", "ime_std",
                     "rank_emd", "rank_ime", "n_eval", "test_accuracy",
                     "k_ig", "k_ratio", "sec_per_sample"]
    write_csv(os.path.join(OUT_DIR, "master_results.csv"), master_fields, rows)

    # Raw per-sample arrays are kept separately for the stats script.
    raw_path = os.path.join(OUT_DIR, "raw_per_sample.json")
    with open(raw_path, "w") as f:
        json.dump({f"{r['scenario']}|{r['alpha']}|{r['method']}": r["raw"]
                   for r in rows if r["raw"]}, f)

    # Config-level summary (K selection + runtime).
    cfg_fields = ["scenario", "scenario_label", "alpha", "test_accuracy",
                  "n_eval", "k_ig", "k_ratio", "skipped"]
    write_csv(os.path.join(OUT_DIR, "config_summary.csv"), cfg_fields, configs)

    # ---------------- stdout report ----------------
    aig = {r["alpha"]: r for r in rows
           if r["method"] == "analytical_ig"}
    cfg_by = {(c["scenario"], c["alpha"]): c for c in configs}

    print("== Evaluated configs:", sum(1 for c in configs if not c["skipped"]),
          "| gated:", sum(1 for c in configs if c["skipped"]))

    # K-selection table
    print("\n== K-selection per scenario (alpha: K) ==")
    for scen in SCENARIO_LABELS:
        alphas = sorted({r["alpha"] for r in rows if r["scenario"] == scen})
        ks = [cfg_by[(scen, a)]["k_ig"] for a in alphas]
        print(f"  {SCENARIO_LABELS[scen]:11s} " +
              " ".join(f"{a}:{k}" for a, k in zip(alphas, ks)))

    # Runtime table
    print("\n== Runtime (s/sample): AIG / PL / speedup ==")
    for scen in SCENARIO_LABELS:
        alphas = sorted({r["alpha"] for r in rows if r["scenario"] == scen})
        for a in alphas:
            ta = next((r["sec_per_sample"] for r in rows
                       if r["scenario"] == scen and r["alpha"] == a
                       and r["method"] == "analytical_ig"), None)
            tp = next((r["sec_per_sample"] for r in rows
                       if r["scenario"] == scen and r["alpha"] == a
                       and r["method"] == "patternlocal"), None)
            if ta and tp:
                print(f"  {SCENARIO_LABELS[scen]:11s} a={a}: "
                      f"{ta:.4f} / {tp:.4f} / {tp/ta:.1f}x")

    # Headline checks
    print("\n== Headline checks ==")
    aig_rows = [r for r in rows if r["method"] == "analytical_ig"]
    wins = [(r["scenario_label"], r["alpha"]) for r in aig_rows
            if r.get("rank_emd") == 1 and r.get("rank_ime") == 1]
    print("  AIG double wins (EMD r1 & IME r1):", wins)
    ime_r12 = sum(1 for r in aig_rows if r.get("rank_ime") in (1, 2))
    print(f"  AIG IME rank 1-2 at {ime_r12}/{len(aig_rows)} points")
    ta_all = [r["sec_per_sample"] for r in aig_rows if r["sec_per_sample"]]
    tp_all = [r["sec_per_sample"] for r in rows
              if r["method"] == "patternlocal" and r["sec_per_sample"]]
    print(f"  AIG runtime range: {min(ta_all):.4f}-{max(ta_all):.4f} s/sample")
    print(f"  PL  runtime range: {min(tp_all):.4f}-{max(tp_all):.4f} s/sample")


if __name__ == "__main__":
    main()
