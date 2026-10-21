# clark_xai_tris — vendor of the ORIGINAL XAI-TRIS generator (Clark et al.)

Source: https://github.com/braindatalab/xai-tris (GPL-3.0), the authors' own
code for "XAI-TRIS: Non-linear benchmarks to quantify ML explanation
performance" — the benchmark the PatternLocal paper builds on.

Cloned (shallow) 2026-10-21. Upstream `main` at time of clone: repo last
updated 2025-05 (see `reference/xai-tris` locally; reference clone is
gitignored, this directory is the pinned vendored copy used on Kaggle).

## Files

| file | upstream origin |
|---|---|
| `common.py`   | `common.py` |
| `utils.py`    | `utils.py` |
| `data_utils.py` | `data/data_utils.py` |
| `generate_data.py` | `data/generate_data.py` |
| `conversion.py` | (ours) pickle → npz for the sweep pipeline |
| `kaggle_datagen.py` | (ours) kernel entrypoint |

## Modifications (everything else byte-identical)

1. Absolute imports → relative imports (packaging only).
2. `main()` path handling unused; the driver calls
   `data_generation_process(config, output_dir)` directly with the config
   dict built in `kaggle_datagen.py` (same structure and values as the
   shipped `data/data_config.json`, except `use_imagenet: false` and the
   alpha sweep 0.1–0.9 per background taken from the PatternLocal paper).

## Faithfulness notes

- Module-level seeding (SEED = 4000002) is preserved at import time
  (PYTHONHASHSEED / random / np.random), matching upstream numerics order.
- Their code is run **verbatim**: global Frobenius normalization,
  `x = alpha * n_pattern + (1 - alpha) * n_background`, global
  `1/max|x|` rescale, `float16` storage, 90/5/5 stratified split,
  smoothing_sigma=10 CORR background, per-sample gaussian_filter sigma=1.5,
  no 5% support threshold in the generator (the PatternLocal paper's
  threshold lives in the *mask definition*, Eq. 13 — applied in
  `conversion.py`).
- Pattern geometry (from upstream `get_patterns`):
  - t = [[1,0],[1,1],[1,0]], l = [[1,0],[1,0],[1,1]], kron'd by
    `pattern_scale`: scale 8 → 24×16 px (LIN/XOR), scale 4 → 12×8 px
    (RIGID, per upstream config).
  - LIN/XOR positions [[8,8],[32,40]]; RIGID: random rot90(0..3) + random
    position anywhere (**single** tetromino per sample; the PatternLocal
    paper *text* describes a composite aT ± aL — divergent; we follow the
    shipped code and document this).

## Why this exists

Our per-sample-L2 reimplementation of the generator (paper Eq. 12 read as
per-sample normalization) produced a global EMD scale offset (~1.5–2x vs
paper Fig 4) and, in XOR, no floor — see results/FILTER_DIVERGENCE_REPORT.md
and the 2026-10-21 INVESTIGATION_LIVE.md entry. The paper Eq. 12 text
("horizontally concatenated into matrices and normalized by the Frobenius
norm") matches upstream's global normalization exactly; this vendor makes
the reproduction run the paper's actual data pipeline.
