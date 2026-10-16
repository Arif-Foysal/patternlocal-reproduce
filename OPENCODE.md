# PatternLocal Reproduction Pipeline — Operating Guide

## Project
Reproduce the main experimental findings (Figure 4) of the NeurIPS 2025 paper
"Minimizing False-Positive Attributions in Explanations of Non-Linear Models"
(PatternLocal) using a faithful re-implementation in `src/paper_repro/`.

## Architecture
- `src/paper_repro/` — core library. Uploaded to Kaggle as the project's core
  dataset `mdariffaysalnayem/patternlocal-repro-core`. `pipeline.py upload-src`
  flattens `src/paper_repro/*` into the dataset root, so the notebooks can
  import `data`, `models`, `xai`, `metrics`, `experiments` directly.
  - `data/xai_tris.py` — XAI-TRIS generator (LIN/XOR/RIGID x WHITE/CORR x alpha)
  - `models/mlp.py`, `models/train.py` — MLP 128-64-32-16, Adam, early stopping
  - `xai/patternlocal.py` — paper Eq. (7): `a = C w / (w^T C w + lambda)`
  - `xai/baselines.py` — Sobel/Laplace/gradient baselines
  - `metrics/emd.py`, `metrics/ime.py` — EMD (Eq. 10) and IME (Eq. 11)
  - `experiments/xor_toy.py` — Section 3.2 XOR suppressor toy problem
- `pipeline.py` — CLI: upload-src / generate / push / status / wait / fetch /
  tail / results / upload-results.
- `config.yaml` — Kaggle username, dataset slugs, phases, success criteria.
- `templates/` — pristine notebook templates (checked in).
- `notebooks/` — working copies you edit (gitignored).
- `results/` — downloaded Kaggle outputs (gitignored).

## Environment
The Kaggle API token lives in `.env` (gitignored) and is loaded automatically by
`pipeline.py`. Create the venv and install the pipeline deps:
```bash
python3 -m venv .venv
.venv/bin/pip install kaggle pyyaml python-dotenv
source .venv/bin/activate
python pipeline.py push templates/xor_toy.py   # token loads from .env
```
If `.env` is missing, create it with `KAGGLE_API_TOKEN=KGAT_...` from
kaggle.com/settings.

## Running an experiment
1. Copy a template to `notebooks/<phase>.py` (or `python pipeline.py generate <phase>`).
2. Edit the working copy if needed (parameters, etc.).
3. `python pipeline.py push notebooks/<phase>.py`
4. `python pipeline.py wait <phase>` — single blocking call; prints one line per
   state change. Returns 0 on complete, non-zero on error/timeout.
5. `python pipeline.py fetch <phase>`
6. Read `results/<slug>/*.json` and grep `[RESULT]` lines to judge success. Use
   `python pipeline.py tail <phase>` for the last 40 log lines if you need to
   debug — never `cat` the full `.log`.

## Iteration protocol
When a notebook fails or success criteria aren't met:
- **Notebook bug** (syntax, shapes, logic) -> edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** -> edit `src/paper_repro/`, run
  `python pipeline.py upload-src`, **wait ~5 min**, re-push the notebook
- **Parameter issue** -> tweak the notebook's config section, re-push
- **After 5 retries on the same notebook** -> stop and ask the human

## Between phases
When phase N's outputs feed phase N+1:
1. `python pipeline.py upload-results phaseN`
2. Add the new dataset slug to `config.yaml` under `kaggle.datasets`
3. In `config.yaml` under the next notebook, add the key to its `datasets:` list

## Output conventions (do not break)
- Each template defines `format_result(key, value)` -> greppable `[RESULT]` line
- Each template defines `save_results(dict, path)` -> JSON to `/kaggle/working/`
- Plots saved as PNG to `/kaggle/working/`

## Token-cost discipline (read this — it's why the harness exists)
- **Only three signals matter for "did this work?"**: `results/<slug>/*.json`,
  `[RESULT]` lines (`pipeline.py results <phase>` or `fetch`), and the kernel
  state from `pipeline.py wait`. Don't Read raw `.log` files into context; use
  `pipeline.py tail <phase>` if you need recent log lines.
- **Use `wait`, not a polling loop.** One `python pipeline.py wait <phase>` is
  dramatically cheaper in context tokens than running `status` 6x in a row.
- **Keep `pipeline.py`, `src/paper_repro/`, `templates/` stable across
  iterations.** They are the cacheable prompt prefix; edit them only for durable
  changes. Iterate on `notebooks/<phase>.py`.
- **Don't print large arrays.** Truncate to a few items + length. `_pretty_print`
  already enforces this for fetched JSON.

## When to stop and ask
- Phase boundary transitions (gate decisions)
- 5 retries exhausted
- Borderline / ambiguous results
- Anything that needs scientific judgement
- Before destructive actions (deleting kernels, force-pushing, etc.)

## Gotchas
- **Kaggle dataset propagation lag**: new dataset versions take ~5 minutes to
  mount in kernels. After `upload-src` or `upload-results`, wait before re-pushing.
- **`datasets create` silently no-ops** on existing datasets. The CLI handles
  this by trying `datasets version` first; don't bypass that pattern.
- **Kaggle datasets**: when uploading datasets with subdirectories to Kaggle,
  the CLI requires `--dir-mode zip` or `--dir-mode tar`. Without it, directories
  are skipped. `pipeline.py` includes `--dir-mode zip`.
- **Mount paths drift**: always `os.walk` to find the package root (the dir
  containing `xai/patternlocal.py`) rather than hardcoding a slug.
- **GPU quota** is ~30h/week on free Kaggle. Set `gpu: false` in `config.yaml`
  for any notebook that doesn't need it.
- **Auth**: `pipeline.py` loads `KAGGLE_API_TOKEN` from `.env` automatically. If
  commands fail with 401/403, verify `.env` exists at repo root and contains a
  valid `KGAT_*` token.
- **Determinism**: `data/xai_tris.py` seeds the split with a fixed seed, but the
  data-generation seed in the scenario templates uses `hash(config_name)`,
  which is randomized per process. Use `zlib.crc32` or a fixed mapping if exact
  reproducibility across runs is required.
