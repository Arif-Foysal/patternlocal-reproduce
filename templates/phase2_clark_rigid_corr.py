#!/usr/bin/env python3
"""Clark et al. ORIGINAL XAI-TRIS data generation: RIGID CORR.

Runs braindatalab/xai-tris (the benchmark's authors' own generator, GPL-3.0)
verbatim with the PatternLocal paper's alpha grid 0.1-0.9 for one scenario
and background, converts to the sweep pipeline npz format, and trains the
same MLP as the previous data-gen kernels. Accuracy gate: > 90%.

Rationale: our per-sample-L2 reimplementation diverged from the paper's own
Eq. 12 (global Frobenius normalization; see INVESTIGATION_LIVE.md 2026-10-21).
This kernel replaces the generator with the authors' actual code.
"""
import os
import sys

for root, dirs, files in os.walk("/kaggle/input"):
    if "clark_xai_tris" in dirs and "models" in dirs:
        root_dir = root
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "clark_xai_tris"))
        break
else:
    raise RuntimeError("clark_xai_tris/models package not found in mounts")

SCENARIO = "RIGID"
NOISE = "CORR"


def main():
    # 1) Import upstream FIRST so its module-level seeding (SEED=4000002)
    #    happens at import time, exactly like running upstream's __main__.
    import clark_xai_tris.generate_data  # noqa: F401

    import kaggle_datagen
    kaggle_datagen.run(SCENARIO, NOISE)


if __name__ == "__main__":
    main()
