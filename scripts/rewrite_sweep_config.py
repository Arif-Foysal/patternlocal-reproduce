#!/usr/bin/env python3
"""Rewrite config.yaml sweep phases from 5 parts to 9 single-alpha parts.

Robust version: loads the YAML, rebuilds the phase dict (replacing every
``phase3_<scenario>_sweep_<n>`` entry with 9 single-alpha entries), and dumps it
back.  This avoids the line-surgery approach that silently duplicated blocks.

Usage: python3 scripts/rewrite_sweep_config.py [path-to-config]
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config.yaml"

# scenario key (as it appears in the phase name) -> (notebook id prefix,
# kernel source)
SCENARIOS = {
    "xor_corr": ("xor_corr_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-xor-corr"),
    "xor_white": ("xor_white_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-xor-white"),
    "lin_corr": ("lin_corr_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-lin-corr"),
    "lin_white": ("lin_white_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-lin-white"),
    "rigid_corr": ("rigid_corr_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-rigid-corr"),
    "rigid_white": ("rigid_white_sweep", "mdariffaysalnayem/patternlocal-repro-xai-tris-rigid-white"),
}


def new_phases():
    out = {}
    for scen, (nb_prefix, src) in SCENARIOS.items():
        for n in range(1, 10):
            alpha = n / 10.0
            out[f"phase3_{scen}_sweep_{n}"] = {
                "description": f"SNR sweep part {n}: alpha = {alpha:.1f}",
                "notebooks": [{
                    "id": f"{nb_prefix}_{n}",
                    "gpu": False,
                    "datasets": [],
                    "success_criteria": [
                        "all 1 alpha complete for 13 methods",
                    ],
                    "kernel_sources": [src],
                }],
            }
    return out


def main():
    cfg = yaml.safe_load(CFG.read_text())
    phases = cfg["pipeline"]["phases"]

    # Drop every old sweep phase (any phase whose name matches
    # phase3_<scenario>_sweep_<n>), then splice the new ones in at the position
    # of the first old sweep phase so ordering stays sensible.
    old_sweep = [k for k in phases
                 if k.startswith("phase3_") and "_sweep_" in k
                 and k.rsplit("_", 1)[-1].isdigit()]
    first_pos = min(phases.keys(), key=lambda k: list(phases).index(k)
                    if k in old_sweep else 10**9) if old_sweep else None

    rebuilt = {}
    inserted = False
    for key, val in phases.items():
        if key in old_sweep:
            if not inserted:
                rebuilt.update(new_phases())
                inserted = True
            continue
        rebuilt[key] = val
    if not inserted:
        rebuilt.update(new_phases())

    cfg["pipeline"]["phases"] = rebuilt
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False,
                                  width=1000, allow_unicode=True))
    print(f"removed {len(old_sweep)} old sweep phases; "
          f"wrote {len(new_phases())} new ones")
    print(f"total phases now: {len(rebuilt)}")


if __name__ == "__main__":
    main()
