"""Kaggle data-gen driver for Clark et al.'s ORIGINAL XAI-TRIS generator.

Runs braindatalab/xai-tris' `data_generation_process` VERBATIM for one
scenario (both backgrounds, 9 alphas from the PatternLocal paper's Fig-4
grid), converts the records to the sweep pipeline's npz format, trains the
same MLP model as the previous (per-sample-L2) data-gen kernels, and writes
the same training_summary artifact.

Note on faithful value ranges: Clark stores x as torch float16; we cast to
float32 in the npz (same numbers, compatible dtype for the eval core).

Pickles go to /tmp (ephemeral, keeps the persisted /kaggle/working under the
output cap); npzs to /kaggle/working/xai_tris_data.
"""

import json
import os
import shutil

import numpy as np

from conversion import convert_record, load_record, pattern_norm_stats, save_npz

OUTPUT_DIR = "/kaggle/working/xai_tris_data"
PICKLE_DIR = "/tmp/clark_pkl"
N_SAMPLES = 40000
ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]

# The PatternLocal paper's Fig-4 SNR grid, applied to both backgrounds.
# Third list is a placeholder: imagenet backgrounds are disabled
# (use_imagenet -> False), so the [2] index is never selected.
SNRS = [ALPHAS, ALPHAS, ALPHAS]

SCENARIO_PARAMS = {
    "LIN": {
        "scenario": "linear",
        "manipulation_type": "additive",
        "pattern_scale": 8,
    },
    "XOR": {
        "scenario": "xor",
        "manipulation_type": "additive",
        "pattern_scale": 8,
    },
    "RIGID": {
        "scenario": "translations_rotations",
        "manipulation_type": "additive",
        "pattern_scale": 4,
    },
}


def build_config(my_name):
    params = SCENARIO_PARAMS[my_name]
    return {
        "data_scenario": "tetris_data",
        "output_dir": PICKLE_DIR,
        "num_experiments": 1,
        "sample_size": N_SAMPLES,
        "test_split": 0.1,
        "image_shape": [8, 8],       # * image_scale 8 -> 64x64 images
        "image_scale": 8,
        "mean_data": 0,
        "var_data": 0.5,             # np.random.normal(mean, scale) -> sigma 0.5
        "smoothing_sigma": 10,       # CORR background G
        "manipulation": 1.0,
        "positions": [[8, 8], [32, 40]],
        "patterns": ["t", "l"],
        "use_imagenet": False,
        "pattern_scale": params["pattern_scale"],
        "parameterizations": {
            params["scenario"]: {
                "scenario": params["scenario"],
                "manipulation_type": params["manipulation_type"],
                "pattern_scales": [params["pattern_scale"]],
                "snrs": SNRS,
            },
        },
    }


def clark_record_key(my_name, alpha, noise):
    """Mirror of upstream scenario_key:
    f'{scenario}_{scale}d{pattern_scale}p_{alpha}_{background}'."""
    params = SCENARIO_PARAMS[my_name]
    bg = {"WHITE": "uncorrelated", "CORR": "correlated"}[noise]
    return (f"{params['scenario']}_8d{params['pattern_scale']}p_"
            f"{alpha:.1f}_{bg}")


def pickles_for(my_name, noise):
    """Ordered list of (alpha, pickle_path) for the requested background."""
    for alpha in ALPHAS:
        key = clark_record_key(my_name, alpha, noise)
        yield alpha, os.path.join(PICKLE_DIR, f"{key}.pkl")


def as_y(y):
    return np.asarray(y).astype(np.int64).ravel()


def run(scenario, noise):
    """Full data-gen flow for one (scenario, noise) pair.

    The caller must import `clark_xai_tris.generate_data` BEFORE calling
    this so that upstream module-level seeding (SEED=4000002 at import
    time) happens once and generation follows immediately (same effect as
    running upstream's __main__).
    """
    import torch
    from clark_xai_tris.generate_data import data_generation_process
    from models.mlp import MLPModel
    from models.train import train_mlp, evaluate_model

    config = build_config(scenario)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cleanup_pickles()
    os.makedirs(PICKLE_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Clark et al. ORIGINAL XAI-TRIS data generation: {scenario} {noise}")
    print(f"sample_size={N_SAMPLES}, alphas={ALPHAS}")
    print("Generator: global Frobenius Eq.12, float16, SEED=4000002")
    print("=" * 60)

    # 1) Upstream generation, verbatim (one pickle per (alpha, background)).
    data_generation_process(config=config, output_dir=PICKLE_DIR)

    # 2) Convert + train per alpha.
    summary = {}
    for alpha, pkl_path in pickles_for(scenario, noise):
        config_name = f"{scenario}_{noise}_alpha{alpha:.1f}"
        print(f"\n--- {config_name} ---")
        record = load_record(pkl_path)
        arrays, stats = convert_record(record, scenario)
        pnorm = pattern_norm_stats(arrays["masks_test_raw"])
        print(f"  n_train={stats['n_train']} mask_px(Eq13)={stats['mask_px_mean']:.1f} "
              f"(abs05={stats['mask_px_abs05_mean']:.1f}) "
              f"x=[{stats['x_min']:.4f},{stats['x_max']:.4f}] "
              f"|p|={pnorm['pattern_norm_mean']:.3f}")
        print(f"  [RESULT] {config_name}_mask_px: {stats['mask_px_mean']:.1f}")
        print(f"  [RESULT] {config_name}_pattern_norm: {pnorm['pattern_norm_mean']:.4f}")

        X_train, y_train = arrays["X_train"], arrays["y_train"]
        X_val, y_val = arrays["X_val"], arrays["y_val"]
        X_test, y_test = arrays["X_test"], arrays["y_test"]
        save_npz(arrays, os.path.join(OUTPUT_DIR, f"{config_name}.npz"))

        model = MLPModel(input_dim=4096)
        model, history, best_val_acc = train_mlp(
            model, X_train, y_train, X_val, y_val,
            lr=1e-4, batch_size=128, max_epochs=500,
            patience=100, seed=42,
        )
        test_acc = evaluate_model(model, X_test, y_test)
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR, f"{config_name}_model.pt"))

        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  [RESULT] {config_name}_test_acc: {test_acc:.4f}")
        summary[config_name] = {
            "test_acc": test_acc,
            "class_balance": stats["class_balance"],
            "n_train": stats["n_train"], "n_val": stats["n_val"],
            "n_test": stats["n_test"],
            "mask_px_mean": stats["mask_px_mean"],
            "mask_px_std": stats["mask_px_std"],
            "mask_px_abs05_mean": stats["mask_px_abs05_mean"],
            "pattern_norm_mean": pnorm["pattern_norm_mean"],
            "generator": "braindatalab/xai-tris verbatim (global Frobenius)",
            "seed_upstream": 4000002,
        }
        del X_train, y_train, X_val, y_val, X_test, y_test, arrays, record, model
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

    summary_path = os.path.join(
        OUTPUT_DIR, f"training_summary_{scenario}_{noise}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Completed {len(summary)} alphas for {scenario} {noise}")
    print("=" * 60)
    if summary:
        min_acc = min(v["test_acc"] for v in summary.values())
        print(f"Minimum test accuracy: {min_acc:.4f}")
        print(f"[RESULT] min_test_acc: {min_acc:.4f}")
        print(f"[RESULT] {'PASS' if min_acc > 0.90 else 'FAIL'} "
              f"90% accuracy criterion")


def cleanup_pickles():
    shutil.rmtree(PICKLE_DIR, ignore_errors=True)
