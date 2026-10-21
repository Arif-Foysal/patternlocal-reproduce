"""Convert Clark et al. XAI-TRIS DataRecord pickles into the sweep pipeline's npz format.

Output contract (identical to the previous per-sample-L2 data-gen kernels):
    {SCENARIO}_{NOISE}_alpha{a}.npz  with keys
        X_train f32, y_train i64, X_val f32, y_val i64,
        X_test f32, y_test i64,
        masks_test f32 (binary), masks_val f32 (binary),
        masks_test_raw f16, masks_val_raw f16  (Clark's continuous blurred
            pattern images, unnormalised -> diagnostics/provenance),
        masks_test_abs05 f32, masks_val_abs05 f32  (Clark's own eval rule,
            absolute >= 0.05 -> for a one-off diagnostic comparison).

Mask semantics:
    * Paper Eq. 13/14 (used for the main binary masks): F+ = pixels where the
      smoothed signal is non-zero, where the smoothing H INCLUDES the
      relative 5% max-support threshold  ->  mask = |m| >= 0.05 * max|m|
      per sample, sign-preserving.
    * LIN (Eq. 14): the union of the T- and L- tetromino supports for BOTH
      classes ("presence of aT is as informative as absence of aL"), so the
      same union mask is used for every sample.
"""

import os
import pickle
import numpy as np

REL_THRESHOLD = 0.05  # paper: "maximum support threshold of 5%"


def _record_to_arrays(record):
    """Clark DataRecord -> (X_*, y_*, masks_* raw np arrays)."""
    x_train = np.asarray(record.x_train, dtype=np.float16).astype(np.float32)
    y_train = np.asarray(record.y_train).astype(np.int64).ravel()
    x_val = np.asarray(record.x_val, dtype=np.float16).astype(np.float32)
    y_val = np.asarray(record.y_val).astype(np.int64).ravel()
    x_test = np.asarray(record.x_test, dtype=np.float16).astype(np.float32)
    y_test = np.asarray(record.y_test).astype(np.int64).ravel()
    m_val = np.asarray(record.masks_val, dtype=np.float32)
    m_test = np.asarray(record.masks_test, dtype=np.float32)
    return (
        (x_train, y_train, x_val, y_val, x_test, y_test),
        (m_val, m_test),
    )


def _rel5_binary(m):
    """Sign-preserving relative 5% threshold (paper Eq. 13/14)."""
    peak = np.max(np.abs(m), axis=1, keepdims=True)
    return (np.abs(m) >= REL_THRESHOLD * peak).astype(np.float32)


def _abs05_binary(m):
    """Clark's own eval threshold: absolute >= 0.05."""
    return (m >= REL_THRESHOLD).astype(np.float32)


def convert_record(record, scenario):
    """One Clark DataRecord -> dict of npz-ready arrays.

    Returns (arrays, stats) where arrays maps npz-key -> ndarray.
    """
    (x_train, y_train, x_val, y_val, x_test, y_test), (m_val, m_test) = (
        _record_to_arrays(record))

    t_mask = _rel5_binary(m_test)
    v_mask = _rel5_binary(m_val)

    if scenario == "LIN":
        # Eq. 14: union of aT and aL supports for every sample. Each stored
        # Clark mask only contains the present tetromino, but T and L sit at
        # fixed positions, so the union is class-independent. Restore the
        # missing counterpart by OR-ing the per-class masks.
        support_t = t_mask[y_test == 0].any(axis=0, keepdims=True)
        support_l = t_mask[y_test == 1].any(axis=0, keepdims=True)
        union = np.maximum(support_t, support_l)
        t_mask = np.repeat(union, t_mask.shape[0], axis=0)
        union_v = np.maximum(
            v_mask[y_val == 0].any(axis=0, keepdims=True),
            v_mask[y_val == 1].any(axis=0, keepdims=True))
        v_mask = np.repeat(union_v, v_mask.shape[0], axis=0)

    arrays = {
        "X_train": x_train, "y_train": y_train,
        "X_val": x_val, "y_val": y_val,
        "X_test": x_test, "y_test": y_test,
        "masks_test": t_mask.astype(np.float32),
        "masks_val": v_mask.astype(np.float32),
        "masks_test_raw": m_test.astype(np.float16),
        "masks_val_raw": m_val.astype(np.float16),
        "masks_test_abs05": _abs05_binary(m_test).astype(np.float32),
        "masks_val_abs05": _abs05_binary(m_val).astype(np.float32),
    }

    stats = {
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "class_balance": float(np.mean(y_test)),
        "mask_px_mean": float(t_mask.sum(axis=1).mean()),
        "mask_px_std": float(t_mask.sum(axis=1).std()),
        "mask_px_abs05_mean": float(arrays["masks_test_abs05"].sum(axis=1).mean()),
        "x_min": float(x_test.min()), "x_max": float(x_test.max()),
        "x_dtype_input": str(np.asarray(record.x_train).dtype),
    }
    return arrays, stats


def load_record(pickle_path):
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        key = next(iter(payload))
        return payload[key]
    return payload


def save_npz(arrays, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def pattern_norm_stats(m_test):
    """Per-sample blurred-pattern Frobenius norms (the UNNORMALIZED signal a(n)).

    Diagnoses the alpha-mapping difference vs our old per-sample-L2 scheme:
    in the paper/Clark semantics the per-sample SNR scales with ||p_i||,
    which varies only weakly here (fixed geometry + blur).
    """
    p = np.asarray(m_test, dtype=np.float64)
    norms = np.linalg.norm(p, axis=1)
    return {
        "pattern_norm_mean": float(norms.mean()),
        "pattern_norm_std": float(norms.std()),
        "pattern_norm_min": float(norms.min()),
        "pattern_norm_max": float(norms.max()),
    }
