"""Earth Mover's Distance (EMD) metric following Eq. (10) of the paper.

EMD = OT(|s|, F+) / delta_max
where OT is the optimal transport cost using Euclidean distance as ground metric,
and delta_max is the maximum Euclidean distance between any two pixels.
"""

import numpy as np


def compute_emd(explanation, ground_truth_mask, image_shape=(64, 64)):
    """Compute EMD between explanation and ground truth.

    Args:
        explanation: Feature importance map (D,), will be normalized to |s|
        ground_truth_mask: Binary ground truth mask (D,)
        image_shape: Shape of the image (H, W)

    Returns:
        emd_score: Normalized EMD value
    """
    s = np.abs(explanation.ravel())
    s_sum = s.sum()
    if s_sum > 0:
        s = s / s_sum
    else:
        s = np.ones_like(s) / len(s)

    fp = ground_truth_mask.ravel().astype(np.float64)
    fp_sum = fp.sum()
    if fp_sum > 0:
        fp = fp / fp_sum
    else:
        fp = np.ones_like(fp) / len(fp)

    h, w = image_shape
    coords = np.array([(i, j) for i in range(h) for j in range(w)], dtype=np.float64)
    delta_max = np.sqrt((h - 1)**2 + (w - 1)**2)
    if delta_max == 0:
        return 0.0

    try:
        import ot
        M = ot.dist(coords, coords, metric="euclidean")
        ot_cost = ot.emd2(s, fp, M)
        return ot_cost / delta_max
    except ImportError:
        return _greedy_emd(s, fp, coords) / delta_max


def _greedy_emd(s, fp, coords):
    """Fallback greedy EMD computation when POT is not available."""
    n = len(s)
    s_copy = s.copy()
    fp_copy = fp.copy()
    total_cost = 0.0

    nonzero_s = np.where(s_copy > 1e-12)[0]
    nonzero_fp = np.where(fp_copy > 1e-12)[0]

    for i in nonzero_s:
        if s_copy[i] < 1e-12:
            continue
        dists = np.sqrt(np.sum((coords[nonzero_fp] - coords[i])**2, axis=1)) if len(nonzero_fp) > 0 else []
        if len(dists) == 0:
            break
        j_idx = np.argmin(dists)
        j = nonzero_fp[j_idx]
        flow = min(s_copy[i], fp_copy[j])
        total_cost += flow * dists[j_idx]
        s_copy[i] -= flow
        fp_copy[j] -= flow
        if fp_copy[j] < 1e-12:
            nonzero_fp = nonzero_fp[nonzero_fp != j]

    return total_cost
