"""Earth Mover's Distance (EMD) metric following Eq. (10) of the paper.

EMD = OT(|s|, F+) / delta_max
where OT is the optimal transport cost using Euclidean distance as ground metric,
and delta_max is the maximum Euclidean distance between any two pixels.

Performance notes
----------------
The cost matrix M (pixels x pixels) depends only on ``image_shape`` and is
cached, so it is built once per grid rather than once per explanation.

The ground-truth mask F+ is sparse (typically a few hundred pixels out of
4096). Because a zero-mass column can never receive flow, dropping those
columns leaves the optimal transport value unchanged while shrinking the
linear program from N x N to N x |F+|. This is exact and much faster than
running the network simplex on the full dense grid.
"""

import warnings

import numpy as np

# Cache of (coords, delta_max, M) keyed by image_shape. M is None when POT is
# unavailable so the greedy fallback can be used.
_COST_CACHE = {}

_NUM_ITER_MAX = 1_000_000


def _get_cost(image_shape):
    h, w = image_shape
    key = (int(h), int(w))
    if key not in _COST_CACHE:
        coords = np.array(
            [(i, j) for i in range(h) for j in range(w)], dtype=np.float64
        )
        delta_max = float(np.sqrt((h - 1) ** 2 + (w - 1) ** 2))
        M = None
        try:
            import ot

            M = ot.dist(coords, coords, metric="euclidean")
        except ImportError:
            M = None
        _COST_CACHE[key] = (coords, delta_max, M)
    return _COST_CACHE[key]


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

    coords, delta_max, M = _get_cost(image_shape)
    if delta_max == 0:
        return 0.0

    if M is None:
        return _greedy_emd(s, fp, coords) / delta_max

    support = fp > 0
    if not support.any():
        return 0.0

    b = fp[support]
    M_sub = M[:, support]

    try:
        import ot

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ot_cost = ot.emd2(s, b, M_sub, numItermax=_NUM_ITER_MAX)

        if any("numItermax" in str(w.message) for w in caught):
            # Solver hit its iteration cap; fall back to entropic OT.
            try:
                ot_cost = float(ot.sinkhorn2(s, b, M_sub, reg=0.1))
            except Exception:
                ot_cost = _greedy_emd(s, fp, coords) * delta_max

        return ot_cost / delta_max
    except ImportError:
        return _greedy_emd(s, fp, coords) / delta_max


def _greedy_emd(s, fp, coords):
    """Fallback greedy EMD computation when POT is not available.

    Processes source pixels in order of increasing distance to the nearest
    target pixel and greedily assigns mass to the closest target with
    remaining capacity. Approximate, but consistent across methods.
    """
    n = len(s)
    s_copy = s.copy()
    fp_copy = fp.copy()
    total_cost = 0.0

    nonzero_s = np.where(s_copy > 1e-12)[0]
    nonzero_fp = np.where(fp_copy > 1e-12)[0]
    if len(nonzero_fp) == 0:
        return 0.0

    # Order sources by distance to the nearest target for a better heuristic.
    dists_to_nearest = np.sqrt(
        np.sum((coords[nonzero_s][:, None, :] - coords[nonzero_fp][None, :, :]) ** 2, axis=2)
    ).min(axis=1)
    order = np.argsort(dists_to_nearest)

    for i in nonzero_s[order]:
        if s_copy[i] < 1e-12:
            continue
        dists = np.sqrt(np.sum((coords[nonzero_fp] - coords[i]) ** 2, axis=1))
        j_idx = np.argmin(dists)
        j = nonzero_fp[j_idx]
        flow = min(s_copy[i], fp_copy[j])
        total_cost += flow * dists[j_idx]
        s_copy[i] -= flow
        fp_copy[j] -= flow
        if fp_copy[j] < 1e-12:
            nonzero_fp = nonzero_fp[nonzero_fp != j]
            if len(nonzero_fp) == 0:
                break

    return total_cost
