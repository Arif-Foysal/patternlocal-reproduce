"""Importance Mass Error (IME) metric following Eq. (11) of the paper.

IME = 1 - sum(|s_i| for i in F+) / sum(|s_i| for all i)

This penalizes false positives while largely ignoring false negatives.
"""

import numpy as np


def compute_ime(explanation, ground_truth_mask):
    """Compute Importance Mass Error.

    Args:
        explanation: Feature importance map (D,)
        ground_truth_mask: Binary ground truth mask (D,)

    Returns:
        ime_score: IME value in [0, 1]
    """
    s = np.abs(explanation.ravel())
    mask = ground_truth_mask.ravel().astype(bool)

    total_mass = s.sum()
    if total_mass == 0:
        return 1.0

    true_mass = s[mask].sum()
    return 1.0 - true_mass / total_mass
