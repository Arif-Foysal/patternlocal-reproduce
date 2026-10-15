"""Faithful PatternLocal implementation following Eq. (7) of the paper.

Paper Eq. (7): a_l2 = Cov_Pi[x, y~] / (Var_Pi[y~] + lambda)
where y~ = w^T x, expectations are kernel-weighted over local neighborhood.

This implementation uses the local covariance matrix formulation:
a = C * w / (w^T * C * w + lambda)
where C = Cov_Pi[x, x] is the local covariance matrix.
"""

import numpy as np
from scipy.spatial.distance import cdist


def kernel_weights(distances, bandwidth, kernel="gaussian"):
    """Compute kernel weights from distances."""
    u = distances / bandwidth
    if kernel == "gaussian":
        w = np.exp(-0.5 * u**2)
    elif kernel == "epanechnikov":
        w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return w


def patternlocal_explain(
    lime_weights,
    lime_intercept,
    instance,
    X_train,
    bandwidth=None,
    kernel="gaussian",
    k_ratio=0.1,
    lam=1e-5,
    use_global_cov=False,
):
    """Compute PatternLocal explanation following paper Eq. (7).

    Args:
        lime_weights: LIME coefficient vector (D,)
        lime_intercept: LIME intercept (scalar)
        instance: Instance to explain (D,)
        X_train: Training data (N, D)
        bandwidth: Kernel bandwidth. If None, auto-estimate as median distance.
        kernel: Kernel function ('gaussian' or 'epanechnikov')
        k_ratio: Ratio of samples to use for local estimation (0, 1]
        lam: Ridge regularization parameter (default 1e-5)
        use_global_cov: If True, use global covariance (all data, uniform weights).
                       Matches theoretical XOR toy problem in Section 3.2.

    Returns:
        pattern_weights: PatternLocal explanation (D,)
    """
    n_total, d = X_train.shape

    if use_global_cov:
        # Global covariance: uniform weights over all training data
        X_local = X_train
        weights = np.ones(n_total) / n_total
    else:
        k = max(1, int(k_ratio * n_total))
        distances = cdist(X_train, instance.reshape(1, -1), metric="euclidean").ravel()
        sorted_idx = np.argsort(distances)
        k_idx = sorted_idx[:k]
        distances_k = distances[k_idx]

        if bandwidth is None:
            bandwidth = np.median(distances_k)
            if bandwidth == 0:
                bandwidth = np.finfo(float).eps

        weights = kernel_weights(distances_k, bandwidth, kernel)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError("Kernel weights sum to zero. Increase k or adjust bandwidth.")
        weights /= weight_sum

        X_local = X_train[k_idx]

    weighted_mean = np.average(X_local, weights=weights, axis=0)
    diffs = X_local - weighted_mean

    # Local/global covariance matrix: C = sum_i w_i * diffs_i * diffs_i^T
    C = np.zeros((d, d))
    for i in range(len(weights)):
        C += weights[i] * np.outer(diffs[i], diffs[i])

    # PatternLocal weights: a = C * w / (w^T * C * w + lambda)
    Cw = C @ lime_weights
    wTCw = lime_weights @ Cw
    var_y = wTCw + lam

    if var_y <= 1e-15:
        return np.zeros(d)

    pattern_weights = Cw / var_y

    return pattern_weights