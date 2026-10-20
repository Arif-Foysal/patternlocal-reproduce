"""Faithful PatternLocal implementation following Eq. (7) of the paper.

Paper Eq. (7): a_l2 = Cov_Pi[x, y~] / (Var_Pi[y~] + lambda)
where y~ = w^T x, expectations are kernel-weighted over local neighborhood.

This implementation uses the local covariance matrix formulation:
a = C * w / (w^T * C * w + lambda)
where C = Cov_Pi[x, x] is the local covariance matrix.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Lasso, ElasticNet


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
    k_ratio=1.0,
    lam=1e-5,
    use_global_cov=False,
    shrinkage=0.0,
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
        shrinkage: Ledoit-Wolf-style shrinkage intensity in [0, 1]. Shrinks the
                       local covariance toward avg_variance * I for stability in
                       high dimensions (matches upstream PatternLocal).

    Returns:
        pattern_weights: PatternLocal explanation (D,)
    """
    n_total, d = X_train.shape
    X_f32 = X_train if X_train.dtype == np.float32 else np.asarray(X_train, dtype=np.float32)

    # Kernel weights in *training-data row order* (zeros outside the local set),
    # so no gather/copy of the N x D matrix is needed for any k_ratio.
    w_full = np.zeros(n_total, dtype=np.float64)

    if use_global_cov:
        # Global covariance: uniform weights over all training data
        w_full[:] = 1.0 / n_total
        bandwidth = np.inf  # informational; weights are uniform
    else:
        k = max(1, int(k_ratio * n_total))
        # BLAS-backed (float32, no copy of X_train) — ~2x faster than cdist;
        # squared-distance cancellation error ~1e-7 at unit-norm scale.
        instance_2d = np.asarray(instance, dtype=np.float32).reshape(1, -1)
        distances = euclidean_distances(X_f32, instance_2d).ravel()
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
        w_full[k_idx] = weights / weight_sum

    # --- a = C w / (w^T C w + lambda)  WITHOUT forming C ---------------------
    # C = sum_i v_i (x_i - mu)(x_i - mu)^T  with mu = sum_i v_i x_i, so
    #   t_i   = (x_i - mu) . w
    #   Cw    = sum_i v_i t_i x_i      (because sum_i v_i t_i = 0)
    # Three BLAS gemvs over N x D instead of an O(N D^2) matmul: ~4000x cheaper
    # and it never allocates an N x D temporary (1.3 GB at N=40k, D=4096).
    v = w_full.astype(np.float32)
    w = np.asarray(lime_weights, dtype=np.float32).ravel()
    mu = v @ X_f32                                     # (D,)
    t = (X_f32 @ w) - float(mu @ w)                    # (N,)
    q = v * t
    Cw = q @ X_f32                                     # (D,)
    var_y = float(w @ Cw)                              # w^T C w

    if shrinkage and shrinkage > 0:
        # avg_variance = mean_j diag(C)_j ; diag(C) = sum_i v_i x_ij^2 - mu_j^2
        acc = np.zeros(d, dtype=np.float64)
        block = 4096
        for start in range(0, n_total, block):
            blk = X_f32[start:start + block].astype(np.float64)
            acc += w_full[start:start + block] @ (blk * blk)
        avg_variance = float(np.mean(acc - mu.astype(np.float64) ** 2))
        Cw = (1.0 - shrinkage) * Cw + shrinkage * avg_variance * w
        var_y = float(w @ Cw)

    var_y = var_y + lam
    if var_y <= 1e-15:
        return np.zeros(d)

    pattern_weights = Cw.astype(np.float64) / var_y

    return pattern_weights


def patternlocal_lasso_explain(
    lime_weights,
    lime_intercept,
    instance,
    X_train,
    bandwidth=None,
    kernel="gaussian",
    k_ratio=0.05,
    alpha=0.01,
    use_global_cov=False,
):
    """PatternLocal with L1 Lasso regression instead of closed-form Ridge.

    Solves: argmin_a ||X_local @ a - y||^2 + alpha * ||a||_1
    where y = X_local @ lime_weights + lime_intercept (LIME's local predictions).
    Input simplification uses identity mapping h(x) = x.

    Args:
        lime_weights: LIME coefficient vector (D,)
        lime_intercept: LIME intercept (scalar)
        instance: Instance to explain (D,)
        X_train: Training data (N, D)
        bandwidth: Kernel bandwidth. If None, auto-estimate as median distance.
        kernel: Kernel function ('gaussian' or 'epanechnikov')
        k_ratio: Ratio of samples to use for local neighborhood (0, 1]
        alpha: Lasso L1 regularization strength
        use_global_cov: If True, use global covariance (all data, uniform weights).

    Returns:
        pattern_weights: PatternLocal-Lasso explanation (D,)
    """
    n_total, d = X_train.shape

    if use_global_cov:
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

    # Target: LIME's local linear predictions (identity mapping h(x) = x)
    y = X_local @ lime_weights + lime_intercept

    # Standardize features — REQUIRED for Lasso L1 penalty to be fair across
    # features with different scales (our X data has tiny range ~[-0.004, 0.06]).
    X_mean = X_local.mean(axis=0)
    X_std = X_local.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_local_std = (X_local - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_std_centered = (y - y_mean) / y_std

    # Weighted L1 Lasso regression on standardized data
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000,
                  random_state=42)
    model.fit(X_local_std, y_std_centered, sample_weight=weights)

    # Transform coefficients back to original scale
    pattern_weights = model.coef_ * (y_std / X_std)

    return pattern_weights


def patternlocal_elastic_net_explain(
    lime_weights,
    lime_intercept,
    instance,
    X_train,
    bandwidth=None,
    kernel="gaussian",
    k_ratio=0.05,
    alpha=0.0001,
    l1_ratio=0.1,
    use_global_cov=False,
):
    """PatternLocal with Elastic Net regression (L1 + L2).

    Elastic Net: argmin_a ||X_local @ a - y||^2 + alpha * (
        l1_ratio * ||a||_1 + 0.5 * (1 - l1_ratio) * ||a||_2^2
    )

    The L2 component encourages grouped selection of correlated features
    (pixels), producing denser attribution maps than pure Lasso.
    Input simplification uses identity mapping h(x) = x.

    Args:
        lime_weights: LIME coefficient vector (D,)
        lime_intercept: LIME intercept (scalar)
        instance: Instance to explain (D,)
        X_train: Training data (N, D)
        bandwidth: Kernel bandwidth. If None, auto-estimate as median distance.
        kernel: Kernel function ('gaussian' or 'epanechnikov')
        k_ratio: Ratio of samples to use for local neighborhood (0, 1]
        alpha: Elastic Net overall regularization strength
        l1_ratio: Mixing parameter (0 = pure Ridge, 1 = pure Lasso).
                  Lower values = denser solutions.
        use_global_cov: If True, use global covariance (all data, uniform weights).

    Returns:
        pattern_weights: PatternLocal-ElasticNet explanation (D,)
    """
    n_total, d = X_train.shape

    if use_global_cov:
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

    # Target: LIME's local linear predictions (identity mapping h(x) = x)
    y = X_local @ lime_weights + lime_intercept

    # Standardize features
    X_mean = X_local.mean(axis=0)
    X_std = X_local.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_local_std = (X_local - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_std_centered = (y - y_mean) / y_std

    # Weighted Elastic Net regression on standardized data
    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
        max_iter=10000, random_state=42,
    )
    model.fit(X_local_std, y_std_centered, sample_weight=weights)

    # Transform coefficients back to original scale
    pattern_weights = model.coef_ * (y_std / X_std)

    return pattern_weights