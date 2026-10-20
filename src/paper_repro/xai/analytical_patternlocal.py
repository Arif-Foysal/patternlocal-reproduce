"""Analytical PatternLocal: fast, hyperparameter-free local explanations.

A drop-in, O(K * D) replacement for LIME-based PatternLocal. Instead of
fitting a local linear surrogate (LIME) and forming the D x D local
covariance, it uses:

1. the exact input gradient of the predicted-class logit (replaces LIME),
   optionally smoothed with SmoothGrad, and
2. the *kernel-weighted* empirical covariance of the K nearest neighbours
   (replaces the kernel-weighted global covariance of paper Eq. (7)).

The explanation is therefore ``a = C_K w`` with
``C_K = sum_i w_i x~_i x~_i^T`` and ``x~_i = x_i - mu`` the centred
K-neighbourhood, ``w_i`` the (normalized) kernel weights. The matrix-vector
product is evaluated as ``X~^T (w * (X~ w))``, avoiding both the O(D^2)
covariance matrix and any O(D^3) solve.

This is the gradient-surrogate special case of paper Eq. (7): the scalar
denominator ``Var[y~] + lambda`` cancels after unit-mass normalisation, so
only the direction of ``C_K w`` matters.
"""

import numpy as np
import torch
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neighbors import NearestNeighbors


def kernel_weights(distances, bandwidth, kernel="gaussian"):
    """Compute kernel weights from distances."""
    u = distances / bandwidth
    if kernel == "gaussian":
        return np.exp(-0.5 * u ** 2)
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1, 0.75 * (1.0 - u ** 2), 0.0)
    if kernel == "uniform":
        return np.ones_like(distances)
    raise ValueError(f"Unknown kernel: {kernel}")


class AnalyticalPatternLocalExplainer:
    """Exact-gradient (+ SmoothGrad) + kernel-weighted KNN-covariance explainer.

    Args:
        model: Differentiable PyTorch model. Moved to ``device`` and set to
            eval mode.
        X_train: Training data, numpy array or torch tensor of shape (N, D).
        k_neighbors: Number of nearest neighbours for the local covariance,
            clamped to ``N``.
        device: ``'cpu'`` or ``'cuda'``.
        normalize: If True, scale each explanation to max-abs 1.
        kernel: Locality kernel ('gaussian', 'epanechnikov', 'uniform').
        bandwidth: Kernel bandwidth. If None, auto-estimated per instance as
            the median distance to its K neighbours.
    """

    def __init__(self, model, X_train, k_neighbors=100, device="cpu",
                 normalize=True, kernel="gaussian", bandwidth=None):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        self.X_train = np.asarray(X_train, dtype=np.float32)
        if self.X_train.ndim != 2:
            raise ValueError("X_train must be 2-D (N, D)")
        self.n_train, self.dim = self.X_train.shape
        self.k_neighbors = int(max(1, min(k_neighbors, self.n_train)))
        self.normalize = normalize
        self.kernel = kernel
        self.bandwidth = bandwidth

        self.knn = NearestNeighbors(
            n_neighbors=self.k_neighbors, metric="euclidean", n_jobs=-1
        )
        self.knn.fit(self.X_train)

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def gradient(self, x_star):
        """Exact gradient of the predicted-class logit at ``x_star``."""
        x = torch.as_tensor(
            x_star, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        x.requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(x)
        if logits.dim() > 1:
            logits = logits[0]
        if logits.numel() > 1:
            target = logits[logits.argmax()]
        else:
            target = logits.reshape(())
        target.backward()
        return x.grad.detach().cpu().numpy().flatten()

    def smoothgrad(self, x_star, n_samples=50, noise_level=0.1, seed=0):
        """SmoothGrad surrogate: mean gradient over Gaussian-perturbed copies.

        Smoother spatial focus than the raw gradient, which helps EMD while
        keeping the false-positive suppression of the covariance transform.

        Args:
            x_star: Instance (D,).
            n_samples: Number of perturbed copies.
            noise_level: Sigma = noise_level * std(x).
            seed: RNG seed.

        Returns:
            Mean gradient (D,).
        """
        rng = np.random.default_rng(seed)
        x_np = np.asarray(x_star, dtype=np.float32).ravel()
        x_std = float(np.std(x_np))
        sigma = noise_level * x_std if x_std > 0 else noise_level
        grads = np.zeros(self.dim, dtype=np.float64)
        for i in range(n_samples):
            eps = rng.normal(0.0, sigma, x_np.shape).astype(np.float32)
            grads += self.gradient(x_np + eps)
        return grads / n_samples

    def integrated_gradients(self, x_star, steps=50, baseline=None, seed=0):
        """Integrated Gradients surrogate: mean gradient along path baseline->x.

        Satisfies completeness; smoother spatial focus than raw gradient, which
        helps EMD while keeping IME low.

        Args:
            x_star: Instance (D,).
            steps: Number of interpolated steps.
            baseline: Baseline instance (D,), defaults to zeros.
            seed: Unused (kept for signature consistency with smoothgrad).

        Returns:
            Integrated gradients (D,).
        """
        x_np = np.asarray(x_star, dtype=np.float32).ravel()
        if baseline is None:
            baseline = np.zeros_like(x_np)
        else:
            baseline = np.asarray(baseline, dtype=np.float32).ravel()
        grads = np.zeros(self.dim, dtype=np.float64)
        for s in range(1, steps + 1):
            alpha = s / steps
            interpolated = baseline + alpha * (x_np - baseline)
            grads += self.gradient(interpolated)
        grads /= steps
        return (x_np - baseline) * grads

    def _bandwidth_for(self, distances):
        if self.bandwidth is not None:
            return float(self.bandwidth)
        bw = float(np.median(distances))
        return bw if bw > 0 else np.finfo(float).eps

    def _attribution(self, w_grad, X_K, weights):
        """a = C_K w  (kernel-weighted local covariance times w_grad).

        Centre-free form: with mu = sum_i v_i x_i and t_i = (x_i - mu) . w,
        sum_i v_i t_i = 0, so C w = sum_i v_i t_i x_i — three BLAS gemvs and
        no N x D temporary (the old X - mu / v[:, None] * X form allocated two
        N x D copies, 2.6 GB at N=40k, D=4096, and upcast to float64).
        """
        weights = np.asarray(weights, dtype=np.float64)
        w_sum = weights.sum()
        if w_sum <= 0:
            raise ValueError("Kernel weights sum to zero. Increase k or bandwidth.")
        v = (weights / w_sum).astype(np.float32)

        a = self._cov_transform(w_grad, X_K, v)

        if self.normalize:
            max_abs = float(np.max(np.abs(a)))
            if max_abs > 0:
                a = a / max_abs
        return a

    def _cov_transform(self, w_grad, X, v):
        """a = X^T (v * (X - mu).w) with mu = v @ X, v normalised, X row order."""
        X32 = X if X.dtype == np.float32 else np.asarray(X, dtype=np.float32)
        v32 = np.asarray(v, dtype=np.float32)
        w = np.asarray(w_grad, dtype=np.float32).ravel()
        mu = v32 @ X32                                   # (D,)
        t = (X32 @ w) - float(mu @ w)                    # (N,)
        return (v32 * t) @ X32                           # (D,)

    def analytical_attr_lasso(self, x_star, grad_vector, dists, idx_nn, k,
                               alpha=0.01):
        """Analytical-IG attribution with L1 Lasso regression.

        Solves: argmin_a ||X_K @ a - y||^2 + alpha * ||a||_1
        where y = X_K @ grad_vector (IG's local linear predictions).
        Input simplification uses identity mapping h(x) = x.

        Args:
            x_star: Instance to explain (D,).
            grad_vector: IG attribution vector (D,).
            dists: Distances to K neighbors (max_k,).
            idx_nn: Indices of K neighbors (max_k,).
            k: Number of neighbors to use.
            alpha: Lasso L1 regularization strength.

        Returns:
            attribution: Analytical-IG-Lasso explanation (D,).
        """
        dists = np.asarray(dists).ravel()
        idx_nn = np.asarray(idx_nn).ravel()
        X_K = self.X_train[idx_nn[:k]]
        dists_k = dists[:k]
        bw = float(np.median(dists_k))
        if bw == 0:
            bw = np.finfo(float).eps
        weights = kernel_weights(dists_k, bw, self.kernel)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Kernel weights sum to zero.")
        weights = weights / weight_sum

        # Target: IG's local linear predictions (identity mapping h(x) = x)
        y = X_K @ grad_vector

        # Standardize features — REQUIRED for Lasso L1 penalty to be fair
        # across features with different scales.
        X_mean = X_K.mean(axis=0)
        X_std = X_K.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_K_std = (X_K - X_mean) / X_std

        y_mean = y.mean()
        y_std = y.std()
        if y_std == 0:
            y_std = 1.0
        y_std_centered = (y - y_mean) / y_std

        # Weighted L1 Lasso regression on standardized data
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000,
                      random_state=42)
        model.fit(X_K_std, y_std_centered, sample_weight=weights)

        # Transform coefficients back to original scale
        attribution = model.coef_ * (y_std / X_std)

        return attribution

    def analytical_attr_elastic_net(self, x_star, grad_vector, dists, idx_nn, k,
                                     alpha=0.0001, l1_ratio=0.1):
        """Analytical-IG attribution with Elastic Net regression (L1 + L2).

        The L2 component encourages grouped selection of correlated pixels,
        producing denser attribution maps than pure Lasso.

        Args:
            x_star: Instance to explain (D,).
            grad_vector: IG attribution vector (D,).
            dists: Distances to K neighbors (max_k,).
            idx_nn: Indices of K neighbors (max_k,).
            k: Number of neighbors to use.
            alpha: Elastic Net overall regularization strength.
            l1_ratio: Mixing parameter (0 = pure Ridge, 1 = pure Lasso).

        Returns:
            attribution: Analytical-IG-ElasticNet explanation (D,).
        """
        dists = np.asarray(dists).ravel()
        idx_nn = np.asarray(idx_nn).ravel()
        X_K = self.X_train[idx_nn[:k]]
        dists_k = dists[:k]
        bw = float(np.median(dists_k))
        if bw == 0:
            bw = np.finfo(float).eps
        weights = kernel_weights(dists_k, bw, self.kernel)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Kernel weights sum to zero.")
        weights = weights / weight_sum

        # Target: IG's local linear predictions (identity mapping h(x) = x)
        y = X_K @ grad_vector

        # Standardize features
        X_mean = X_K.mean(axis=0)
        X_std = X_K.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_K_std = (X_K - X_mean) / X_std

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
        model.fit(X_K_std, y_std_centered, sample_weight=weights)

        # Transform coefficients back to original scale
        attribution = model.coef_ * (y_std / X_std)

        return attribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, x_star, surrogate="gradient", k=None, bandwidth=None,
                kernel=None, **surrogate_kwargs):
        """Return the attribution vector (D,) for a single instance.

        Args:
            surrogate: 'gradient' (exact input gradient) or 'smoothgrad'
                (mean gradient over Gaussian perturbations).
            k: Number of neighbours (default self.k_neighbors).
            bandwidth: Kernel bandwidth (default self.bandwidth; None = auto).
            kernel: Kernel name (default self.kernel).
            surrogate_kwargs: passed to ``smoothgrad`` when
                surrogate='smoothgrad'.
        """
        k = int(k if k is not None else self.k_neighbors)
        k = max(1, min(k, self.n_train))
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        ker = kernel if kernel is not None else self.kernel

        if surrogate == "gradient":
            w_grad = self.gradient(x_star)
        elif surrogate == "smoothgrad":
            w_grad = self.smoothgrad(x_star, **surrogate_kwargs)
        else:
            raise ValueError(f"Unknown surrogate: {surrogate}")

        x_np = np.asarray(x_star, dtype=np.float32).reshape(1, -1)
        distances, indices = self.knn.kneighbors(x_np, n_neighbors=k)
        X_K = self.X_train[indices[0]]
        dists = distances[0]
        if bw is None:
            bw = self._bandwidth_for(dists)
        weights = kernel_weights(dists, bw, ker)
        return self._attribution(w_grad, X_K, weights)

    def explain_batch(self, X_batch, surrogate="gradient", k=None,
                      bandwidth=None, kernel=None, **surrogate_kwargs):
        """Return attributions (B, D) for a batch of instances."""
        X_batch = np.asarray(X_batch, dtype=np.float32)
        k = int(k if k is not None else self.k_neighbors)
        k = max(1, min(k, self.n_train))
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        ker = kernel if kernel is not None else self.kernel

        distances, indices = self.knn.kneighbors(X_batch, n_neighbors=k)
        out = np.empty((len(X_batch), self.dim), dtype=np.float64)
        for i in range(len(X_batch)):
            if surrogate == "gradient":
                w_grad = self.gradient(X_batch[i])
            elif surrogate == "smoothgrad":
                w_grad = self.smoothgrad(X_batch[i], **surrogate_kwargs)
            else:
                raise ValueError(f"Unknown surrogate: {surrogate}")
            X_K = self.X_train[indices[i]]
            dists = distances[i]
            if bw is None:
                bw = self._bandwidth_for(dists)
            weights = kernel_weights(dists, bw, ker)
            out[i] = self._attribution(w_grad, X_K, weights)
        return out


def analytical_patternlocal_explain(explainer, x_star, **kwargs):
    """Functional wrapper around :class:`AnalyticalPatternLocalExplainer`."""
    return explainer.explain(x_star, **kwargs)
